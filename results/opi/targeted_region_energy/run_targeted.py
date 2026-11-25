import os,sys,json,time
import numpy as np,pandas as pd
from scipy.stats import wilcoxon
BASE=r"C:\Users\damsw\Desktop\spectral_injection"; OUT=os.path.join(BASE,"results","opi","targeted_region_energy")
os.chdir(BASE);sys.path.insert(0,BASE)
from spectral_injection.opi import dataset as ds,capture as cm,templates as tm
logf=open(os.path.join(OUT,"run.log"),"w",encoding="utf8",buffering=1)
def log(*x):
 s=" ".join(map(str,x));print(s,flush=True);logf.write(s+"\n")
MODEL="Qwen/Qwen2.5-1.5B-Instruct";SEED=2027
TASKS=["gigaword","hsol","sentiment","spam"]; ATTACKS=["combine","escape","ignore","naive"]
REG=["task","normal","preamble","slot","tail","template"]; RP=[(REG[i],REG[j]) for i in range(6) for j in range(i,6)]
df,_=ds.load(os.path.join(BASE,"data","opi","data","train-00000-of-00001.parquet"))
template,_=tm.resolve(MODEL);cap=cm.Capturer(MODEL,dtype="float32",max_tokens=512);tm.verify(cap.tokenizer,template)
log("model",MODEL,"layers",cap.n_layers,"heads",cap.n_heads,"device",cap.device)
rng=np.random.default_rng(SEED);sample=[]
for a in ATTACKS:
 for t in TASKS:
  g=df[(df.attack_type==a)&(df.task_type==t)].reset_index(drop=True);sample.append(g.iloc[int(rng.integers(len(g)))])
sample=pd.DataFrame(sample).reset_index(drop=True); filler=ds.BenignFiller(df,seed=SEED)
def nt(i,e):
 txt=cm.format_user_only(cap.tokenizer,cm.build_user_turn(i,e),template)
 return len(cap.tokenizer(txt,add_special_tokens=False)["input_ids"])
def regions(seg,instruction,comps):
 out=np.full(len(seg),"template",dtype="<U10");u0=seg.user_start;u1=u0+len(instruction);e0=u1+2
 ranges=[];p=e0
 for name,txt in comps:
  ranges.append((name,p,p+len(txt)));p+=len(txt)
 e1=e0+len("".join(x[1] for x in comps).rstrip())
 for k,(lo,hi) in enumerate(seg.offsets):
  m=(lo+hi)/2 if hi>lo else float(lo)
  if u0<=m<u1:out[k]="task"
  elif e0<=m<e1:
   for name,x,y in ranges:
    if x<=m<min(y,e1):out[k]=name;break
 return out
def calc(att,lab):
 L=len(att);H=att[0].shape[0];N=len(lab);idx={r:np.flatnonzero(lab==r) for r in REG}
 en=np.zeros((L,H,len(RP)));den=np.zeros_like(en);var=np.zeros((L,H));loop=np.zeros_like(var);flow=np.zeros((L,H,6,6))
 for l,A0 in enumerate(att):
  A=np.asarray(A0,dtype=np.float64);W=.5*(A+A.transpose(0,2,1));d=W.sum(-1)
  iv=np.divide(1,np.sqrt(np.maximum(d,1e-12)),out=np.zeros_like(d),where=d>1e-12);M=W*iv[:,:,None]*iv[:,None,:]
  diag=np.diagonal(M,axis1=1,axis2=2);Q=M*M;ii=np.arange(N);Q[:,ii,ii]=0
  loop[l]=((diag-diag.mean(axis=1,keepdims=True))**2).sum(axis=1)/N;var[l]=Q.sum((1,2))/N+loop[l]
  for k,(r,s) in enumerate(RP):
   x,y=idx[r],idx[s]
   if not len(x) or not len(y):continue
   z=Q[:,x[:,None],y].sum((1,2))
   poss=len(x)*(len(x)-1) if r==s else 2*len(x)*len(y)
   if r!=s:z+=Q[:,y[:,None],x].sum((1,2))
   en[l,:,k]=z/N;den[l,:,k]=z/max(poss,1)
  for i,r in enumerate(REG):
   x=idx[r]
   if not len(x):continue
   for j,s in enumerate(REG):
    y=idx[s]
    if len(y):flow[l,:,i,j]=A[:,x,:][:,:,y].sum((1,2))/len(x)
 return en,den,var,loop,flow
records=[];start=time.time()
for pi,row in sample.iterrows():
 ext4=ds.external_C4(row);base=nt(row.instruction,ds.external_C2(row));target=nt(row.instruction,ext4)-base
 fill,fd=filler.fit(row,nt,target)
 if fill is None:raise RuntimeError("Filler failed "+str(row.uid))
 ext3=ds.external_C3(row,fill)
 if nt(row.instruction,ext3)!=nt(row.instruction,ext4):raise RuntimeError("length mismatch "+str(row.uid))
 comps={"C3":[("normal",row.normal_input),("preamble",row.P_atk),("slot",fill),("tail",row.D_inj)],
        "C4":[("normal",row.normal_input),("preamble",row.P_atk),("slot",row.I_inj),("tail",row.D_inj)]}
 pid=f"{row.attack_type}__{row.task_type}__{int(row.sample_id)}"; rec={"pairid":pid,"attack_type":row.attack_type,"task_type":row.task_type,"sample_id":int(row.sample_id),"token_count":nt(row.instruction,ext3)}
 for c,e in [("C3",ext3),("C4",ext4)]:
  seg=cm.segment(cap.tokenizer,row.instruction,e,template); audit=cm.verify_segmentation(seg);lab=regions(seg,row.instruction,comps[c])
  if len(seg)>512:raise RuntimeError("too long "+pid)
  att=cap.forward(seg,check=True); z=calc(att,lab)
  folder=os.path.join(OUT,"raw",pid);os.makedirs(folder,exist_ok=True)
  np.savez_compressed(os.path.join(folder,c+".npz"),**{f"layer_{l:02d}":a.astype(np.float32) for l,a in enumerate(att)},token_ids=np.array(seg.input_ids,np.int32),tokens=np.array(seg.tokens,str),regions=lab)
  rec[c]={"energy":z[0].mean((0,1)),"density":z[1].mean((0,1)),"variance":z[2].mean(),"loop":z[3].mean(),"flow":z[4].mean((0,1)),"layer_energy":z[0].mean(1),"layer_variance":z[2].mean(1),"layer_loop":z[3].mean(1),"counts":{r:int((lab==r).sum()) for r in REG}}
  log(f"[{pi+1}/16] {pid} {c} tokens={len(seg)} counts={rec[c]['counts']} seg_ok={audit['ok']} elapsed={time.time()-start:.1f}s")
  del att
 rec["filler_matched"]=bool(fd["matched"]);records.append(rec)
log("all prompts captured; analyze paired deltas")
N=len(records);K=len(RP);L=cap.n_layers
E=np.stack([[r[c]["layer_energy"] for c in ("C3","C4")] for r in records])
V=np.stack([[r[c]["layer_variance"] for c in ("C3","C4")] for r in records])
LO=np.stack([[r[c]["layer_loop"] for c in ("C3","C4")] for r in records])
D=np.stack([[r[c]["density"] for c in ("C3","C4")] for r in records])
F=np.stack([[r[c]["flow"] for c in ("C3","C4")] for r in records])
dEl=E[:,1]-E[:,0];de=dEl.mean(1);dd=D[:,1]-D[:,0];dv=(V[:,1]-V[:,0]).mean(1);dl=(LO[:,1]-LO[:,0]).mean(1);dflo=F[:,1]-F[:,0]
def boot(X,seed=2027,B=10000):
 rr=np.random.default_rng(seed); groups={a:[i for i,r in enumerate(records) if r["attack_type"]==a] for a in ATTACKS}
 out=np.empty((B,)+X.shape[1:])
 for b in range(B):
  ii=np.concatenate([rr.choice(g,len(g),replace=True) for g in groups.values()]);out[b]=X[ii].mean(0)
 return out
def bh(p):
 p=np.array(p);order=np.argsort(p);q=np.empty(len(p));run=1.
 for rank in range(len(p)-1,-1,-1):
  i=order[rank];run=min(run,p[i]*len(p)/(rank+1));q[i]=run
 return q
be=boot(de);bd=boot(dd,2028);bf=boot(dflo,2029);me=de.mean(0);md=dd.mean(0);mf=dflo.mean(0)
rows=[];ps=[]
for k,(a,b) in enumerate(RP):
 x=de[:,k];p=float(wilcoxon(x).pvalue) if np.any(x) else 1.;ps.append(p)
 c3=np.mean([r["C3"]["energy"][k] for r in records])
 share=np.mean([r["C4"]["energy"][k]/max(r["C4"]["energy"].sum(),1e-30)-r["C3"]["energy"][k]/max(r["C3"]["energy"].sum(),1e-30) for r in records])
 rows.append({"region_pair":a+"--"+b,"delta_energy_per_token":me[k],"ci95_low":np.quantile(be[:,k],.025),"ci95_high":np.quantile(be[:,k],.975),"delta_mean_sq_weight_per_possible_edge":md[k],"delta_energy_share":share,"wilcoxon_p":p,"BH_q":np.nan})
q=bh(ps)
for i,x in enumerate(q):rows[i]["BH_q"]=x
tab=pd.DataFrame(rows).sort_values("delta_energy_per_token",key=lambda x:abs(x),ascending=False);tab.to_csv(os.path.join(OUT,"region_pair_energy.csv"),index=False)
flows=[];pfs=[]
for i,a in enumerate(REG):
 for j,b in enumerate(REG):
  x=dflo[:,i,j];p=1.0 if not np.any(x) else float(wilcoxon(x).pvalue);pfs.append(p)
  flows.append({"query_region":a,"key_region":b,"delta_attention_mass":mf[i,j],"ci95_low":np.quantile(bf[:,i,j],.025),"ci95_high":np.quantile(bf[:,i,j],.975),"wilcoxon_p":p,"BH_q":np.nan})
qf=bh(pfs)
for i,x in enumerate(qf):flows[i]["BH_q"]=x
ft=pd.DataFrame(flows).sort_values("delta_attention_mass",key=lambda x:abs(x),ascending=False);ft.to_csv(os.path.join(OUT,"directed_attention_region_flows.csv"),index=False)
lr=[]
for k,(a,b) in enumerate(RP):
 for l in range(L):lr.append({"layer":l,"region_pair":a+"--"+b,"delta_energy_per_token":dEl[:,l,k].mean(),"delta_primary_variance":(V[:,1,l]-V[:,0,l]).mean(),"delta_loop_component":(LO[:,1,l]-LO[:,0,l]).mean()})
pd.DataFrame(lr).to_csv(os.path.join(OUT,"layer_region_energy.csv"),index=False)
first=records[0]["pairid"];raw=np.load(os.path.join(OUT,"raw",first,"C4.npz"));A=raw["layer_12"][0].astype(float);W=.5*(A+A.T);deg=W.sum(1);iv=1/np.sqrt(np.maximum(deg,1e-12));M=W*iv[:,None]*iv[None,:];Lm=np.eye(len(deg))-M;eig=np.linalg.eigvalsh((Lm+Lm.T)/2);di=np.diag(M);off=M.copy();np.fill_diagonal(off,0);resid=abs(np.var(eig)-((off**2).sum()+((di-di.mean())**2).sum())/len(di))
meta={"model":MODEL,"seed":SEED,"n_pairs":N,"strata":{a:4 for a in ATTACKS},"tasks":TASKS,"conditions":["C3","C4"],"token_lengths_exact":all(r["token_count"]>0 for r in records),"all_fillers_matched":all(r["filler_matched"] for r in records),"region_order":REG,"unit_of_inference":"prompt pair; layer/head averaged within prompt","bootstrap":"10000 stratified resamples, preserving attack type counts","graph":"W=(A+A.T)/2, self loops retained; region energy = off-diagonal M_ij^2/N","identity_residual":float(resid),"scope":"exploratory 16 pairs, one model"}
json.dump(meta,open(os.path.join(OUT,"manifest.json"),"w",encoding="utf8"),indent=2)
metarows=[]
for r in records:
 metarows.append({k:v for k,v in r.items() if k not in ("C3","C4")} | {"C3_tokens_by_region":r["C3"]["counts"],"C4_tokens_by_region":r["C4"]["counts"]})
pd.DataFrame(metarows).to_json(os.path.join(OUT,"pair_manifest.json"),orient="records",force_ascii=False,indent=2)
def fmt(x):return f"{float(x):+.4g}"
rep=["# Recapture C3/C4 : énergie d’attention par régions","",f"- {MODEL}; {N} paires, 4 exemples par type d’attaque; tâches : {', '.join(TASKS)}.","- Longueurs C3/C4 exactement appariées; captures brutes vérifiées causalement et normalisées.","- Unité statistique : prompt; agrégation des 28 couches et 12 têtes; bootstrap stratifié 10 000 fois.","- Tests des blocs exploratoires, correction BH sur 21 blocs et 36 flux dirigés.",f"- Résidu de l’identité variance-spectre sur une matrice : {resid:.3e}.","","![Carte thermique des différences C4 moins C3](region_energy_heatmap.png)\\n\\n## Effet général","",f"- Δ variance spectrale primaire : {dv.mean():+.6g}.",f"- Δ énergie des liens hors diagonale par token : {de.sum(1).mean():+.6g}.",f"- Δ contribution des boucles par token : {dl.mean():+.6g}.","","## Plus fortes variations des blocs","", "| Bloc | Δ énergie/token | IC 95 % | Δ poids carré moyen/lien possible | q BH |","|---|---:|---:|---:|---:|"]
for _,z in tab.head(8).iterrows():rep.append(f"| {z.region_pair} | {fmt(z.delta_energy_per_token)} | [{fmt(z.ci95_low)}, {fmt(z.ci95_high)}] | {fmt(z.delta_mean_sq_weight_per_possible_edge)} | {z.BH_q:.3g} |")
rep += ["","","## Plus fortes variations de flux dirigés","","| Requête → clé | Δ masse attention/token requête | IC 95 % | q BH |","|---|---:|---:|---:|"]
for _,z in ft.head(8).iterrows():rep.append(f"| {z.query_region} → {z.key_region} | {fmt(z.delta_attention_mass)} | [{fmt(z.ci95_low)}, {fmt(z.ci95_high)}] | {z.BH_q:.3g} |")
rep += ["","","## Limites","","Pilote exploratoire : 16 paires, un modèle. Pour ignore/combine, le préambule conservé dans C3 contient déjà une composante adversariale; interpréter les effets par strate. La région slot désigne le filler C3 ou l’instruction injectée C4. Les liens sont symétrisés comme dans le graphe spectral; les flux dirigés sont reportés séparément."]
open(os.path.join(OUT,"RAPPORT_C3_C4_regions.md"),"w",encoding="utf8").write("\n".join(rep))
log("COMPLETE",OUT,"identity_resid",resid)
log("Top graph blocks")
for _,z in tab.head(8).iterrows():log(z.region_pair,fmt(z.delta_energy_per_token),"CI",fmt(z.ci95_low),fmt(z.ci95_high),"q",z.BH_q)
log("Top directed flows")
for _,z in ft.head(8).iterrows():log(z.query_region+"->"+z.key_region,fmt(z.delta_attention_mass),"CI",fmt(z.ci95_low),fmt(z.ci95_high),"q",z.BH_q)
logf.close()
