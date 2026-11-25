import os,json,numpy as np,pandas as pd
from scipy.stats import wilcoxon
OUT=r"C:\Users\damsw\Desktop\spectral_injection\results\opi\targeted_region_energy"
REG=["task","normal","preamble","slot","tail","template"]
RP=[(REG[i],REG[j]) for i in range(6) for j in range(i,6)]
ATTACKS=["combine","escape","ignore","naive"]
def calc(att,lab):
 L=len(att);H=att[0].shape[0];N=len(lab);idx={r:np.flatnonzero(lab==r) for r in REG}
 en=np.zeros((L,H,len(RP)));den=np.zeros_like(en);var=np.zeros((L,H));loop=np.zeros_like(var);flow=np.zeros((L,H,6,6));Pall=np.zeros((L,H))
 for l,A0 in enumerate(att):
  A=np.asarray(A0,dtype=float)
  if not np.isfinite(A).all() or np.max(np.triu(A,1))>1e-4 or np.max(np.abs(A.sum(-1)-1))>1e-4:raise RuntimeError("attention integrity failed")
  W=.5*(A+A.transpose(0,2,1));d=W.sum(-1);iv=np.divide(1,np.sqrt(np.maximum(d,1e-12)),out=np.zeros_like(d),where=d>1e-12)
  M=W*iv[:,:,None]*iv[:,None,:];diag=np.diagonal(M,axis1=1,axis2=2);Q=M*M;ii=np.arange(N);Q[:,ii,ii]=0
  P=Q.sum((1,2))/N;Pall[l]=P;loop[l]=((diag-diag.mean(1,keepdims=True))**2).sum(1)/N;var[l]=P+loop[l]
  for k,(r,s) in enumerate(RP):
   x,y=idx[r],idx[s]
   if not len(x) or not len(y):continue
   z=Q[:,x[:,None],y].sum((1,2));poss=len(x)*(len(x)-1) if r==s else 2*len(x)*len(y)
   if r!=s:z+=Q[:,y[:,None],x].sum((1,2))
   en[l,:,k]=z/N;den[l,:,k]=z/max(poss,1)
  for i,r in enumerate(REG):
   x=idx[r]
   if not len(x):continue
   for j,s in enumerate(REG):
    y=idx[s]
    if len(y):flow[l,:,i,j]=A[:,x,:][:,:,y].sum((1,2))/len(x)
 if not np.allclose(en.sum(-1),Pall,rtol=1e-8,atol=1e-10):raise RuntimeError('regional blocks do not partition total edge energy')
 return en,den,var,loop,flow
folders=sorted([p for p in os.listdir(os.path.join(OUT,"raw")) if os.path.isdir(os.path.join(OUT,"raw",p))])
if len(folders)!=16:raise RuntimeError(f"Expected 16 captured prompts, got {len(folders)}")
records=[]
for pid in folders:
 r={"pairid":pid,"attack_type":pid.split("__")[0]}
 for c in ["C3","C4"]:
  z=np.load(os.path.join(OUT,"raw",pid,c+".npz"),allow_pickle=False)
  layers=sorted([k for k in z.files if k.startswith("layer_")])
  lab=z["regions"]; ids=z["token_ids"]
  if len(layers)!=28 or len(lab)!=len(ids):raise RuntimeError("shape mismatch "+pid+c)
  r[c]=calc([z[k] for k in layers],lab)
  r[c+"_n"]=len(ids);r[c+"_counts"]={x:int((lab==x).sum()) for x in REG}
 if r["C3_n"]!=r["C4_n"]:raise RuntimeError("C3/C4 token mismatch "+pid)
 records.append(r)
print("loaded",len(records),"pairs; all lengths equal")
N=len(records);L=28;K=len(RP);R=len(REG)
E=np.stack([[r[c][0].mean(1) for c in ["C3","C4"]] for r in records]) # N,2,L,K
D=np.stack([[r[c][1].mean((0,1)) for c in ["C3","C4"]] for r in records]) # N,2,K
VLayer=np.stack([[r[c][2].mean(1) for c in ["C3","C4"]] for r in records]);LOLayer=np.stack([[r[c][3].mean(1) for c in ["C3","C4"]] for r in records])
F=np.stack([[r[c][4].mean((0,1)) for c in ["C3","C4"]] for r in records])
dEl=E[:,1]-E[:,0];dE=dEl.mean(1);dD=D[:,1]-D[:,0];V=VLayer.mean(2);LO=LOLayer.mean(2);dV=V[:,1]-V[:,0];dLO=LO[:,1]-LO[:,0];dVLayer=VLayer[:,1]-VLayer[:,0];dLOLayer=LOLayer[:,1]-LOLayer[:,0];dF=F[:,1]-F[:,0]
def boot(X,seed=2027,B=10000):
 rng=np.random.default_rng(seed);groups={a:[i for i,r in enumerate(records) if r["attack_type"]==a] for a in ATTACKS}
 out=np.empty((B,)+X.shape[1:])
 for b in range(B):
  ii=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups.values()]);out[b]=X[ii].mean(0)
 return out
def bh(p):
 p=np.array(p);o=np.argsort(p);q=np.empty(len(p));v=1.
 for rank in range(len(p)-1,-1,-1):
  i=o[rank];v=min(v,p[i]*len(p)/(rank+1));q[i]=v
 return q
bE=boot(dE);bD=boot(dD,2028);bF=boot(dF,2029)
rows=[];ps=[]
for k,(a,b) in enumerate(RP):
 p=float(wilcoxon(dE[:,k]).pvalue);ps.append(p)
 base=np.mean([r["C3"][0][:,:,k].mean() for r in records])
 share=np.mean([r["C4"][0][:,:,k].sum()/max(r["C4"][0].sum(),1e-30)-r["C3"][0][:,:,k].sum()/max(r["C3"][0].sum(),1e-30) for r in records])
 rows.append({"region_pair":a+"--"+b,"delta_energy_per_token":dE[:,k].mean(),"ci95_low":np.quantile(bE[:,k],.025),"ci95_high":np.quantile(bE[:,k],.975),"delta_mean_sq_weight_per_possible_edge":dD[:,k].mean(),"delta_energy_share":share,"wilcoxon_p":p,"BH_q":np.nan})
q=bh(ps)
for i,x in enumerate(q):rows[i]["BH_q"]=x
tab=pd.DataFrame(rows).sort_values("delta_energy_per_token",key=lambda x:abs(x),ascending=False);tab.to_csv(os.path.join(OUT,"region_pair_energy.csv"),index=False)
flows=[];pf=[]
for i,a in enumerate(REG):
 for j,b in enumerate(REG):
  p=1.0 if not np.any(dF[:,i,j]) else float(wilcoxon(dF[:,i,j]).pvalue);pf.append(p)
  flows.append({"query_region":a,"key_region":b,"delta_attention_mass":dF[:,i,j].mean(),"ci95_low":np.quantile(bF[:,i,j],.025),"ci95_high":np.quantile(bF[:,i,j],.975),"wilcoxon_p":p,"BH_q":np.nan})
qf=bh(pf)
for i,x in enumerate(qf):flows[i]["BH_q"]=x
ft=pd.DataFrame(flows).sort_values("delta_attention_mass",key=lambda x:abs(x),ascending=False);ft.to_csv(os.path.join(OUT,"directed_attention_region_flows.csv"),index=False)
lr=[]
for k,(a,b) in enumerate(RP):
 for l in range(L):lr.append({"layer":l,"region_pair":a+"--"+b,"delta_energy_per_token":dEl[:,l,k].mean(),"delta_primary_variance":dVLayer[:,l].mean(),"delta_loop_component":dLOLayer[:,l].mean()})
pd.DataFrame(lr).to_csv(os.path.join(OUT,"layer_region_energy.csv"),index=False)
# Validate spectral variance identity on one real attention head.
z=np.load(os.path.join(OUT,"raw",folders[0],"C4.npz"));A=z["layer_12"][0].astype(float);W=.5*(A+A.T);d=W.sum(1);iv=1/np.sqrt(np.maximum(d,1e-12));M=W*iv[:,None]*iv[None,:];Lm=np.eye(len(d))-M;ev=np.linalg.eigvalsh((Lm+Lm.T)/2);di=np.diag(M);off=M.copy();np.fill_diagonal(off,0)
resid=abs(np.var(ev)-((off**2).sum()+((di-di.mean())**2).sum())/len(d))
meta={"model":"Qwen/Qwen2.5-1.5B-Instruct","n_pairs":N,"conditions":["C3","C4"],"token_lengths_equal":True,"strata_counts":{a:sum(r["attack_type"]==a for r in records) for a in ATTACKS},"region_order":REG,"unit":"prompt pair; heads/layers aggregated within prompt","bootstrap":"10000 stratified resamples by attack_type","variance_identity_residual":float(resid),"scope":"exploratory"}
json.dump(meta,open(os.path.join(OUT,"manifest.json"),"w",encoding="utf8"),indent=2)
pd.DataFrame([{"pairid":r["pairid"],"attack_type":r["attack_type"],"tokens":r["C3_n"],"C3_counts":r["C3_counts"],"C4_counts":r["C4_counts"]} for r in records]).to_json(os.path.join(OUT,"pair_manifest.json"),orient="records",force_ascii=False,indent=2)
def fmt(x):return f"{float(x):+.4g}"
rep=["# Recapture C3/C4 : énergie d’attention par régions","",f"- Qwen 2.5 1.5B; {N} paires; 4 exemples par type d’attaque; tâches gigaword, hsol, sentiment, spam.","- C3/C4 appariés exactement en longueur; captures brutes vérifiées causalement; 28 couches × 12 têtes.","- Unité statistique : prompt; bootstrap stratifié par type d’attaque, 10 000 rééchantillonnages.","- Tests exploratoires; BH sur 21 blocs régionaux et 36 flux dirigés.",f"- Résidu de l’identité variance-spectre sur une matrice : {resid:.3e}.","","## Effet général","",f"- Δ variance spectrale primaire : {dV.mean():+.6g}.",f"- Δ énergie hors diagonale par token : {dE.sum(1).mean():+.6g}.",f"- Δ contribution des boucles propres : {dLO.mean():+.6g}.","","## Blocs de régions avec les plus fortes variations absolues","","| Bloc | Δ énergie/token | IC 95 % | Δ poids carré moyen/lien possible | q BH |","|---|---:|---:|---:|---:|"]
for _,x in tab.head(8).iterrows():rep.append(f"| {x.region_pair} | {fmt(x.delta_energy_per_token)} | [{fmt(x.ci95_low)}, {fmt(x.ci95_high)}] | {fmt(x.delta_mean_sq_weight_per_possible_edge)} | {x.BH_q:.3g} |")
rep+=["","","## Flux dirigés d’attention","","| Région requête → région clé | Δ masse moyenne C4−C3 | IC 95 % | q BH |","|---|---:|---:|---:|"]
for _,x in ft.head(8).iterrows():rep.append(f"| {x.query_region} → {x.key_region} | {fmt(x.delta_attention_mass)} | [{fmt(x.ci95_low)}, {fmt(x.ci95_high)}] | {x.BH_q:.3g} |")
rep+=["","","## Limites","","Pilote exploratoire : 16 paires, un modèle. Pour les attaques ignore/combine, le préambule conservé dans C3 contient déjà une composante adversariale; interpréter les effets par strate. La région slot désigne l’instruction injectée en C4 et le filler en C3."]
open(os.path.join(OUT,"RAPPORT_C3_C4_regions.md"),"w",encoding="utf8").write("\n".join(rep))
print("identity_residual",resid,"primary_delta",dV.mean(),"offdiag_delta",dE.sum(1).mean(),"loop_delta",dLO.mean())
print("TOP BLOCKS")
print(tab.head(10).to_string(index=False))
print("TOP FLOWS")
print(ft.head(10).to_string(index=False))
