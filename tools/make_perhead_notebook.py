"""Genere notebooks/02_perhead_spectral_trust.ipynb.

Construit plutot qu'ecrit a la main : le JSON d'un .ipynb est trop fragile.
Le notebook importe ``spectral_injection.perhead`` (code support, teste
depuis le shell) et ``spectral_trust`` ; les cellules orchestrent, affichent
et journalisent.

    python tools/make_perhead_notebook.py
    jupyter nbconvert --to notebook --execute --inplace \
        --ExecutePreprocessor.timeout=7200 notebooks/02_perhead_spectral_trust.ipynb
"""

import json
import os

CELLS = []


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").split("\n")})


def code(text):
    CELLS.append({"cell_type": "code", "metadata": {}, "execution_count": None,
                  "outputs": [], "source": text.strip("\n").split("\n")})


# ===========================================================================
md(r"""
# Effet d'une injection sur le graphe d'attention — **tête par tête**, via `spectral-trust`

Pilote reproductible sur CPU. Deux jeux de données, un petit modèle instruct, et une règle cardinale :

> **Aucune moyenne de λ₂ entre têtes dans un résultat principal.** Chaque (couche, tête) est une observation conservée individuellement. Toute agrégation entre têtes est étiquetée *SECONDAIRE* et accompagnée de la distribution complète.

## Répartition déclarée des calculs

| `spectral-trust` 0.3.0 | local |
|---|---|
| symétrisation `W = (A+Aᵀ)/2` (`GraphConstructor.symmetrize_attention`) | masques de régions S / U / T sur le graphe **complet** |
| laplacien `L_sym = I − D^{-1/2} W D^{-1/2}` (`construct_laplacian`) | coupes, volumes, conductance, Ncut, flux orientés |
| spectre et métriques par tête (`per_head_metrics`) : `fiedler_value`, `connectivity_ratio`, `spectral_entropy_norm`, `hfer`, `spectral_radius` | appariement, tests par cellule, figures |

`token_span` **n'est jamais utilisé** : il extrait un sous-graphe induit et supprime précisément les arêtes entre segments que l'on veut mesurer.

## Ce qui n'est **pas** défendable — et que ce notebook ne fait pas

1. **Moyenner λ₂ entre têtes** et présenter le résultat comme principal : une moyenne sur 14 têtes noie un effet porté par une seule.
2. **Choisir la « meilleure » cellule après avoir vu la carte** couche × tête. La cellule tracée est déclarée *avant* (§ Configuration) et n'est pas celle qui « marche le mieux ».
3. **Comparer adverse et contrôle sans appariement en longueur** : λ₂ et les masses spectrales dépendent de T.
4. **Lire la classification comportementale automatique comme une mesure** : elle est provisoire ; l'annotation manuelle est en attente.

## Ce que ce pilote ne peut pas établir

Un effet apparié n'est pas un détecteur. Le contraste n'isole pas la malveillance : pour xTRam1, adverse et contrôle sont des **textes différents en tout** sauf la longueur ; pour SPML, le système est identique mais le User Prompt change de contenu et de forme. Une différence significative par cellule dit « ces deux populations diffèrent sur cette tête », pas « cette tête détecte l'injection ».
""")

# ---------------------------------------------------------------------------
md(r"""
## 0. Configuration — tout ce qui est décidé **avant** de voir un résultat
""")
code(r"""
import os, sys, json, time, hashlib, platform
import numpy as np, pandas as pd
import torch
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

import spectral_trust
from spectral_injection.perhead import config as C
from spectral_injection.perhead import data, capture, metrics, stats, figures, behaviour, report

# --- pilote : surcharger ici si besoin -----------------------------------
MODEL_ID   = os.environ.get("PERHEAD_MODEL", C.MODEL_ID)
N_PAIRS    = dict(C.N_PAIRS)          # {"xtram1": 20, "spml": 20}
MAX_TOKENS = dict(C.MAX_TOKENS)       # {"xtram1": 256, "spml": 512}
OUT        = os.path.abspath(os.path.join("..", "results", "perhead")) if os.path.basename(os.getcwd()) == "notebooks" else os.path.abspath("results/perhead")
os.makedirs(OUT, exist_ok=True)
for sub in ("attention", "figures"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)

META = {
    "run_started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "spectral_trust_version": spectral_trust.__version__,
    "torch": torch.__version__, "python": platform.python_version(),
    "seed": C.SEED, "n_pairs": N_PAIRS, "max_tokens": MAX_TOKENS,
    "length_tolerance": {"frac": C.LENGTH_TOL_FRAC, "min_tokens": C.LENGTH_TOL_MIN},
    "stats": {"primary_metric": C.PRIMARY_METRIC, "fdr_q": C.FDR_Q, "n_perm": C.N_PERM,
              "head_selection_rule": "TOUTES les tetes ; BH q=%.2f (principal) + max-stat FWER (secondaire)" % C.FDR_Q},
    "xtram1": {"primary_system": C.PRIMARY_SYSTEM, "system_configs": list(C.SYSTEM_CONFIGS)},
    "spml": {},
    "limits": [],
}
print("spectral-trust", spectral_trust.__version__, "| torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("modele :", MODEL_ID)
print("paires :", N_PAIRS, "| max_tokens :", MAX_TOKENS)
print("metrique primaire :", C.PRIMARY_METRIC, "| regle de selection :", META["stats"]["head_selection_rule"])
print("sorties ->", OUT)
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Le template du modèle, tel qu'il est **réellement** rendu

Le protocole impose d'utiliser le template officiel et de documenter ce qu'il fait. Trois configurations du message système sont rendues sur un texte sonde. **Si le template injecte un système par défaut non vide quand on n'en passe aucun, c'est signalé ici.**
""")
code(r"""
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
PROBE = "Translate to French: hello"
rendered = {}
for name, sysv in C.SYSTEM_CONFIGS.items():
    s = capture.render(tok, sysv, PROBE)
    n = len(tok(s, add_special_tokens=False)["input_ids"])
    rendered[name] = {"system_arg": sysv, "rendered": s, "n_tokens": n}
    print(f"=== config '{name}'  (system_arg={sysv!r}, {n} tokens)")
    print(repr(s)); print()

injected, content = capture.default_system_injected(tok)
META["template"] = {"chat_template_sha256": hashlib.sha256((tok.chat_template or "").encode()).hexdigest(),
                    "default_system_injected": injected, "default_system_content": content,
                    "rendered_probe": rendered}
if injected:
    print("!!! SIGNALEMENT : sans message systeme, le template INJECTE un systeme par defaut non vide :")
    print("   ", repr(content))
    print("    -> la configuration 'defaut_qwen' contient ce bloc ; 'vide' donne un bloc systeme VIDE ; 'minimal' un bloc explicite.")
    print("    -> configuration PRIMAIRE pour xTRam1 :", C.PRIMARY_SYSTEM)
else:
    print("Le template n'injecte aucun systeme par defaut.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. Chargement et inspection des deux jeux de données
""")
code(r"""
ds_x = data.load_xtram1()
meta_x = data.inspect_xtram1(ds_x)
print(json.dumps({k: v for k, v in meta_x.items() if k != "features"}, indent=1, ensure_ascii=False))
print("\nfeatures :", meta_x["features"])
print("\n--- exemples label 0 (legitime) ---")
for i in [0, 1, 2]:
    r = ds_x["train"][i] if ds_x["train"][i]["label"] == 0 else ds_x["train"][i + 10]
    print(f"  [{r['label']}] {r['text'][:140]!r}")
print("--- exemples label 1 (injection) ---")
k = 0
for r in ds_x["train"]:
    if r["label"] == 1:
        print(f"  [{r['label']}] {r['text'][:140]!r}"); k += 1
    if k >= 3: break
META["xtram1"].update({k: v for k, v in meta_x.items() if k not in ("features",)})
""")
code(r"""
df_s = data.load_spml()
meta_s = data.inspect_spml(df_s)
print(json.dumps(meta_s, indent=1, ensure_ascii=False))
print("\nSystem Prompt constant ?", meta_s["system_prompt_constant"], "->", meta_s["n_system_prompts_distinct"], "systemes distincts,",
      meta_s["n_systems_with_both_labels"], "ont les deux labels")
print("\n--- exemple injection=1 ---")
r = df_s[df_s["Prompt injection"] == 1].iloc[0]
print("  SYS :", repr(r["System Prompt"][:160])); print("  USR :", repr(r["User Prompt"][:160])); print("  Degree :", r["Degree"], "| Source :", r["Source"])
print("--- exemple injection=0 ---")
r = df_s[df_s["Prompt injection"] == 0].iloc[0]
print("  SYS :", repr(r["System Prompt"][:160])); print("  USR :", repr(r["User Prompt"][:160]))
META["spml"].update(meta_s)
with open(os.path.join(OUT, "datasets_metadata.json"), "w", encoding="utf-8") as fh:
    json.dump({"xtram1": meta_x, "spml": meta_s}, fh, indent=1, ensure_ascii=False)
print("\n-> datasets_metadata.json")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Construction des conditions et appariement en longueur

Longueur mesurée **après tokenisation du prompt complet formaté, système inclus**.

- **xTRam1** : adverse = `label 1`, contrôle = `label 0` de longueur la plus proche (vivier non réutilisé). Les deux textes sont différents en tout sauf la longueur — contrôle **entre** exemples. Le système est celui de la configuration primaire, identique dans les deux conditions.
- **SPML** : adverse = `Prompt injection 1`, contrôle = `Prompt injection 0` **du même System Prompt**, de longueur la plus proche. Le système est identique dans la paire par construction.

Les exemples écartés (trop longs, non appariables) sont journalisés, jamais imputés.
""")
code(r"""
sys_primary = C.SYSTEM_CONFIGS[C.PRIMARY_SYSTEM]
n_tok_x = lambda text: len(tok(capture.render(tok, sys_primary, text), add_special_tokens=False)["input_ids"])
n_tok_s = lambda s, u: len(tok(capture.render(tok, s, u), add_special_tokens=False)["input_ids"])

pairs_x, log_x = data.pair_xtram1(ds_x, n_tok_x, N_PAIRS["xtram1"], MAX_TOKENS["xtram1"], C.SEED)
pairs_s, log_s = data.pair_spml(df_s, n_tok_s, N_PAIRS["spml"], MAX_TOKENS["spml"], C.SEED)

def _pairing_meta(P, log):
    g = P.gap_tokens.abs()
    return dict(log, gap_abs_median=float(g.median()), gap_abs_max=int(g.max()),
                gap_signed_mean=float(P.gap_tokens.mean()))
META["xtram1"]["pairing"] = _pairing_meta(pairs_x, log_x)
META["spml"]["pairing"] = _pairing_meta(pairs_s, log_s)

print("xTRam1 :", json.dumps(META["xtram1"]["pairing"]))
print(pairs_x[["pair_id", "adv_tokens", "ctl_tokens", "gap_tokens"]].describe().loc[["mean", "min", "max"]].round(1))
print("\nSPML :", json.dumps(META["spml"]["pairing"]))
print(pairs_s[["pair_id", "adv_tokens", "ctl_tokens", "gap_tokens", "degree"]].describe().loc[["mean", "min", "max"]].round(1))
print("Degree des paires SPML :", pairs_s.degree.value_counts().sort_index().to_dict())
pairs_x.to_csv(os.path.join(OUT, "pairs_xtram1.csv"), index=False)
pairs_s.to_csv(os.path.join(OUT, "pairs_spml.csv"), index=False)
print("\n-> pairs_xtram1.csv, pairs_spml.csv (exemples retenus, journalises)")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Capture de l'attention — une seule passe par exemple — et métriques pour **toutes** les têtes

Pour chaque prompt : segmentation S/U/T par offsets, passe avant (`eager`, float32), **stockage de l'attention brute** (npz compressé, une clé par couche), puis :

- `per_head_metrics` de `spectral-trust` sur chaque couche → 5 métriques × **toutes** les têtes, rangées dans un tableau long `exemple_id, condition, couche, tete, metrique, valeur` ;
- quantités de région locales (coupe, conductance, Ncut de U contre son complément ; flux orientés) sur le graphe complet, **par tête**, ajoutées au même tableau ;
- vecteur de Fiedler sur la **cellule déclarée** uniquement, avec contrôle croisé de λ₂ contre `fiedler_value`.

Un échec sur une tête est journalisé et laissé NaN — jamais remplacé par une moyenne.
""")
code(r"""
cap = capture.Capturer(MODEL_ID, max_tokens=max(MAX_TOKENS.values()))
CELL = C.declared_cell(cap.n_layers)
META["model"] = {"id": MODEL_ID, "n_layers": cap.n_layers, "n_heads": cap.n_heads,
                 "n_kv_heads": cap.n_kv_heads, "device": cap.device, "dtype": "float32",
                 "attn_implementation": "eager"}
META["declared_cell"] = list(CELL)
cfg = metrics.library_config()
print(f"{MODEL_ID} : {cap.n_layers} couches x {cap.n_heads} tetes (kv={cap.n_kv_heads}), {cap.device}")
print(f"cellule DECLAREE (vecteur de Fiedler, adjacences) : couche {CELL[0]}, tete {CELL[1]}")
print("GSPConfig :", {k: getattr(cfg, k) for k in ("symmetrization", "normalization", "remove_self_loops", "eigen_solver")})
""")
code(r"""
LONG, ERRORS, FIEDLER, INTEGRITY, PROMPTS = [], [], {}, [], {}

def run_prompt(key, system, user, example_id, condition, dataset, sysname):
    p = capture.segment(tok, system, user)
    att = cap.forward(p)
    nbytes = capture.save_attention(os.path.join(OUT, "attention", f"{key}.npz"), att)
    long_df, errs = metrics.per_head_long(att, cfg, example_id, condition)
    long_df["dataset"] = dataset; long_df["system_config"] = sysname
    # quantites de region locales, par tete, graphe complet
    rq_rows = []
    for l in range(cap.n_layers):
        for h in range(cap.n_heads):
            rq = metrics.region_quantities(att[l][h], p.regions, cfg, "U")
            for m, v in rq.items():
                rq_rows.append((example_id, condition, l, h, "local_" + m, v, dataset, sysname))
    rq_df = pd.DataFrame(rq_rows, columns=list(long_df.columns))
    LONG.append(pd.concat([long_df, rq_df], ignore_index=True)); ERRORS.extend(errs)
    # cellule declaree : vecteur de Fiedler + controle croise
    f = metrics.fiedler(att[CELL[0]][CELL[1]], cfg)
    lib = float(long_df[(long_df.couche == CELL[0]) & (long_df.tete == CELL[1]) & (long_df.metrique == "fiedler_value")].valeur.iloc[0])
    FIEDLER[key] = {"u2": f["fiedler_vector"], "lambda_2": f["lambda_2"], "W": f["W"], "regions": p.regions, "tokens": p.tokens}
    s_to_u = rq_df[(rq_df.couche == CELL[0]) & (rq_df.tete == CELL[1]) & (rq_df.metrique == "local_Fbar_S_to_U")].valeur
    INTEGRITY.append({"key": key, "T": len(p), "n_S": p.meta["n_S"], "n_U": p.meta["n_U"], "n_T": p.meta["n_T"],
                      "lambda2_crosscheck_absdiff": abs(f["lambda_2"] - lib),
                      "Fbar_S_to_U_declared_cell": float(s_to_u.iloc[0]) if len(s_to_u) else float("nan"),
                      "attention_bytes": nbytes, "system_effective": p.meta["system_effective"]})
    PROMPTS[key] = p
    return p

t0 = time.time(); n_done = 0
# --- xTRam1 : trois configurations systeme, conditions ADV / CTL -----------
for sysname, sysv in C.SYSTEM_CONFIGS.items():
    for _, r in pairs_x.iterrows():
        for cond, text in (("ADV", r.adv_text), ("CTL", r.ctl_text)):
            run_prompt(f"xtram1__{sysname}__p{r.pair_id:03d}__{cond}", sysv, text, int(r.pair_id), cond, "xtram1", sysname)
            n_done += 1
    print(f"  xTRam1 [{sysname}] : {2*len(pairs_x)} prompts  ({time.time()-t0:.0f}s cumules)")
# --- SPML : systeme fourni, identique dans la paire ------------------------
for _, r in pairs_s.iterrows():
    for cond, user in (("ADV", r.adv_user), ("CTL", r.ctl_user)):
        run_prompt(f"spml__given__p{r.pair_id:03d}__{cond}", r.system_prompt, user, int(r.pair_id), cond, "spml", "given")
        n_done += 1
print(f"  SPML : {2*len(pairs_s)} prompts  ({time.time()-t0:.0f}s cumules)")

LONG = pd.concat(LONG, ignore_index=True)
LONG.to_parquet(os.path.join(OUT, "metrics_long.parquet"), index=False)
pd.DataFrame(ERRORS).to_csv(os.path.join(OUT, "errors.csv"), index=False)
INT = pd.DataFrame(INTEGRITY); INT.to_csv(os.path.join(OUT, "integrity.csv"), index=False)
print(f"\n{n_done} prompts, {len(LONG):,} lignes dans le tableau long, {len(ERRORS)} erreurs de tete")
print(f"attention stockee : {INT.attention_bytes.sum()/2**30:.2f} GiB dans {OUT}/attention/")
print("\nCONTROLES D'INTEGRITE")
print(f"  masque causal, lignes stochastiques, finitude : verifies a chaque passe (Capturer)")
print(f"  |lambda_2(eigh, laplacien spectral-trust) - fiedler_value(per_head_metrics)| max : {INT.lambda2_crosscheck_absdiff.max():.2e}  (arrondi float32 attendu ~1e-7)")
has_S = INT[INT.n_S > 0]
print(f"  Fbar_S->U sur la cellule declaree, prompts avec systeme : max {has_S.Fbar_S_to_U_declared_cell.abs().max():.2e}  (doit etre 0 : S precede U, masque causal)")
print(f"  tailles de T (template) par configuration :", LONG.groupby('system_config').size().to_dict() and {k: sorted(set(INT[INT.key.str.contains(k)].n_T)) for k in C.SYSTEM_CONFIGS} )
print(f"  metriques par tete NaN : {int(LONG[LONG.metrique.isin(['fiedler_value'])].valeur.isna().sum())}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Résultats principaux — **par cellule (couche, tête)**, toutes les têtes

Pour chaque jeu de données, sur la métrique primaire `fiedler_value` (λ₂) :

1. **distribution de λ₂ par condition sur l'ensemble des têtes** — chaque point est une (paire, couche, tête), rien n'est moyenné ;
2. **Δλ₂ = λ₂(adverse) − λ₂(contrôle) pour CHAQUE cellule**, test apparié par inversion de signe, puis **carte couche × tête des Δ médians** ;
3. **nombre de cellules significatives et leur sens**, Benjamini-Hochberg q = 0,05 (principal) et max-statistic FWER (secondaire, plus sévère).

La table complète des cellules est sauvegardée. Elle est affichée **triée par p**, ce qui n'est pas une sélection : toutes les cellules y sont et la correction porte sur toutes.
""")
code(r"""
RESULTS = {}

def analyse(dataset, sysname, label):
    sub = LONG[(LONG.dataset == dataset) & (LONG.system_config == sysname)]
    A, names, pairs = stats.cell_array(sub, C.PRIMARY_METRIC, cap.n_layers, cap.n_heads)
    cells = stats.test_cells(A, names, n_perm=C.N_PERM, seed=C.SEED, fdr_q=C.FDR_Q)
    counts = stats.count_significant(cells)
    cells.to_csv(os.path.join(OUT, f"cells_{label}.csv"), index=False)
    figures.lambda2_distribution(sub, C.PRIMARY_METRIC, f"{label} — λ₂ par condition, toutes les têtes",
                                 os.path.join(OUT, "figures", f"lambda2_dist_{label}.png"))
    figures.delta_map(cells, cap.n_layers, cap.n_heads, f"{label} — Δλ₂ médian par cellule (n={cells.attrs['n_pairs']} paires)",
                      os.path.join(OUT, "figures", f"delta_map_{label}.png"))
    RESULTS[label] = {"n_pairs": cells.attrs["n_pairs"], "dropped": cells.attrs["dropped"], "counts": counts}
    print(f"=== {label} : {cells.attrs['n_pairs']} paires, {counts['n_cells']} cellules, {len(cells.attrs['dropped'])} paires ecartees")
    print(f"    BH q<{C.FDR_Q}  : {counts['sig_bh']} cellules significatives  (+{counts['sig_bh_plus']} / -{counts['sig_bh_minus']})")
    print(f"    FWER      : {counts['sig_fwer']} cellules significatives  (+{counts['sig_fwer_plus']} / -{counts['sig_fwer_minus']})")
    print(f"    direction des Δ medians : {int((cells.direction=='+').sum())} cellules +, {int((cells.direction=='-').sum())} cellules -")
    print("    toutes les cellules, triees par p brute (pas une selection) — 10 premieres :")
    print(cells.sort_values("p_raw")[["cellule", "delta_median", "win_rate", "p_raw", "p_bh", "p_fwer"]].head(10).to_string(index=False))
    return cells

cells_x = analyse("xtram1", C.PRIMARY_SYSTEM, f"xtram1_{C.PRIMARY_SYSTEM}")
print()
cells_s = analyse("spml", "given", "spml")
""")
code(r"""
from IPython.display import Image, display
for lab in (f"xtram1_{C.PRIMARY_SYSTEM}", "spml"):
    display(Image(os.path.join(OUT, "figures", f"lambda2_dist_{lab}.png")))
    display(Image(os.path.join(OUT, "figures", f"delta_map_{lab}.png")))
""")

# ---------------------------------------------------------------------------
md(r"""
### 5b. Résumé **SECONDAIRE** par couche — avec la distribution complète

Ceci n'est **pas** un résultat principal. La colonne `mediane_entre_tetes_SECONDAIRE` ne doit jamais être lue seule : les colonnes `min`/`q25`/`q75`/`max` et les comptes de têtes par signe montrent ce qu'elle cache.
""")
code(r"""
for lab, cells in ((f"xtram1_{C.PRIMARY_SYSTEM}", cells_x), ("spml", cells_s)):
    print(f"=== {lab}  (SECONDAIRE)")
    print(stats.secondary_head_summary(cells).round(4).to_string()); print()
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Vecteur de Fiedler le long des tokens — **cellule déclarée**, jamais une moyenne

Couche et tête fixées en § 0 avant tout résultat. Première paire de chaque jeu, adverse et contrôle superposés, régions ombrées. Les matrices d'adjacence `W` sont tracées avec une **échelle de couleur commune** aux deux conditions.
""")
code(r"""
for dataset, sysname, label in (("xtram1", C.PRIMARY_SYSTEM, f"xtram1_{C.PRIMARY_SYSTEM}"), ("spml", "given", "spml")):
    ka, kc = f"{dataset}__{sysname}__p000__ADV", f"{dataset}__{sysname}__p000__CTL"
    fa, fc = FIEDLER[ka], FIEDLER[kc]
    figures.fiedler_plot(fa["u2"], fc["u2"], fa["regions"], fc["regions"], fa["lambda_2"], fc["lambda_2"], CELL,
                         f"{label} — paire 0", os.path.join(OUT, "figures", f"fiedler_{label}_p000.png"))
    figures.adjacency_pair(fa["W"], fc["W"], fa["regions"], fc["regions"],
                           f"{label} — paire 0, W couche {CELL[0]} tête {CELL[1]} (échelle commune)",
                           os.path.join(OUT, "figures", f"adjacency_{label}_p000.png"))
    display(Image(os.path.join(OUT, "figures", f"fiedler_{label}_p000.png")))
    display(Image(os.path.join(OUT, "figures", f"adjacency_{label}_p000.png")))
    print(f"{label} paire 0 : lambda_2 adverse={fa['lambda_2']:.5f}  controle={fc['lambda_2']:.5f}  (T={fa['u2'].size} / {fc['u2'].size})")
    # Descriptif sur la cellule DECLAREE et la paire 0 uniquement : ou se
    # trouve la masse de u_2 ? (aucune autre cellule n'est examinee)
    for cond, f in (("adverse", fa), ("controle", fc)):
        m = f["u2"] ** 2; reg = f["regions"]; top = int(np.argmax(m))
        print(f"   {cond:<9} masse de u2^2 : S {m[reg=='S'].sum():.3f}  U {m[reg=='U'].sum():.3f}  "
              f"T {m[reg=='T'].sum():.3f}  | token le plus charge : pos {top} ({reg[top]}) "
              f"{f['tokens'][top]!r} = {m[top]:.3f}")
""")
md(r"""
### Observation sur la cellule déclarée — descriptive, une paire, une cellule

Sur les paires affichées, u₂ n'est **pas** une bipartition système / utilisateur : sa masse se concentre sur **un seul token**, et ce token change entre conditions (voir les lignes « masse de u₂² » ci-dessus). Δλ₂ à cette cellule compare donc l'isolement de deux tokens différents, pas une même quantité structurelle dans deux états.

C'est précisément ce qu'une moyenne entre têtes — ou un λ₂ lu sans son vecteur — aurait caché. Aucune analyse supplémentaire n'a été ajoutée après ce constat : une quantification sur toutes les cellules (indice de localisation de u₂, région du token le plus chargé) serait une **suite à déclarer**, pas un résultat de ce pilote.
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. SPML — analyse par `Degree` (intensité d'injection) — **descriptive**

`Degree = 0 ⟺ injection = 0` exactement ; `Degree ∈ {1,…,10}` gradue l'injection. Avec 20 paires réparties sur 5 niveaux, aucun test par niveau n'est défendable : on rapporte, par niveau, la fraction de cellules dont le Δ médian est positif, et la distribution. Sous-puissance assumée.
""")
code(r"""
sub = LONG[(LONG.dataset == "spml") & (LONG.metrique == C.PRIMARY_METRIC)]
piv = sub.pivot_table(index=["exemple_id", "couche", "tete"], columns="condition", values="valeur")
piv["delta"] = piv["ADV"] - piv["CTL"]
piv = piv.reset_index().merge(pairs_s[["pair_id", "degree"]], left_on="exemple_id", right_on="pair_id")
rows = []
for dgr, g in piv.groupby("degree"):
    med = g.groupby(["couche", "tete"]).delta.median()
    rows.append({"degree": int(dgr), "n_paires": int(g.exemple_id.nunique()),
                 "cellules_delta_median_>0": int((med > 0).sum()), "cellules_delta_median_<0": int((med < 0).sum()),
                 "delta_q25": float(med.quantile(.25)), "delta_median": float(med.median()), "delta_q75": float(med.quantile(.75))})
deg_tab = pd.DataFrame(rows); print(deg_tab.round(5).to_string(index=False))
deg_tab.to_csv(os.path.join(OUT, "spml_by_degree_descriptive.csv"), index=False)
META["spml"]["degree_analysis"] = "descriptive seulement : 20 paires sur 5 niveaux, aucun test par niveau"
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Sensibilité au message système — xTRam1

Trois configurations, **même système dans les deux conditions de chaque configuration** : `vide` (bloc système vide), `defaut_qwen` (le système que le template injecte quand on n'en passe aucun), `minimal` (« You are a helpful assistant. »). On vérifie que les Δλ₂ **par cellule** ne dépendent pas de ce choix : corrélation de Spearman des 336 Δ médians entre configurations, et comptes de cellules significatives.
""")
code(r"""
from scipy.stats import spearmanr
sens = {}
for sysname in C.SYSTEM_CONFIGS:
    lab = f"xtram1_{sysname}"
    if sysname == C.PRIMARY_SYSTEM:
        sens[sysname] = cells_x
    else:
        sens[sysname] = analyse("xtram1", sysname, lab); print()
names = list(C.SYSTEM_CONFIGS)
print("Spearman des Δ medians par cellule entre configurations :")
lines = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a = sens[names[i]].set_index("cellule").delta_median; b = sens[names[j]].set_index("cellule").delta_median.reindex(a.index)
        rho, p = spearmanr(a, b)
        agree = float((np.sign(a) == np.sign(b)).mean())
        s = f"  {names[i]:<12} vs {names[j]:<12}  rho={rho:.3f}  p={p:.1e}  meme signe sur {agree:.1%} des cellules"
        print(s); lines.append(s.strip())
print("\nCellules significatives (BH) par configuration :", {k: stats.count_significant(v)["sig_bh"] for k, v in sens.items()})
META["sensitivity"] = {"spearman": lines, "sig_bh_by_config": {k: stats.count_significant(v)["sig_bh"] for k, v in sens.items()}}
for sysname in C.SYSTEM_CONFIGS:
    display(Image(os.path.join(OUT, "figures", f"delta_map_xtram1_{sysname}.png")))
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Mesure comportementale — génération à paramètres identiques, règle explicite et **provisoire**

Génération gloutonne (`do_sample=False`, même `max_new_tokens`) pour chaque prompt des conditions primaires. Classement par une règle de mots-clés (`suit` / `ignore` / `ambigu`) **annoncée comme provisoire** ; la colonne `annotation_manuelle` est vide et attend une lecture humaine. Les cas ambigus sont conservés. Les réponses aux contrôles donnent le taux de « suit » de la règle hors injection.
""")
code(r"""
beh = []
t0 = time.time()
for key, p in PROMPTS.items():
    ds_, sysname, pid, cond = key.split("__")
    if not ((ds_ == "xtram1" and sysname == C.PRIMARY_SYSTEM) or ds_ == "spml"):
        continue
    resp = cap.generate(p, C.GEN_MAX_NEW_TOKENS, C.GEN_DO_SAMPLE)
    beh.append({"key": key, "dataset": ds_, "pair_id": int(pid[1:]), "condition": cond, "reponse": resp, **behaviour.provisional_rule(resp)})
BEH = pd.DataFrame(beh)
BEH.to_csv(os.path.join(OUT, "behaviour_provisional.csv"), index=False)
print(f"{len(BEH)} generations en {time.time()-t0:.0f}s  (max_new_tokens={C.GEN_MAX_NEW_TOKENS}, do_sample={C.GEN_DO_SAMPLE})")
print("\nclasse PROVISOIRE par dataset x condition :")
print(pd.crosstab([BEH.dataset, BEH.condition], BEH.classe_provisoire))
print("\nexemples (adverse) :")
for _, r in BEH[BEH.condition == "ADV"].head(4).iterrows():
    print(f"  [{r.dataset} p{r.pair_id:03d}] {r.classe_provisoire:<7} {r.reponse[:110]!r}")
print("\n-> behaviour_provisional.csv : colonne 'annotation_manuelle' VIDE, a remplir ; la classe automatique n'est pas une mesure.")
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. Métadonnées, limites et rapport
""")
code(r"""
META["limits"] = [
    f"Pilote : {N_PAIRS['xtram1']} + {N_PAIRS['spml']} paires, un seul modele ({MODEL_ID}). Aucune generalisation.",
    "xTRam1 : controle ENTRE exemples (textes differents en tout sauf la longueur) ; les exemples adverses sont des phrases adversariales autonomes, pas des injections enfouies dans un contenu externe.",
    "SPML : systeme identique dans la paire, mais le User Prompt change de contenu, de forme et d'intention simultanement ; une difference par cellule n'isole pas la malveillance.",
    f"Appariement en longueur : ecart residuel max {META['xtram1']['pairing']['gap_abs_max']} (xTRam1) et {META['spml']['pairing']['gap_abs_max']} (SPML) tokens ; {META['spml']['pairing'].get('systems_unmatched', 0)} systemes SPML ecartes faute de controle assez proche.",
    "Aucune etiquette de succes d'attaque : la mesure comportementale est provisoire (regle de mots-cles) et non annotee.",
    "Aucune performance de detection n'est rapportee : un effet apparie n'est pas un detecteur.",
    "Analyse par Degree descriptive seulement (sous-puissance).",
    "per_head_metrics calcule en float32 : le controle croise lambda_2 porte un residu ~1e-7 attendu.",
]
META["run_finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
META["results"] = RESULTS
with open(os.path.join(OUT, "run_metadata.json"), "w", encoding="utf-8") as fh:
    json.dump(META, fh, indent=1, ensure_ascii=False, default=str)
txt = report.write_report(os.path.join(OUT, "RAPPORT.md"), META, RESULTS)
print(txt)
print("\n-> run_metadata.json, RAPPORT.md")
""")

# ===========================================================================
NB = {"cells": CELLS,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}
for cell in NB["cells"]:
    src = cell["source"]
    cell["source"] = [ln + "\n" for ln in src[:-1]] + [src[-1]]
dest = os.path.join("notebooks", "02_perhead_spectral_trust.ipynb")
os.makedirs("notebooks", exist_ok=True)
with open(dest, "w", encoding="utf-8") as fh:
    json.dump(NB, fh, ensure_ascii=False, indent=1)
n_code = sum(c["cell_type"] == "code" for c in NB["cells"])
print(f"-> {dest}  ({len(NB['cells'])} cellules, dont {n_code} de code)")
