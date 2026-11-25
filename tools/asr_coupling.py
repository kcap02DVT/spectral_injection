"""L'amplitude du Delta spectral suit-elle le SUCCES de l'injection ?

C'est le test de pertinence le plus exigeant du protocole. Si la signature
mesuree n'etait qu'une reaction a la presence de texte imperatif, elle
serait presente que l'attaque aboutisse ou non. Si elle est liee au
mecanisme de detournement, son amplitude doit suivre le succes.

DEUX NIVEAUX, de valeur tres inegale.

1. INTER-MODELES. Correlation entre le taux de detournement d'un modele et
   l'amplitude mediane de son Delta. n = nombre de modeles, soit au mieux
   6 : aucune puissance, et les modeles diffèrent par tout a la fois
   (taille, famille, nombre de tetes). Rendu pour completude, lu comme
   descriptif, JAMAIS comme une preuve.

2. INTRA-MODELE, APPARIE -- le seul qui porte quelque chose. A l'interieur
   d'un modele, les paires se separent en deux groupes selon que C4 a
   detourne la tache ou non. Les deux groupes partagent le modele, le
   template, le protocole et le plan de sondage ; ils ne diffèrent que par
   l'issue de l'attaque. On compare donc le Delta des paires DETOURNEES a
   celui des paires NON DETOURNEES, par Mann-Whitney (echantillons
   independants : une paire est dans un groupe ou dans l'autre, pas dans
   les deux).

CE QUI RESTE CONFONDU, et doit etre dit. Le succes du detournement n'est
pas assigne au hasard : il depend du type d'attaque et de la tache injectee.
Un ecart entre groupes peut donc refleter la composition en strates plutot
que le succes lui-meme. Le script rend la composition des deux groupes par
``attack_type`` et par ``injected_task``, et conduit en plus le test
STRATIFIE par type d'attaque quand chaque strate contient les deux issues.
Un ecart qui disparait a l'interieur des strates est un effet de
composition, pas un effet du succes.

Les paires dont l'ASR n'est pas defini (tache injectee a sortie libre) sont
exclues, jamais comptees comme echecs.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Quantites suivies, declarees. Ce sont celles de plus forte amplitude du
#: pilote, pas un choix fait apres lecture des resultats de la campagne.
TRACKED_FLOWS = [("slot", "task"), ("slot", "normal"), ("tail", "slot"),
                 ("template", "slot")]
TRACKED_BLOCKS = ["slot--slot", "task--slot", "slot--tail"]


def pair_deltas(d: str, arm: str) -> pd.DataFrame:
    """Une ligne par paire, une colonne par quantite suivie.

    Le Delta d'une paire est la moyenne sur ses L x H cellules, comme dans
    l'analyse principale.
    """
    out = {}
    fl = pd.read_parquet(os.path.join(d, "flows_long.parquet"))
    for q, k in TRACKED_FLOWS:
        s = fl[(fl["query"] == q) & (fl["key"] == k)]
        g = s.groupby(["pair_id", "arm"]).flow.mean().unstack("arm")
        if "C4" in g and arm in g:
            out[f"{q}|{k}"] = g["C4"] - g[arm]
    del fl
    bl = pd.read_parquet(os.path.join(d, "blocks_long.parquet"))
    for b in TRACKED_BLOCKS:
        s = bl[bl.bloc == b]
        g = s.groupby(["pair_id", "arm"]).share.mean().unstack("arm")
        if "C4" in g and arm in g:
            out[f"share:{b}"] = g["C4"] - g[arm]
    del bl
    return pd.DataFrame(out)


def composition(sub: pd.DataFrame, col: str) -> Dict:
    t = pd.crosstab(sub[col], sub.hijacked)
    return {str(i): {str(c): int(t.loc[i, c]) for c in t.columns}
            for i in t.index}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_energy"))
    ap.add_argument("--arm", default="C3_opi")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tags = [t for t in sorted(os.listdir(args.root))
            if os.path.exists(os.path.join(args.root, t, "provenance.json"))]

    rows_within, rows_between = [], []
    for t in tags:
        d = os.path.join(args.root, t)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        if not prov.get("asr"):
            continue
        pairs = pd.read_csv(os.path.join(d, "pairs.csv"))
        if "asr_defined_C4" not in pairs:
            continue
        D = pair_deltas(d, args.arm)
        m = pairs.set_index("pair_id")
        keep = m.index[m.asr_defined_C4.fillna(False).astype(bool)]
        keep = [p for p in keep if p in D.index]
        sub = D.loc[keep].copy()
        sub["hijacked"] = m.loc[keep, "asr_hijacked_C4"].astype(bool).to_numpy()
        sub["attack_type"] = m.loc[keep, "attack_type"].to_numpy()
        sub["injected_task"] = m.loc[keep, "injected_task"].to_numpy()

        n1 = int(sub.hijacked.sum())
        n0 = int((~sub.hijacked).sum())
        rate = n1 / (n1 + n0) if (n1 + n0) else np.nan
        print(f"\n=== {t}  detournees {n1}  non detournees {n0}  "
              f"taux {rate:.0%}")
        print(f"    composition par attack_type : "
              f"{json.dumps(composition(sub, 'attack_type'))}")

        quantities = [c for c in D.columns]
        if n1 < 4 or n0 < 4:
            print("    trop desequilibre pour un test intra-modele, "
                  "ignore a ce niveau")
        else:
            for q in quantities:
                a = sub.loc[sub.hijacked, q].dropna().to_numpy()
                b = sub.loc[~sub.hijacked, q].dropna().to_numpy()
                if a.size < 4 or b.size < 4:
                    continue
                u = mannwhitneyu(a, b, alternative="two-sided")
                # test stratifie : meme comparaison dans chaque strate
                strat = []
                for s, g in sub.groupby("attack_type"):
                    aa = g.loc[g.hijacked, q].dropna().to_numpy()
                    bb = g.loc[~g.hijacked, q].dropna().to_numpy()
                    if aa.size >= 3 and bb.size >= 3:
                        strat.append((s, float(np.median(aa) - np.median(bb)),
                                      float(mannwhitneyu(
                                          aa, bb,
                                          alternative="two-sided").pvalue)))
                same = (all(np.sign(x[1]) == np.sign(strat[0][1])
                            for x in strat) if strat else None)
                rows_within.append({
                    "model": t, "quantity": q,
                    "median_hijacked": float(np.median(a)),
                    "median_not": float(np.median(b)),
                    "diff_medians": float(np.median(a) - np.median(b)),
                    "mannwhitney_p": float(u.pvalue),
                    "n_hijacked": a.size, "n_not": b.size,
                    "n_strata_tested": len(strat),
                    "strata_same_sign": same,
                    "strata_detail": json.dumps(strat),
                })

        for q in quantities:
            rows_between.append({"model": t, "quantity": q,
                                 "hijack_rate": rate,
                                 "median_delta": float(
                                     np.nanmedian(sub[q].to_numpy()))})

    if not rows_between:
        raise SystemExit("aucun modele avec ASR")

    W = pd.DataFrame(rows_within)
    B = pd.DataFrame(rows_between)
    W.to_csv(os.path.join(args.root, "asr_within_model.csv"), index=False)

    print("\n" + "=" * 78)
    print("INTRA-MODELE : Delta des paires detournees vs non detournees")
    print("=" * 78)
    if W.empty:
        print("  aucun modele suffisamment equilibre")
    else:
        for q, g in W.groupby("quantity"):
            sig = g[g.mannwhitney_p < 0.05]
            cons = g[g.strata_same_sign == True]
            print(f"  {q:22s} significatif dans {len(sig)}/{len(g)} modeles"
                  f"   |   signe constant entre strates dans "
                  f"{len(cons)}/{len(g)}")
            for _, r in g.iterrows():
                print(f"      {r.model:14s} med_detourne {r.median_hijacked:+.5f}"
                      f"  med_non {r.median_not:+.5f}"
                      f"  ecart {r.diff_medians:+.5f}  p={r.mannwhitney_p:.3f}")

    print("\n" + "=" * 78)
    print("INTER-MODELES : correlation taux de detournement x amplitude")
    print("(descriptif, n = nombre de modeles ; aucune puissance)")
    print("=" * 78)
    bet = []
    for q, g in B.groupby("quantity"):
        g = g.dropna(subset=["hijack_rate", "median_delta"])
        if len(g) < 3:
            continue
        rho, p = spearmanr(g.hijack_rate, g.median_delta)
        bet.append({"quantity": q, "n_models": len(g), "spearman_rho": rho,
                    "p": p})
        print(f"  {q:22s} n={len(g)}  rho={rho:+.3f}  p={p:.3f}")
    pd.DataFrame(bet).to_csv(
        os.path.join(args.root, "asr_between_models.csv"), index=False)
    print(f"\necrits : asr_within_model.csv, asr_between_models.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
