"""Delta C4 - C3 sur l'energie de Dirichlet par tete.

MEME MACHINERIE que ``analyse_region_energy`` -- bootstrap stratifie,
Wilcoxon apparie, correction BH -- importee de ce module plutot que
reecrite, pour qu'aucune divergence ne s'installe entre les deux analyses.

FAMILLE HYBRIDE. Le signal est le flux residuel : ces quantites ne sont PAS
attention-only. Elles sont corrigees par BH SEPAREMENT des 81 quantites de
la campagne regionale, jamais dans le meme lot -- melanger les deux
familles dans une seule correction melerait des mesures de natures
differentes.

QUANTITE PRINCIPALE : ``smoothness_sym``, le quotient de Rayleigh matriciel
``tr(X^T L X) / ||X||_F^2``. ``energy_sym`` est rendue aussi mais la norme
du flux residuel croit de plusieurs ordres de grandeur avec la profondeur,
donc un Delta sur l'energie brute melange un effet de graphe et un effet de
norme. ``*_comb`` est le controle d'integrite : la moyenne par tete y egale
l'energie du graphe moyen par linearite de ``L = D - W``.

AGREGATION PAR PAIRE. Moyenne ET mediane sur les L x H cellules, les deux
rendues : la carte par cellule reste la sortie primaire du runner, et
l'agregat est declare ici, pas suppose.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analyse_region_energy import (BOOT_SEED, FDR_Q, N_BOOT, bh_fdr, dz,
                                   paired_test, strat_bootstrap)

QUANTITIES = ["smoothness_sym", "energy_sym", "smoothness_comb",
              "energy_comb", "signal_norm"]
CONTROL_ARMS = ["C3_opi", "C3_wiki"]


def pair_deltas(d: str, agg: str, arm: str) -> pd.DataFrame:
    """Delta apparie CELLULE PAR CELLULE, puis agrege par paire.

    L'appariement est a DEUX niveaux et dans cet ordre, qui n'est pas
    indifferent :

    1. la cellule. ``C4`` et ``C3`` ont le meme nombre de couches et de
       tetes, donc la cellule (l, h) existe dans les deux bras et designe
       la meme tete du meme modele. La soustraction se fait la, avant toute
       agregation.
    2. la paire. Les deux bras partagent l'exemple, la tache, le template
       et -- par construction a Delta N = 0 -- le nombre de tokens de
       chaque region. Le Wilcoxon porte sur ces differences par paire.

    Agreger AVANT de soustraire donnerait, pour la mediane, la "difference
    des medianes" et non la "mediane des differences" : deux quantites
    distinctes, dont seule la seconde est appariee. Pour la moyenne les
    deux coincident par linearite, ce qui sert ici de controle.
    """
    D = pd.read_parquet(os.path.join(d, "dirichlet_long.parquet"))
    dup = D.duplicated(["pair_id", "arm", "couche", "tete"]).sum()
    if dup:
        raise SystemExit(f"{dup} cellules dupliquees : identifiant de paire "
                         f"non unique, resultats non interpretables")
    D = D[D.arm.isin(["C4", arm])]
    piv = D.pivot_table(index=["pair_id", "couche", "tete"], columns="arm",
                        values=QUANTITIES)
    f = np.mean if agg == "mean" else np.median
    out = {}
    for q in QUANTITIES:
        if (q, "C4") not in piv.columns or (q, arm) not in piv.columns:
            continue
        cell = piv[(q, "C4")] - piv[(q, arm)]       # apparie par cellule
        n_missing = int(cell.isna().sum())
        if n_missing:
            raise SystemExit(f"{q} : {n_missing} cellules sans contrepartie "
                             f"dans {arm} ; l'appariement est incomplet")
        out[q] = cell.groupby("pair_id").agg(f)     # puis agrege
    return pd.DataFrame(out)


def analyse(d: str, agg: str) -> pd.DataFrame:
    pairs = pd.read_csv(os.path.join(d, "pairs.csv")).set_index("pair_id")

    rows = []
    for arm in CONTROL_ARMS:
        Dt = pair_deltas(d, agg, arm)
        pids = [p for p in Dt.index if p in pairs.index]
        Dt = Dt.loc[pids]
        strata = (pairs.loc[pids, "task_type"].astype(str) + "|"
                  + pairs.loc[pids, "attack_type"].astype(str)).to_numpy()
        cols = list(Dt.columns)
        if not cols:
            continue
        Dm = Dt.to_numpy()                                 # [n, k]
        B = strat_bootstrap(Dm, strata, N_BOOT, BOOT_SEED)
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        p = paired_test(Dm)
        q_bh = bh_fdr(p)
        e = dz(Dm)
        for k, name in enumerate(cols):
            x = Dm[:, k][np.isfinite(Dm[:, k])]
            rows.append({
                "arm": arm, "quantity": name, "agg": agg, "n_pairs": x.size,
                "mean_delta": float(np.nanmean(Dm[:, k])),
                "median_delta": float(np.nanmedian(Dm[:, k])),
                "ci_lo": float(lo[k]), "ci_hi": float(hi[k]),
                "ci_excludes_0": bool(lo[k] > 0 or hi[k] < 0),
                "wilcoxon_p": float(p[k]), "bh_q": float(q_bh[k]),
                "bh_signif": bool(q_bh[k] < FDR_Q),
                "dz": float(e[k]),
                "frac_pairs_positive": float(np.mean(x > 0)) if x.size else np.nan,
            })
    return pd.DataFrame(rows)


def cell_agreement(d: str, arm: str, q: str) -> Dict:
    """Part des cellules (couche, tete) dont le Delta median a le meme signe."""
    D = pd.read_parquet(os.path.join(d, "dirichlet_long.parquet"))
    D = D[D.arm.isin(["C4", arm])]
    piv = D.pivot_table(index=["pair_id", "couche", "tete"], columns="arm",
                        values=q)
    delta = (piv["C4"] - piv[arm]).rename("_d").reset_index()
    m = delta.groupby(["couche", "tete"])["_d"].median()
    sign = np.sign(np.nanmedian(m.to_numpy()))
    return {"n_cells": int(m.size),
            "frac_same_sign": float(np.mean(np.sign(m.to_numpy()) == sign)),
            "dominant_sign": "+" if sign > 0 else "-"}


def block_deltas(d: str, agg: str, arm: str, what: str) -> pd.DataFrame:
    """Delta apparie par cellule sur les 21 blocs, puis agrege par paire.

    Meme ordre d'operations que ``pair_deltas`` : soustraction au niveau de
    la cellule, agregation ensuite.
    """
    B = pd.read_parquet(os.path.join(d, "dirichlet_blocks_long.parquet"))
    B = B[B.arm.isin(["C4", arm])]
    piv = B.pivot_table(index=["pair_id", "couche", "tete", "bloc"],
                        columns="arm", values=what)
    cell = (piv["C4"] - piv[arm]).rename("_d").reset_index()
    f = "mean" if agg == "mean" else "median"
    return cell.pivot_table(index="pair_id", columns="bloc", values="_d",
                            aggfunc=f)


def analyse_blocks(d: str, agg: str, what: str = "d_share") -> pd.DataFrame:
    """Les 21 blocs, meme machinerie appariee que la famille scalaire.

    BH est appliquee sur les 21 blocs d'un bras, pas sur un bloc choisi
    apres coup : ``PRIMARY_BLOCK`` est declare dans le module de calcul,
    avant la campagne, et les 20 autres sont rapportes pour qu'on voie si
    le bloc primaire se distingue ou se noie.
    """
    pairs = pd.read_csv(os.path.join(d, "pairs.csv")).set_index("pair_id")
    rows = []
    for arm in CONTROL_ARMS:
        Dt = block_deltas(d, agg, arm, what)
        pids = [p for p in Dt.index if p in pairs.index]
        Dt = Dt.loc[pids]
        strata = (pairs.loc[pids, "task_type"].astype(str) + "|"
                  + pairs.loc[pids, "attack_type"].astype(str)).to_numpy()
        Dm = Dt.to_numpy()
        B = strat_bootstrap(Dm, strata, N_BOOT, BOOT_SEED)
        lo, hi = np.percentile(B, [2.5, 97.5], axis=0)
        p = paired_test(Dm)
        q_bh = bh_fdr(p)
        e = dz(Dm)
        for k, name in enumerate(Dt.columns):
            x = Dm[:, k][np.isfinite(Dm[:, k])]
            rows.append({
                "arm": arm, "bloc": name, "measure": what, "agg": agg,
                "n_pairs": x.size,
                "mean_delta": float(np.nanmean(Dm[:, k])),
                "ci_lo": float(lo[k]), "ci_hi": float(hi[k]),
                "wilcoxon_p": float(p[k]), "bh_q": float(q_bh[k]),
                "bh_signif": bool(q_bh[k] < FDR_Q), "dz": float(e[k]),
                "frac_pairs_positive": float(np.mean(x > 0)) if x.size else np.nan,
            })
    return pd.DataFrame(rows)


def block_cell_agreement(d: str, arm: str, bloc: str,
                         what: str = "d_share") -> Dict:
    B = pd.read_parquet(os.path.join(d, "dirichlet_blocks_long.parquet"))
    B = B[(B.arm.isin(["C4", arm])) & (B.bloc == bloc)]
    piv = B.pivot_table(index=["pair_id", "couche", "tete"], columns="arm",
                        values=what)
    delta = (piv["C4"] - piv[arm]).rename("_d").reset_index()
    m = delta.groupby(["couche", "tete"])["_d"].median().to_numpy()
    sign = np.sign(np.nanmedian(m))
    return {"n_cells": int(m.size),
            "frac_same_sign": float(np.mean(np.sign(m) == sign)),
            "dominant_sign": "+" if sign > 0 else "-"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "dirichlet"))
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    tags = [t for t in sorted(os.listdir(args.root))
            if os.path.exists(os.path.join(args.root, t,
                                           "dirichlet_long.parquet"))]
    if not tags:
        raise SystemExit(f"aucun run avec dirichlet_long.parquet dans {args.root}")

    allrows = []
    for t in tags:
        d = os.path.join(args.root, t)
        prov = json.load(open(os.path.join(d, "provenance.json"),
                              encoding="utf-8"))
        print("=" * 78)
        print(f"{t}   {prov['n_layers']} couches x {prov['n_heads']} tetes   "
              f"{prov['n_pairs_kept']} paires")
        lin = prov.get("dirichlet_linearity") or {}
        for nm, v in lin.items():
            flag = ("  <- doit etre ~0" if nm == "none"
                    else "  <- la quantite d'interet")
            print(f"  ecart moyenne-par-tete vs graphe-moyen, {nm:5s} : "
                  f"max {v['rel_max']:.2e}  median {v['rel_median']:.2e}{flag}")
        print("=" * 78)

        for agg in ("mean", "median"):
            R = analyse(d, agg)
            R.insert(0, "model", t)
            allrows.append(R)
            print(f"\n  agregation par paire : {agg}")
            for arm, g in R.groupby("arm"):
                print(f"    bras {arm}")
                for _, r in g.iterrows():
                    mark = "*" if r.bh_signif else " "
                    print(f"      {mark} {r.quantity:17s} "
                          f"delta_moyen {r.mean_delta:+.6g}  "
                          f"IC95 [{r.ci_lo:+.4g}, {r.ci_hi:+.4g}]  "
                          f"p={r.wilcoxon_p:.4f}  q={r.bh_q:.4f}  "
                          f"dz={r.dz:+.3f}  "
                          f"paires>0 {100 * r.frac_pairs_positive:.0f}%")

        if os.path.exists(os.path.join(d, "dirichlet_blocks_long.parquet")):
            pb = prov.get("dirichlet_primary_block", "task--slot")
            for c in (prov.get("dirichlet_decomposition_check") or []):
                print(f"  somme des 21 blocs vs tr(X^T L X), "
                      f"{c['normalization']:5s} : {c['rel_max']:.2e}")
            for what in ("d_share", "d_density"):
                BL = analyse_blocks(d, "mean", what)
                BL.insert(0, "model", t)
                allrows.append(BL.rename(columns={"bloc": "quantity"}))
                print(f"\n  blocs, mesure {what}, agregation mean "
                      f"(bloc primaire declare : {pb})")
                for arm, g in BL.groupby("arm"):
                    g = g.reindex(g.mean_delta.abs()
                                  .sort_values(ascending=False).index)
                    print(f"    bras {arm} -- 5 plus fortes amplitudes sur 21")
                    for _, r in g.head(5).iterrows():
                        star = "*" if r.bh_signif else " "
                        tag = "  <== PRIMAIRE" if r.bloc == pb else ""
                        print(f"      {star} {r.bloc:20s} "
                              f"{r.mean_delta:+.6g}  IC95 "
                              f"[{r.ci_lo:+.3g}, {r.ci_hi:+.3g}]  "
                              f"q={r.bh_q:.4f}  dz={r.dz:+.3f}  "
                              f"paires>0 {100 * r.frac_pairs_positive:.0f}%{tag}")
                    rp = g[g.bloc == pb]
                    if not rp.empty and rp.iloc[0].bloc not in \
                            g.head(5).bloc.tolist():
                        r = rp.iloc[0]
                        star = "*" if r.bh_signif else " "
                        print(f"      {star} {r.bloc:20s} "
                              f"{r.mean_delta:+.6g}  q={r.bh_q:.4f}  "
                              f"dz={r.dz:+.3f}  <== PRIMAIRE, hors du top 5")
            for arm in CONTROL_ARMS:
                a = block_cell_agreement(d, arm, pb)
                print(f"    accord entre cellules sur {pb} ({arm}) : "
                      f"{a['frac_same_sign'] * 100:.0f} % des {a['n_cells']} "
                      f"cellules, signe {a['dominant_sign']}")

        print("\n  accord entre cellules sur smoothness_sym "
              "(part des L x H cellules de meme signe)")
        for arm in CONTROL_ARMS:
            try:
                a = cell_agreement(d, arm, "smoothness_sym")
                print(f"    {arm:8s} {a['frac_same_sign'] * 100:.0f} % des "
                      f"{a['n_cells']} cellules, signe {a['dominant_sign']}")
            except (KeyError, ValueError):
                pass

    out = pd.concat(allrows, ignore_index=True)
    p = os.path.join(args.root, "dirichlet_tests.csv")
    out.to_csv(p, index=False)
    print(f"\necrit : {p}")
    print(f"correction BH appliquee SEPAREMENT de la famille attention-only "
          f"(q < {FDR_Q})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
