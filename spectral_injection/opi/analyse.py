"""Analyse d'une campagne : contraste principal et comparaisons secondaires.

    python -m spectral_injection.opi.analyse results/opi/smoke

Lit le tenseur produit par ``run.py``, agrege couches et tetes au niveau de
l'exemple, puis applique le module statistique. Rien n'est selectionne a
partir des resultats : la famille confirmatoire est celle de ``stats.py``,
fixee avant execution.
"""

import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd

from .stats import (CONFIRMATORY, CONFIRMATORY_TRAJ, PRIMARY_AGGREGATOR,
                    SECONDARY_CONTRASTS, aggregate_examples,
                    aggregate_trajectories, analyse_contrast)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="analyse statistique OPI")
    p.add_argument("run_dir")
    p.add_argument("--n-perm", type=int, default=10000)
    p.add_argument("--n-boot", type=int, default=5000)
    p.add_argument("--aggregator", default=PRIMARY_AGGREGATOR,
                   choices=["median", "mean"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--secondary", action="store_true",
                   help="ajoute les contrastes secondaires")
    args = p.parse_args(argv)

    pd.set_option("display.width", 200)
    d = np.load(os.path.join(args.run_dir, "values.npz"), allow_pickle=True)
    V = d["values"]
    names = [str(x) for x in d["metrics"]]
    conds = [str(x) for x in d["conditions"]]
    meta = pd.read_csv(os.path.join(args.run_dir, "rows.csv"))

    print(f"campagne : {args.run_dir}")
    print(f"  tenseur H1-H5 {V.shape}, conditions {conds}")
    has_traj = "trajectories" in d
    if has_traj:
        T = d["trajectories"]
        tnames = [str(x) for x in d["traj_metrics"]]
        print(f"  tenseur H6    {T.shape}")
    print(f"  agregation couches x tetes -> exemple : {args.aggregator}")
    print(f"  unite statistique : l'exemple ({V.shape[0]} au total)")
    print()

    # strates : tache x famille d'attaque
    strata = (meta.task_type + "|" + meta.attack_type).to_numpy()
    print(f"  strates (tache x attaque) : {len(set(strata))} pour "
          f"{len(strata)} exemples")
    nm = meta.filler_matched.sum() if "filler_matched" in meta else len(meta)
    print(f"  appariement C3 complet : {nm}/{len(meta)}")
    print()

    # ------------------------------------------------------------------
    # FAMILLE UNIQUE : H1-H5 et H6 sont concatenees en une seule matrice,
    # testees par une seule procedure de permutation, avec une seule
    # correction max-statistic. Deux appels separes auraient donne deux
    # familles independantes et sous-corrige l'ensemble.
    # ------------------------------------------------------------------
    A = aggregate_examples(V, args.aggregator)
    all_names = list(names)
    blocks = [A]
    if has_traj:
        At = aggregate_trajectories(T, args.aggregator)
        blocks.append(At)
        all_names = all_names + list(tnames)
    A_all = np.concatenate(blocks, axis=2)
    fam = [f for f in list(CONFIRMATORY) + list(CONFIRMATORY_TRAJ)
           if f in all_names]
    missing = [f for f in list(CONFIRMATORY) + list(CONFIRMATORY_TRAJ)
               if f not in all_names]
    if missing:
        print(f"  ATTENTION : metriques confirmatoires absentes : {missing}")
    print(f"  famille confirmatoire : {len(fam)} metriques, "
          f"une seule correction max-statistic")
    rows, pdata = analyse_contrast(A_all, all_names, conds, fam, "C4", "C3",
                                   strata=strata, n_perm=args.n_perm,
                                   n_boot=args.n_boot, seed=args.seed)

    df = pd.DataFrame(rows)
    print("=" * 110)
    print(f"CONTRASTE PRINCIPAL  C4 - C3   "
          f"(famille confirmatoire de {len(df)} metriques, correction "
          f"max-statistic)")
    print(f"exemples retenus {pdata.n}, ecartes {pdata.dropped.size}")
    print()
    show = ["metric", "n", "median", "dz", "win_rate", "p_raw", "p_fwer",
            "mean_lo", "mean_hi", "stratified", "degenerate"]
    print(df[show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print()
    meth = df.method.iloc[0]
    print(f"methode de permutation : {meth} "
          f"({int(df.n_perm_effective.iloc[0])} configurations)")
    print(f"plancher exact bilateral 2^(1-n) : {df.p_floor_exact.iloc[0]:.5f}")
    if meth == "monte_carlo":
        print(f"resolution Monte-Carlo 1/(B+1)   : {df.p_floor_mc.iloc[0]:.5f}")
    print(f"plancher effectivement applicable : {df.p_floor.iloc[0]:.5f}")
    if df.p_floor.iloc[0] > 0.05:
        print("  AUCUNE p-valeur ne peut etre significative a 5 % avec cet "
              "effectif : le test est exsangue par construction.")
    deg = df[df.degenerate != "ok"]
    if len(deg):
        print(f"metriques degenerees (conservees dans la famille) : "
              f"{dict(zip(deg.metric, deg.degenerate))}")
    sig = int((df.p_fwer < 0.05).sum())
    print(f"metriques significatives apres correction : {sig} / {len(df)}")
    df.to_csv(os.path.join(args.run_dir, "stats_primary.csv"), index=False)

    if args.secondary:
        sec = []
        for a, b in SECONDARY_CONTRASTS:
            rr, _ = analyse_contrast(A_all, all_names, conds, fam, a, b,
                                     strata=strata, n_perm=args.n_perm,
                                     n_boot=args.n_boot, seed=args.seed)
            sec += rr
        sdf = pd.DataFrame(sec)
        print()
        print("=" * 110)
        print("CONTRASTES SECONDAIRES  (statut exploratoire, famille "
              "corrigee separement par contraste)")
        print(sdf[["contrast", "metric", "median", "dz", "p_fwer"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        sdf.to_csv(os.path.join(args.run_dir, "stats_secondary.csv"),
                   index=False)

    with open(os.path.join(args.run_dir, "analysis_config.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"aggregator": args.aggregator, "n_perm": args.n_perm,
                   "n_boot": args.n_boot, "seed": args.seed,
                   "confirmatory": list(CONFIRMATORY),
                   "confirmatory_traj": list(CONFIRMATORY_TRAJ),
                   "primary_contrast": ["C4", "C3"],
                   "secondary_contrasts": [list(x) for x in SECONDARY_CONTRASTS],
                   "n_examples": int(V.shape[0]),
                   "n_kept": int(pdata.n),
                   "n_dropped": int(pdata.dropped.size)}, fh, indent=1)
    print(f"\n-> {args.run_dir}/stats_primary.csv, analysis_config.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
