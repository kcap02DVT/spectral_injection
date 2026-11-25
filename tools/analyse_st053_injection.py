"""Analyse de l'essai 0.5.3 : 17 quantites, 3 familles, 2 contrastes.

PRIMAIRE : PAR TETE. Pour chaque (quantite, tete), Wilcoxon apparie
bilateral sur les paires, dz de Cohen, Benjamini-Hochberg a q=0,05 DANS
CHAQUE FAMILLE separement -- jamais sur les trois familles ensemble, la
famille B dependant des etats caches et la famille C des matrices de
valeurs. Aucune moyenne entre tetes dans le resultat primaire.

SECONDAIRE, ETIQUETE COMME AGREGAT : la mediane sur les tetes par paire,
puis le meme test. Rendu pour la lisibilite, pas pour conclure.

SPECIFICITE. rapport_dz = |dz(C4 - C3_opi)| / max(|dz(C3_wiki - C3_opi)|,
0,05). Un rapport proche de 1 signifie que la quantite bouge des qu'on
change le texte du slot, injection ou pas.

    python tools/analyse_st053_injection.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import bh_fdr, dz, paired_test  # noqa: E402
from test_st053_injection import (ARM_ADV, ARM_BEN, ARM_REF, FAM,  # noqa: E402
                                  QUANTITES)

RACINE = os.path.join("results", "opi", "st053")
DZ_PLANCHER = 0.05
FDR_Q = 0.05


def deltas(D: pd.DataFrame, alt: str) -> pd.DataFrame:
    """Un Delta par (pair_id, couche, tete, quantite) : alt - ref."""
    k = ["pair_id", "couche", "tete"]
    a = D[D.arm == alt].set_index(k)[QUANTITES].sort_index()
    b = D[D.arm == ARM_REF].set_index(k)[QUANTITES].sort_index()
    inter = a.index.intersection(b.index)
    return (a.loc[inter] - b.loc[inter]).reset_index()


def par_tete(Del: pd.DataFrame) -> pd.DataFrame:
    """Test par (quantite, couche, tete). BH dans chaque famille."""
    paires = sorted(Del.pair_id.unique())
    lignes = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            piv = Del.pivot_table(index="pair_id", columns=["couche", "tete"],
                                  values=q).reindex(paires)
            V = piv.to_numpy(dtype=np.float64)           # [paires, cellules]
            p = paired_test(V)
            d = dz(V)
            for j, (cch, tete) in enumerate(piv.columns):
                bloc.append({"famille": fam, "quantite": q, "couche": int(cch),
                             "tete": int(tete), "n": int(np.sum(
                                 np.isfinite(V[:, j]))),
                             "delta_median": float(np.nanmedian(V[:, j])),
                             "dz": float(d[j]), "p": float(p[j]),
                             "part_positive": float(np.nanmean(V[:, j] > 0))})
        b = pd.DataFrame(bloc)
        b["q_bh"] = bh_fdr(b.p.to_numpy())
        b["signif"] = b.q_bh < FDR_Q
        lignes.append(b)
    return pd.concat(lignes, ignore_index=True)


def agregat(Del: pd.DataFrame) -> pd.DataFrame:
    """AGREGAT : mediane sur les tetes par paire, puis le meme test."""
    paires = sorted(Del.pair_id.unique())
    lignes = []
    for fam, qs in FAM.items():
        bloc = []
        for q in qs:
            s = (Del.groupby("pair_id")[q].median().reindex(paires)
                 .to_numpy(dtype=np.float64)[:, None])
            bloc.append({"famille": fam, "quantite": q,
                         "delta_median": float(np.nanmedian(s)),
                         "dz": float(dz(s)[0]), "p": float(paired_test(s)[0]),
                         "part_positive": float(np.nanmean(s > 0))})
        b = pd.DataFrame(bloc)
        b["q_bh"] = bh_fdr(b.p.to_numpy())
        b["signif"] = b.q_bh < FDR_Q
        lignes.append(b)
    return pd.concat(lignes, ignore_index=True)


def resume(Ta: pd.DataFrame, Tb: pd.DataFrame) -> pd.DataFrame:
    """Par quantite : combien de tetes passent BH, et la specificite."""
    a = Ta.set_index(["quantite", "couche", "tete"])
    b = Tb.set_index(["quantite", "couche", "tete"])
    inter = a.index.intersection(b.index)
    a, b = a.loc[inter], b.loc[inter]
    rap = a.dz.abs() / np.clip(b.dz.abs(), DZ_PLANCHER, None)
    g = pd.DataFrame({"famille": a.famille, "dz_adv": a.dz, "dz_ben": b.dz,
                      "signif_adv": a.signif, "signif_ben": b.signif,
                      "rapport": rap,
                      "delta_adv": a.delta_median}).reset_index()
    out = []
    for q, d in g.groupby("quantite", sort=False):
        j = d.dz_adv.abs().idxmax()
        out.append({
            "famille": d.famille.iloc[0], "quantite": q,
            "n_tetes": len(d),
            "tetes_signif_adv": int(d.signif_adv.sum()),
            "tetes_signif_ben": int(d.signif_ben.sum()),
            "dz_adv_max": float(d.loc[j, "dz_adv"]),
            "tete_dz_max": int(d.loc[j, "tete"]),
            "dz_ben_meme_tete": float(d.loc[j, "dz_ben"]),
            "rapport_meme_tete": float(d.loc[j, "rapport"]),
            "dz_adv_median": float(d.dz_adv.median()),
            "rapport_median": float(d.rapport.median()),
        })
    R = pd.DataFrame(out)
    R["quantite"] = pd.Categorical(R.quantite, QUANTITES, ordered=True)
    return R.sort_values(["famille", "quantite"]).reset_index(drop=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    tags = [d for d in sorted(os.listdir(racine))
            if os.path.isfile(os.path.join(racine, d, "mesures.parquet"))]
    if not tags:
        print(f"rien dans {racine}")
        return 1
    for tag in tags:
        dd = os.path.join(racine, tag)
        D = pd.read_parquet(os.path.join(dd, "mesures.parquet"))
        man = json.load(open(os.path.join(dd, "manifeste.json")))
        print("=" * 74)
        print(f"{tag}  --  spectral-trust {man['version']}  "
              f"{man['pairs']} paires  couches {man['couches']}  "
              f"lambda_max={man['lambda_max']}")
        Ta = par_tete(deltas(D, ARM_ADV))
        Tb = par_tete(deltas(D, ARM_BEN))
        Aa = agregat(deltas(D, ARM_ADV))
        Ab = agregat(deltas(D, ARM_BEN))
        R = resume(Ta, Tb)
        Ta.to_csv(os.path.join(dd, "par_tete_adv.csv"), index=False)
        Tb.to_csv(os.path.join(dd, "par_tete_benin.csv"), index=False)
        R.to_csv(os.path.join(dd, "resume.csv"), index=False)
        Aa.merge(Ab, on=["famille", "quantite"], suffixes=("_adv", "_ben")
                 ).to_csv(os.path.join(dd, "agregat.csv"), index=False)

        print("\n-- PRIMAIRE, par tete (BH q<0,05 dans chaque famille)")
        print(f"{'quantite':<19}{'tetes+':>7}{'/benin':>7}"
              f"{'dz_max':>9}{'tete':>5}{'dz_ben':>9}{'rapp':>7}"
              f"{'dz_med':>9}{'rapp_med':>10}")
        fam_cur = None
        for _, r in R.iterrows():
            if r.famille != fam_cur:
                fam_cur = r.famille
                print(f"  [{fam_cur}]")
            print(f"{r.quantite:<19}{r.tetes_signif_adv:>7}"
                  f"{r.tetes_signif_ben:>7}{r.dz_adv_max:>+9.2f}"
                  f"{r.tete_dz_max:>5}{r.dz_ben_meme_tete:>+9.2f}"
                  f"{r.rapport_meme_tete:>7.1f}{r.dz_adv_median:>+9.2f}"
                  f"{r.rapport_median:>10.1f}")

        print("\n-- AGREGAT (mediane sur les tetes ; lisibilite seulement)")
        print(f"{'quantite':<19}{'delta_med':>13}{'dz':>8}{'q_BH':>10}"
              f"{'paires+':>9}{'dz_ben':>8}")
        mg = Aa.merge(Ab, on=["famille", "quantite"],
                      suffixes=("_adv", "_ben"))
        mg["quantite"] = pd.Categorical(mg.quantite, QUANTITES, ordered=True)
        for _, r in mg.sort_values(["famille", "quantite"]).iterrows():
            flag = "*" if r.signif_adv else " "
            print(f"{str(r.quantite):<19}{r.delta_median_adv:>+13.4e}"
                  f"{r.dz_adv:>+8.2f}{r.q_bh_adv:>10.2e}{flag}"
                  f"{r.part_positive_adv:>8.0%}{r.dz_ben:>+8.2f}")
        print(f"\n-> {dd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
