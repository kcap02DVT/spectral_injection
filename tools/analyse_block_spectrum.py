"""Le spectre de l'union de deux regions bouge-t-il, et specifiquement ?

TROIS BRAS, DEUX CONTRASTES. ``C3_opi`` est la reference.
    contraste injection  :  C4       - C3_opi
    contraste benin      :  C3_wiki  - C3_opi
Un deplacement present dans les deux n'apprend rien sur l'injection : il
se produit des qu'on change le texte du slot. Le rapport des |dz| est la
mesure de specificite, avec un plancher a 0,05 sur le denominateur.

QUANTITES, declarees : sur chaque cellule (paire, couche, tete, bloc),
  frac_high  part de l'energie a lambda >= 1   (lecture declaree du depot)
  m0         masse du mode trivial
  D          distance de Kolmogorov entre les deux cumulees de la cellule
Les valeurs propres sont ramenees dans [0, 2] avant tout calcul : le mode
trivial vaut +-1e-17 et une evaluation en lambda = 0 compterait ce mode
d'un seul cote.

AMPLITUDE CONTRE SIGNE. Pour chaque bloc on rend |Delta| median (de
combien une cellule bouge) ET Delta median signe (de combien l'ensemble se
deplace). Leur rapport chiffre l'annulation entre cellules, qui est le
phenomene dominant de ce depot.

BENJAMINI-HOCHBERG q = 0,05 sur l'ensemble des tests d'un contraste, une
seule famille (hybride). Les cellules a ecart identiquement nul sont
exclues : un test sur un vecteur nul n'est pas un test.

    python tools/analyse_block_spectrum.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import bh_fdr, dz, paired_test       # noqa: E402
from analyse_region_spectrum import _kolmogorov                 # noqa: E402

RACINE = os.path.join("results", "opi", "block_spectrum")
ARM_REF, ARM_ADV, ARM_BEN = "C3_opi", "C4", "C3_wiki"
HIGH_CUT = 1.0
FDR_Q = 0.05
DZ_PLANCHER = 0.05
TOL = 1e-12


def _cells(d):
    out = {}
    for k, g in d.groupby(["couche", "tete", "bloc"], sort=False):
        g = g.sort_values("rang")
        lam = np.clip(g["lambda"].to_numpy(np.float64), 0.0, 2.0)
        mf = g["mass_frac"].to_numpy(np.float64)
        out[k] = (lam, mf)
    return out


def contraste(P: pd.DataFrame, alt: str) -> pd.DataFrame:
    """Une ligne par (paire, couche, tete, bloc) : alt - ref."""
    lignes = []
    for pid, gp in P.groupby("pair_id", sort=True):
        A = _cells(gp[gp.arm == alt])
        B = _cells(gp[gp.arm == ARM_REF])
        for k in sorted(set(A) & set(B)):
            la, ma = A[k]
            lb, mb = B[k]
            if la.size != lb.size:
                continue
            fa = float(ma[la >= HIGH_CUT].sum())
            fb = float(mb[lb >= HIGH_CUT].sum())
            lignes.append({
                "pair_id": pid, "couche": int(k[0]), "tete": int(k[1]),
                "bloc": k[2], "n_union": int(la.size),
                "D_kolmo": _kolmogorov(la, ma, lb, mb),
                "delta_m0": float(ma[0] - mb[0]),
                "delta_frac_high": fa - fb,
                "frac_high_ref": fb})
    return pd.DataFrame(lignes)


def tests(C: pd.DataFrame, quantite: str) -> pd.DataFrame:
    paires = sorted(C.pair_id.unique())
    piv = C.pivot_table(index="pair_id", columns=["bloc", "couche", "tete"],
                        values=quantite).reindex(paires)
    V = piv.to_numpy(np.float64)
    nul = np.nanmax(np.abs(V), axis=0) <= TOL
    p = np.full(V.shape[1], np.nan)
    d = np.full(V.shape[1], np.nan)
    if (~nul).any():
        p[~nul] = paired_test(V[:, ~nul])
        d[~nul] = dz(V[:, ~nul])
    T = pd.DataFrame({"bloc": [c[0] for c in piv.columns],
                      "couche": [int(c[1]) for c in piv.columns],
                      "tete": [int(c[2]) for c in piv.columns],
                      "nul_exact": nul, "dz": d, "p": p,
                      "delta_median": np.nanmedian(V, axis=0)})
    T["q_bh"] = np.nan
    m = ~T.nul_exact & T.p.notna()
    if m.any():
        T.loc[m, "q_bh"] = bh_fdr(T.loc[m, "p"].to_numpy())
    T["signif"] = T.q_bh < FDR_Q
    return T


def resume(Ca, Cb, Ta, Tb, bloc_declare):
    a = Ta.set_index(["bloc", "couche", "tete"])
    b = Tb.set_index(["bloc", "couche", "tete"])
    j = a.index.intersection(b.index)
    rap = a.loc[j].dz.abs() / np.clip(b.loc[j].dz.abs(), DZ_PLANCHER, None)
    G = pd.DataFrame({"dz_adv": a.loc[j].dz, "dz_ben": b.loc[j].dz,
                      "sig_adv": a.loc[j].signif,
                      "sig_ben": b.loc[j].signif,
                      "rapport": rap}).reset_index()
    out = []
    for bl, g in G.groupby("bloc"):
        ca = Ca[Ca.bloc == bl]
        cb = Cb[Cb.bloc == bl]
        s = ca.delta_frac_high.median()
        out.append({
            "bloc": bl, "declare": bl == bloc_declare,
            "n_cellules": len(g),
            "sig_adv": int(g.sig_adv.sum()), "sig_ben": int(g.sig_ben.sum()),
            "D_adv": float(ca.D_kolmo.median()),
            "D_ben": float(cb.D_kolmo.median()),
            "abs_dfh_adv": float(ca.delta_frac_high.abs().median()),
            "dfh_adv": float(s),
            "annulation": (float(ca.delta_frac_high.abs().median() / abs(s))
                           if abs(s) > 1e-12 else np.inf),
            "dz_med_adv": float(g.dz_adv.median()),
            "dz_med_ben": float(g.dz_ben.median()),
            "rapport_med": float(g.rapport.median())})
    R = pd.DataFrame(out).sort_values("sig_adv", ascending=False)
    return R.reset_index(drop=True)


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    tags = [t for t in sorted(os.listdir(racine))
            if os.path.isfile(os.path.join(racine, t, "blocs_long.parquet"))]
    for tag in tags:
        dd = os.path.join(racine, tag)
        prov = json.load(open(os.path.join(dd, "provenance.json"),
                              encoding="utf-8"))
        bd = prov["bloc_declare"]
        P = pd.read_parquet(os.path.join(dd, "blocs_long.parquet"))
        print("=" * 96)
        print(f"{tag}  --  {prov['n_pairs_kept']} paires  --  "
              f"{len(prov['blocs'])} blocs  --  bloc declare {bd}")
        Ca = contraste(P, ARM_ADV)
        Cb = contraste(P, ARM_BEN)
        Ta = tests(Ca, "delta_frac_high")
        Tb = tests(Cb, "delta_frac_high")
        R = resume(Ca, Cb, Ta, Tb, bd)
        Ca.to_csv(os.path.join(dd, "cellules_adv.csv"), index=False)
        Cb.to_csv(os.path.join(dd, "cellules_benin.csv"), index=False)
        Ta.to_csv(os.path.join(dd, "tests_adv.csv"), index=False)
        R.to_csv(os.path.join(dd, "resume_blocs.csv"), index=False)

        print(f"\n{'bloc':<20}{'cell. BH inj':>13}{'benin':>8}"
              f"{'D inj':>8}{'D ben':>8}{'|d fh|':>9}{'d fh':>10}"
              f"{'annul':>7}{'rapp dz':>9}")
        for _, r in R.iterrows():
            mark = " <<" if r.declare else ""
            an = "   --" if not np.isfinite(r.annulation) else f"{r.annulation:>7.1f}"
            print(f"{r.bloc:<20}{r.sig_adv:>7}/{r.n_cellules:<5}"
                  f"{r.sig_ben:>8}{r.D_adv:>8.3f}{r.D_ben:>8.3f}"
                  f"{r.abs_dfh_adv:>9.4f}{r.dfh_adv:>+10.4f}{an}"
                  f"{r.rapport_med:>9.1f}{mark}")
        print("\n   cell. BH   = cellules (couche x tete) retenues a q<0,05")
        print("   D          = distance de Kolmogorov mediane entre cumulees")
        print("   |d fh| / d fh = amplitude par cellule / deplacement signe")
        print("   annul      = leur rapport : eleve = les cellules bougent "
              "en sens contraires")
        print("   rapp dz    = |dz injection| / |dz benin|, median ; "
              "proche de 1 = non specifique")

        Dc = pd.read_csv(os.path.join(dd, "controle_croise.csv"))
        dr = Dc[(Dc.arm == ARM_REF) & (Dc.bloc == bd)]
        if not dr.empty:
            print(f"\n-- aretes croisees de {bd} (bras {ARM_REF}, couche "
                  f"{prov['couche_declaree']})")
            print(f"   poids des aretes croisees : "
                  f"{dr.poids_croise.median():.4f} de la masse du sous-graphe")
            print(f"   les annuler deplace le spectre de D = "
                  f"{dr.D_sans_croise.median():.4f} (median), "
                  f"max |d lambda| = {dr.d_lambda_sans_croise.median():.4f}")
        print(f"\n-> {dd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
