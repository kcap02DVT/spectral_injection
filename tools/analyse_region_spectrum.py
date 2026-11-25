"""Les spectres par region sont-ils IDENTIQUES entre les bras ? Mesure.

L'objection est visuelle : les densites par region se superposent. Ce
script la transforme en nombres, avec trois quantites declarees d'avance.

1. ECART EXACT. Pour chaque cellule (paire, couche, tete, region),
   l'ecart maximal entre les deux spectres bruts :
       d_lambda = max_k |lambda_k^adv - lambda_k^ben|
   Les regions entierement anterieures au slot doivent donner EXACTEMENT
   zero : le sous-graphe induit ne depend que de A[R, R], bit-identique
   par causalite. Ce n'est pas un resultat, c'est un theoreme ; le mesurer
   verifie le code, pas le modele.

2. DISTANCE ENTRE DISTRIBUTIONS. La distance de Kolmogorov entre les deux
   energies cumulees de la MEME cellule :
       D = max_lambda |F_adv(lambda) - F_ben(lambda)|      dans [0, 1]
   C'est la reponse directe a << les spectres sont-ils identiques >> :
   D = 0 veut dire identiques, et l'ordre de grandeur de D dit de combien
   ils different quand ils different. Calculee sur la grille exacte des
   802 valeurs propres reunies des deux bras, sans interpolation.

3. TEST APPARIE sur la part d'energie haute frequence,
   frac_high = somme des mass_frac a lambda >= 1, deja la lecture declaree
   du depot. Wilcoxon apparie bilateral par (region, couche, tete), dz de
   Cohen, Benjamini-Hochberg q = 0,05 sur la vue entiere -- une seule
   famille, hybride. Les cellules dont l'ecart est identiquement nul sont
   EXCLUES du BH : un test sur un vecteur nul n'est pas un test, et les
   garder gonflerait artificiellement le denominateur.

CE QUE CE SCRIPT NE PEUT PAS DIRE. La campagne n'a que deux bras, C3_opi
et C4. Sans le bras C3_wiki il est impossible de savoir si un ecart non nul
est specifique a l'injection ou se produit des qu'on change le texte du
slot. Toute difference trouvee ici est donc une borne SUPERIEURE de ce qui
est attribuable a l'injection.

    python tools/analyse_region_spectrum.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyse_region_energy import bh_fdr, dz, paired_test      # noqa: E402

RACINE = os.path.join("results", "opi", "region_spectrum")
ARM_BEN, ARM_ADV = "C3_opi", "C4"
HIGH_CUT = 1.0
FDR_Q = 0.05
TOL = 1e-12


def _cles(d):
    """-> dict (couche, tete, region) -> (lambda trie, mass_frac trie).

    PIEGE NUMERIQUE, CORRIGE ICI. Le mode trivial du laplacien a
    lambda_0 = 0 en theorie ; ``eigh`` rend -1e-17 dans un bras et +1e-17
    dans l'autre. Evaluee en lambda = 0, une cumulee compte alors ce mode
    et l'autre non, et la distance de Kolmogorov saute a la masse entiere
    du mode trivial -- environ 0,39, soit un artefact pur du SIGNE DU ZERO.
    Les valeurs propres sont donc ramenees dans [0, 2] avant tout calcul.
    """
    out = {}
    for k, g in d.groupby(["couche", "tete", "region"], sort=False):
        g = g.sort_values("rang")
        lam = np.clip(g["lambda"].to_numpy(np.float64), 0.0, 2.0)
        out[k] = (lam, g["mass_frac"].to_numpy(np.float64))
    return out


def _kolmogorov(lam_a, mf_a, lam_b, mf_b):
    """max |F_a - F_b|, exact, sur la reunion des valeurs propres."""
    grille = np.union1d(lam_a, lam_b)
    Fa = np.cumsum(mf_a)[np.searchsorted(lam_a, grille, side="right") - 1]
    Fb = np.cumsum(mf_b)[np.searchsorted(lam_b, grille, side="right") - 1]
    Fa = np.where(np.searchsorted(lam_a, grille, side="right") == 0, 0.0, Fa)
    Fb = np.where(np.searchsorted(lam_b, grille, side="right") == 0, 0.0, Fb)
    return float(np.abs(Fa - Fb).max())


def cellules(P: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par (paire, couche, tete, region) : d_lambda, D, Delta."""
    lignes = []
    for pid, gp in P.groupby("pair_id", sort=True):
        A = _cles(gp[gp.arm == ARM_ADV])
        B = _cles(gp[gp.arm == ARM_BEN])
        for k in sorted(set(A) & set(B)):
            la, ma = A[k]
            lb, mb = B[k]
            if la.size != lb.size:
                continue
            fa = float(ma[la >= HIGH_CUT].sum())
            fb = float(mb[lb >= HIGH_CUT].sum())
            lignes.append({
                "pair_id": pid, "couche": int(k[0]), "tete": int(k[1]),
                "region": k[2], "n_region": int(la.size),
                "d_lambda": float(np.abs(la - lb).max()),
                "d_masse": float(np.abs(ma - mb).max()),
                "D_kolmo": _kolmogorov(la, ma, lb, mb),
                # masse du mode trivial : la composante continue du signal
                # sur la region, ou se realise la distance dans la majorite
                # des cellules de 'slot'
                "m0_ben": float(mb[0]), "m0_adv": float(ma[0]),
                "delta_m0": float(ma[0] - mb[0]),
                "frac_high_ben": fb, "frac_high_adv": fa,
                "delta_frac_high": fa - fb})
    return pd.DataFrame(lignes)


def tests(C: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon apparie par (region, couche, tete). BH sur la vue."""
    paires = sorted(C.pair_id.unique())
    piv = C.pivot_table(index="pair_id",
                        columns=["region", "couche", "tete"],
                        values="delta_frac_high").reindex(paires)
    V = piv.to_numpy(np.float64)
    nul = np.nanmax(np.abs(V), axis=0) <= TOL
    p = np.full(V.shape[1], np.nan)
    d = np.full(V.shape[1], np.nan)
    if (~nul).any():
        p[~nul] = paired_test(V[:, ~nul])
        d[~nul] = dz(V[:, ~nul])
    T = pd.DataFrame({"region": [c[0] for c in piv.columns],
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


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    racine = sys.argv[1] if len(sys.argv) > 1 else RACINE
    tags = [t for t in sorted(os.listdir(racine))
            if os.path.isfile(os.path.join(racine, t, "propre_long.parquet"))]
    for tag in tags:
        dd = os.path.join(racine, tag)
        prov = json.load(open(os.path.join(dd, "provenance.json"),
                              encoding="utf-8"))
        regions = list(prov["regions"])
        print("=" * 86)
        print(f"{tag}  --  {prov['n_pairs_kept']} paires  --  vue 'propre' "
              f"(sous-graphe induit)  --  bras {ARM_BEN} vs {ARM_ADV}")
        P = pd.read_parquet(os.path.join(dd, "propre_long.parquet"))
        C = cellules(P)
        T = tests(C)
        C.to_csv(os.path.join(dd, "cellules.csv"), index=False)
        T.to_csv(os.path.join(dd, "tests_par_cellule.csv"), index=False)

        print(f"\n{'region':<10}{'cellules':>9}{'nul exact':>11}"
              f"{'max d_lambda':>14}{'D median':>11}{'cell. BH':>11}")
        for r in regions:
            c = C[C.region == r]
            if c.empty:
                continue
            t = T[T.region == r]
            nz = int((c.d_lambda <= TOL).sum())
            print(f"{r:<10}{len(c):>9}{nz / len(c):>10.0%}"
                  f"{c.d_lambda.max():>14.3e}{c.D_kolmo.median():>11.4f}"
                  f"{int(t.signif.sum()):>8}/{len(t):<4}")

        # -- AMPLITUDE CONTRE SIGNE. Le coeur de l'objection : une cellule
        #    bouge beaucoup (|Delta| median) alors que la moyenne sur les
        #    cellules ne bouge pas (Delta median signe). Le rapport des deux
        #    chiffre l'annulation.
        print(f"\n-- amplitude par cellule CONTRE deplacement moyen")
        print(f"{'region':<10}{'|d m0| med':>12}{'d m0 med':>11}{'rapp':>7}"
              f"{'|d fh| med':>12}{'d fh med':>11}{'rapp':>7}"
              f"{'% |d fh|>0,01':>15}")
        for r in regions:
            c = C[C.region == r]
            if c.empty:
                continue
            a0, s0 = c.delta_m0.abs().median(), c.delta_m0.median()
            ah, sh = (c.delta_frac_high.abs().median(),
                      c.delta_frac_high.median())
            r0 = a0 / abs(s0) if abs(s0) > 1e-12 else np.inf
            rh = ah / abs(sh) if abs(sh) > 1e-12 else np.inf
            print(f"{r:<10}{a0:>12.4f}{s0:>+11.4f}"
                  f"{('    --' if not np.isfinite(r0) else f'{r0:>7.1f}')}"
                  f"{ah:>12.4f}{sh:>+11.4f}"
                  f"{('    --' if not np.isfinite(rh) else f'{rh:>7.1f}')}"
                  f"{(c.delta_frac_high.abs() > 0.01).mean():>14.0%}")
        print("   |d| = amplitude mediane du changement d'une cellule.")
        print("   d   = changement median signe. Un rapport eleve = les "
              "cellules bougent, mais pas dans le meme sens.")
        print("   ATTENTION : |Delta| exige les DEUX bras. Ce n'est pas une "
              "caracteristique calculable sur un seul prompt.")
        print(f"\n-> {dd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
