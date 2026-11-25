"""Spectre complet de L_sym a une profondeur relative donnee, PAR TETE.

POURQUOI UNE SEULE COUCHE. Les profils de profondeur montrent que le
re-ciblage de l'attention s'installe autour du quart de la profondeur. Ce
script ouvre cette couche-la et rend le spectre ENTIER de chaque tete, au
lieu d'un resume scalaire. L'indice de couche est calcule comme
``round(frac * (L - 1))`` et ecrit dans la sortie : il n'est pas devine.

POURQUOI LA COMPARAISON EST PROPRE. Le protocole impose Delta N = 0, donc
C4 et C3 ont EXACTEMENT le meme nombre de tokens. Les deux laplaciens ont
donc la meme taille, leurs spectres ont le meme nombre de valeurs propres,
et la comparaison indice par indice est definie sans interpolation ni
reechantillonnage. C'est ce qui rend le Delta par rang lisible.

AUCUNE MOYENNE ENTRE TETES. Chaque tete donne son propre graphe, son
propre laplacien et son propre spectre. Les sorties sont indexees par
(paire, bras, tete, rang). Toute agregation est laissee a l'appelant.

L'ALGEBRE VIENT DE LA BIBLIOTHEQUE. ``symmetrize_attention`` et
``construct_laplacian`` de ``GraphConstructor``, diagonalisation par
``SpectralAnalyzer.compute_eigendecomposition``. Entree castee en float64.

Entree : les dossiers ``raw/`` d'une campagne (sauvegardes par
``run_region_energy.py --save-raw N``), donc aucune passe de modele.

    python tools/spectrum_at_depth.py --root results/opi/region_energy \\
        --frac 0.25
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer

#: Bandes de [0, 2], qui borne le spectre de L_sym. Declarees, pas ajustees.
BANDS = ((0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0 + 1e-9))
BAND_NAMES = ("band_0_05", "band_05_1", "band_1_15", "band_15_2")
ZERO_TOL = 1e-8


def spectra_for_layer(A_layer: np.ndarray, gc: GraphConstructor,
                      an: SpectralAnalyzer) -> np.ndarray:
    """``A`` [H, T, T] -> valeurs propres triees [H, T]."""
    A = torch.as_tensor(np.asarray(A_layer, dtype=np.float64))
    W = gc.symmetrize_attention(A.unsqueeze(0))[0]
    L = gc.construct_laplacian(W).cpu().numpy()
    out = []
    for h in range(L.shape[0]):
        vals, _ = an.compute_eigendecomposition(L[h])
        out.append(np.sort(np.asarray(vals, dtype=np.float64)))
    return np.stack(out)


def descriptors(vals: np.ndarray) -> Dict[str, float]:
    """Resumes du spectre d'UNE tete. Rendus a cote du spectre, pas a sa place."""
    n = vals.size
    nz = int((vals <= ZERO_TOL).sum())
    tot = float(vals.sum())
    d = {"n_eigs": n, "n_zero": nz,
         "lambda_2": float(vals[1]) if n > 1 else np.nan,
         "lambda_3": float(vals[2]) if n > 2 else np.nan,
         "lambda_max": float(vals[-1]),
         "eigengap_23": float(vals[2] - vals[1]) if n > 2 else np.nan,
         "trace": tot, "median": float(np.median(vals)),
         "q25": float(np.percentile(vals, 25)),
         "q75": float(np.percentile(vals, 75)),
         "std": float(vals.std())}
    for name, (lo, hi) in zip(BAND_NAMES, BANDS):
        d[name] = float(((vals >= lo) & (vals < hi)).sum()) / n
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join("results", "opi",
                                                   "region_energy"))
    ap.add_argument("--frac", type=float, default=0.25,
                    help="profondeur relative, 0 = entree, 1 = sortie")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    gc = GraphConstructor(GSPConfig(symmetrization="symmetric",
                                    normalization="sym",
                                    remove_self_loops=False))
    # eigen_solver='dense' est OBLIGATOIRE ici. Par defaut la bibliotheque
    # utilise ARPACK au-dela de 50 tokens et ne renvoie alors que les
    # ``num_eigenvalues // 2`` plus petites valeurs propres et autant de
    # plus grandes, concatenees : le MILIEU du spectre est absent, et une
    # comparaison par rang porterait sur deux queues recollees. Le dense
    # rend les T valeurs propres.
    an = SpectralAnalyzer(GSPConfig(eigen_solver="dense"))

    tags = [t for t in sorted(os.listdir(args.root))
            if os.path.isdir(os.path.join(args.root, t, "raw"))]
    rows, desc = [], []
    for t in tags:
        rd = os.path.join(args.root, t, "raw")
        pids = sorted(os.listdir(rd))
        for pid in pids:
            d = os.path.join(rd, pid)
            arms = sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))
            layer = None
            for arm in arms:
                z = np.load(os.path.join(d, arm + ".npz"), allow_pickle=True)
                keys = sorted(k for k in z.files if k.startswith("layer_"))
                L = len(keys)
                layer = int(round(args.frac * (L - 1)))
                S = spectra_for_layer(z[keys[layer]], gc, an)
                H, T = S.shape
                rows.append(pd.DataFrame({
                    "model": t, "pair_id": pid, "arm": arm, "n_layers": L,
                    "couche": layer, "frac": args.frac,
                    "tete": np.repeat(np.arange(H), T),
                    "rang": np.tile(np.arange(T), H),
                    "lambda": S.ravel()}))
                for h in range(H):
                    desc.append(dict(model=t, pair_id=pid, arm=arm, tete=h,
                                     n_layers=L, couche=layer,
                                     **descriptors(S[h])))
            print(f"  {t:14s} {pid[:38]:38s} couche {layer}/{L - 1} "
                  f"({args.frac:.2f})  bras {len(arms)}")

    if not rows:
        raise SystemExit(f"aucun dossier raw/ sous {args.root} ; relancer la "
                         f"campagne avec --save-raw N")

    out = args.out or os.path.join(args.root,
                                   f"spectrum_frac{args.frac:.2f}".replace(".", "p"))
    os.makedirs(out, exist_ok=True)
    SP = pd.concat(rows, ignore_index=True)
    DE = pd.DataFrame(desc)
    SP.to_parquet(os.path.join(out, "spectra_long.parquet"), index=False)
    DE.to_csv(os.path.join(out, "descriptors.csv"), index=False)

    # -- Delta apparie par RANG : meme taille entre bras grace a Delta N = 0
    print("\n" + "=" * 78)
    print(f"Delta apparie C4 - C3_opi du spectre, par rang, couche a "
          f"{100 * args.frac:.0f} % de la profondeur")
    print("=" * 78)
    piv = SP.pivot_table(index=["model", "pair_id", "tete", "rang"],
                         columns="arm", values="lambda")
    if "C4" in piv.columns and "C3_opi" in piv.columns:
        dd = (piv["C4"] - piv["C3_opi"]).rename("_d").reset_index()
        for m, g in dd.groupby("model"):
            n = g.rang.max() + 1
            qs = [1, int(.25 * n), int(.5 * n), int(.75 * n), n - 1]
            s = "  ".join(f"r{q}:{g[g.rang == q]['_d'].median():+.4f}" for q in qs)
            print(f"  {m:14s} T={n:3d}  {s}")
            frac = (g.groupby("tete")["_d"].median() > 0).mean()
            print(f"                 tetes dont le Delta median est positif : "
                  f"{100 * frac:.0f} %")

    print(f"\necrits : {out}/spectra_long.parquet, {out}/descriptors.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
