"""H-D : la FORME de la trajectoire inter-couches bat-elle l'instantane ?

Le papier affirme que la morphologie de la trajectoire porte 5,7x plus
d'information stable qu'un instantane d'une seule couche. Il l'etablit avec
un classifieur entraine (regression logistique / GBT, 5-fold CV) sur lambda_2.

Ici, test SANS ENTRAINEMENT, et sur le bon observable. Pour chaque tete, la
suite des valeurs a travers les couches est resumee par six descripteurs
DECLARES AVANT MESURE, puis chaque descripteur est teste en apparie C4-C3
exactement comme n'importe quelle autre metrique.

    total_variation     sum |x_{l+1} - x_l|
    slope               pente de la regression lineaire sur l'indice de couche
    late_minus_early    moyenne du dernier tiers - moyenne du premier tiers
    curvature           moyenne |x_{l+1} - 2 x_l + x_{l-1}|
    range               max - min
    argmin_rel          argmin(x) / (L-1)   ou se situe le creux

COMPARAISON EQUITABLE. Pour chaque tete, on oppose l'effet du meilleur
descripteur de trajectoire a l'effet de la MEILLEURE couche de cette meme
tete. Le maximum sur les couches est optimiste pour le camp << instantane >>
-- il profite d'une selection post-hoc que le camp trajectoire n'a pas. Si
la trajectoire gagne malgre ce handicap, la conclusion est solide.

Metriques testees : h1_spectral_std (resiste au confond de longueur) et
h1_lambda_2 (le choix du papier, colineaire au confond).
"""

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from spectral_injection.opi.stats import paired_differences, sign_flip_test

MODELS = [("llama3.2-1b", "results/opi/pilot"),
          ("qwen2.5-1.5b", "results/opi/multi/qwen2.5-1.5b"),
          ("gemma-2-2b", "results/opi/multi/gemma-2-2b"),
          ("llama3.2-3b", "results/opi/multi/llama3.2-3b")]
METRICS = ["h1_spectral_std", "h1_lambda_2"]
DESCRIPTORS = ("total_variation", "slope", "late_minus_early",
               "curvature", "range", "argmin_rel")


def descriptors(X: np.ndarray) -> dict:
    """``X`` [n, L, H] -> dict de [n, H]. L'axe des couches est resume."""
    n, L, H = X.shape
    d1 = np.diff(X, axis=1)
    idx = np.arange(L, dtype=float)
    idx_c = idx - idx.mean()
    t = max(L // 3, 1)
    return {
        "total_variation": np.abs(d1).sum(1),
        "slope": (X * idx_c[None, :, None]).sum(1) / (idx_c ** 2).sum(),
        "late_minus_early": X[:, -t:].mean(1) - X[:, :t].mean(1),
        "curvature": np.abs(np.diff(X, n=2, axis=1)).mean(1),
        "range": X.max(1) - X.min(1),
        "argmin_rel": np.argmin(X, axis=1) / max(L - 1, 1),
    }


def dz(D: np.ndarray) -> np.ndarray:
    """Taille d'effet appariee par colonne, sd=0 traite comme degenere."""
    mu, sd = np.nanmean(D, 0), np.nanstd(D, 0, ddof=1)
    return np.where(sd > 1e-12, mu / np.where(sd > 1e-12, sd, 1.0), 0.0)


def main() -> int:
    for metric in METRICS:
        print("\n" + "=" * 104)
        print(f"H-D  --  {metric}")
        print("=" * 104)
        for lab, path in MODELS:
            d = np.load(f"{path}/values.npz", allow_pickle=True)
            names = [str(x) for x in d["metrics"]]
            conds = [str(x) for x in d["conditions"]]
            V = d["values"]
            n, _, L, H, _ = V.shape
            j = names.index(metric)
            iC4, iC3 = conds.index("C4"), conds.index("C3")
            X4, X3 = V[:, iC4, :, :, j], V[:, iC3, :, :, j]       # [n, L, H]

            # --- camp INSTANTANE : chaque (couche, tete) separement
            Dp = (X4 - X3).reshape(n, L * H)
            dz_point = np.abs(dz(Dp)).reshape(L, H)
            best_layer_dz = dz_point.max(0)                        # [H], max sur couches

            # --- camp TRAJECTOIRE : descripteurs par tete
            d4, d3 = descriptors(X4), descriptors(X3)
            dz_traj = np.stack([np.abs(dz(d4[k] - d3[k])) for k in DESCRIPTORS])  # [6, H]
            best_traj_dz = dz_traj.max(0)

            # --- tests avec correction, famille = tous les descripteurs x tetes
            A = np.stack([np.stack([d3[k], d4[k]], 1) for k in DESCRIPTORS], -1)  # [n,2,H,6]
            A = A.reshape(n, 2, H * len(DESCRIPTORS))
            nm = [f"{k}_H{h:02d}" for k in DESCRIPTORS for h in range(H)]
            A = np.stack([np.stack([d3[k][:, h], d4[k][:, h]], 1)
                          for k in DESCRIPTORS for h in range(H)], -1)
            pdv = paired_differences(A, nm, ["C3", "C4"], "C4", "C3")
            r = sign_flip_test(pdv, n_perm=4000, seed=0)
            pf = np.array([x["p_fwer"] for x in r])
            win = int((best_traj_dz > best_layer_dz).sum())
            print(f"\n--- {lab}  ({L} couches x {H} tetes, n={n})")
            print(f"    |dz| median : meilleure couche {np.median(best_layer_dz):.3f}"
                  f"   |   meilleur descripteur {np.median(best_traj_dz):.3f}")
            print(f"    tetes ou la TRAJECTOIRE bat la meilleure couche : {win}/{H}"
                  f"   ({win/H:.0%})")
            print(f"    descripteurs significatifs (FWER<.05) : {int((pf<.05).sum())}/{len(nm)}")
            byd = {k: int((pf.reshape(len(DESCRIPTORS), H)[i] < .05).sum())
                   for i, k in enumerate(DESCRIPTORS)}
            print(f"    par descripteur : {byd}")
            dzm = {k: float(np.median(dz_traj[i])) for i, k in enumerate(DESCRIPTORS)}
            print(f"    |dz| median par descripteur : "
                  + ", ".join(f"{k} {v:.3f}" for k, v in dzm.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
