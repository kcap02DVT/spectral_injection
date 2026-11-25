"""Le module de spectral-trust reproduit-il le calcul artisanal precedent ?

Le script ``results/opi/targeted_region_energy/run_targeted.py`` calculait
W, M et Q a la main. Le nouveau module passe par ``GraphConstructor``. Si la
reecriture est fidele, les deux doivent coincider a la precision flottante
sur la MEME attention en cache -- pas approximativement, exactement.

Le test rejoue aussi l'identite variance-spectre et verifie que les parts
sommment a 1.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_trust import GSPConfig, GraphConstructor
from spectral_trust.region_energy import (block_names, check_variance_identity,
                                          region_energy_all_layers)

RAW = os.path.join("results", "opi", "targeted_region_energy", "raw")
REG = ["task", "normal", "preamble", "slot", "tail", "template"]
RP = [(REG[i], REG[j]) for i in range(6) for j in range(i, 6)]


def legacy_calc(att, lab):
    """Copie EXACTE de ``calc`` du script precedent, sans aucune retouche."""
    L = len(att)
    H = att[0].shape[0]
    N = len(lab)
    idx = {r: np.flatnonzero(lab == r) for r in REG}
    en = np.zeros((L, H, len(RP)))
    den = np.zeros_like(en)
    var = np.zeros((L, H))
    loop = np.zeros_like(var)
    flow = np.zeros((L, H, 6, 6))
    for l, A0 in enumerate(att):
        A = np.asarray(A0, dtype=np.float64)
        W = .5 * (A + A.transpose(0, 2, 1))
        d = W.sum(-1)
        iv = np.divide(1, np.sqrt(np.maximum(d, 1e-12)),
                       out=np.zeros_like(d), where=d > 1e-12)
        M = W * iv[:, :, None] * iv[:, None, :]
        diag = np.diagonal(M, axis1=1, axis2=2)
        Q = M * M
        ii = np.arange(N)
        Q[:, ii, ii] = 0
        loop[l] = ((diag - diag.mean(axis=1, keepdims=True)) ** 2).sum(axis=1) / N
        var[l] = Q.sum((1, 2)) / N + loop[l]
        for k, (r, s) in enumerate(RP):
            x, y = idx[r], idx[s]
            if not len(x) or not len(y):
                continue
            z = Q[:, x[:, None], y].sum((1, 2))
            poss = len(x) * (len(x) - 1) if r == s else 2 * len(x) * len(y)
            if r != s:
                z += Q[:, y[:, None], x].sum((1, 2))
            en[l, :, k] = z / N
            den[l, :, k] = z / max(poss, 1)
        for i, r in enumerate(REG):
            x = idx[r]
            if not len(x):
                continue
            for j, s in enumerate(REG):
                y = idx[s]
                if len(y):
                    flow[l, :, i, j] = A[:, x, :][:, :, y].sum((1, 2)) / len(x)
    return en, den, var, loop, flow


def main() -> int:
    cfg = GSPConfig(symmetrization="symmetric", normalization="sym",
                    remove_self_loops=False)
    gc = GraphConstructor(cfg)

    pairs = sorted(os.listdir(RAW))
    if not pairs:
        raise SystemExit(f"aucune capture dans {RAW}")
    print(f"{len(pairs)} paires en cache ; controle sur les 3 premieres, "
          f"les deux conditions.\n")

    worst = {"energy": 0.0, "density": 0.0, "variance": 0.0,
             "loop": 0.0, "flow": 0.0, "identity": 0.0, "share_sum": 0.0}
    n_checked = 0
    for pid in pairs[:3]:
        for cond in ("C3", "C4"):
            f = os.path.join(RAW, pid, cond + ".npz")
            if not os.path.exists(f):
                continue
            z = np.load(f, allow_pickle=True)
            keys = sorted(k for k in z.files if k.startswith("layer_"))
            att = [z[k] for k in keys]
            lab = z["regions"]

            legacy = legacy_calc(att, lab)
            new = region_energy_all_layers(gc, att, lab, REG)

            d = {
                "energy": np.abs(legacy[0] - new.energy).max(),
                "density": np.abs(legacy[1] - new.density).max(),
                "variance": np.abs(legacy[2] - new.variance).max(),
                "loop": np.abs(legacy[3] - new.loop).max(),
                "flow": np.abs(legacy[4] - new.flow).max(),
                "identity": check_variance_identity(gc, att[len(att) // 2]),
                "share_sum": np.abs(new.share.sum(-1) - 1.0).max(),
            }
            for k, v in d.items():
                worst[k] = max(worst[k], float(v))
            n_checked += 1
            print(f"  {pid:28s} {cond}  L={new.n_layers} H={new.n_heads} "
                  f"N={new.n_tokens}  ecart max energy {d['energy']:.3e}  "
                  f"flow {d['flow']:.3e}  identite {d['identity']:.3e}")
            del att, z

    print(f"\n{n_checked} captures controlees.")
    print("ecarts maximaux module spectral-trust vs calcul artisanal :")
    for k in ("energy", "density", "variance", "loop", "flow"):
        print(f"    {k:10s} {worst[k]:.3e}")
    print(f"    {'identite':10s} {worst['identity']:.3e}   "
          f"(var(spec L) == offdiag + loop)")
    print(f"    {'parts':10s} {worst['share_sum']:.3e}   (somme des parts - 1)")

    tol = 1e-12
    bad = [k for k in ("energy", "density", "variance", "loop", "flow")
           if worst[k] > tol]
    if bad or worst["identity"] > 1e-12 or worst["share_sum"] > 1e-9:
        print(f"\nECHEC : {bad or 'identite/parts'} depassent {tol:.0e}")
        return 1
    print(f"\nOK : equivalence etablie sous {tol:.0e}. La reecriture sur "
          f"GraphConstructor ne change aucun chiffre.")
    print(f"Noms de blocs ({len(block_names(REG))}) : "
          f"{', '.join(block_names(REG))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
