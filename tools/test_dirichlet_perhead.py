"""Controles du chemin par tete de l'energie de Dirichlet.

Quatre controles, sur de l'attention REELLE tiree du cache de la campagne
d'energie regionale. Aucun n'ajuste de tolerance apres coup : les seuils
sont poses avant, et deux d'entre eux sont des identites exactes.

1. FORMULE IDENTIQUE. Pour une tete donnee, ``dirichlet_energy`` doit
   rendre exactement ce que rend ``SpectralAnalyzer.compute_dirichlet_energy``
   de la bibliotheque appliquee au laplacien de cette meme tete. Si ce
   controle passe, le module ne reimplemente pas la metrique : il la
   deplace.

2. LINEARITE, normalization='none'. ``L = D - W`` est lineaire en ``W``,
   donc la moyenne des energies par tete egale l'energie du graphe moyen.
   Identite exacte : tolerance 1e-12 en relatif.

3. ECART REEL, normalization='sym'. La meme comparaison doit au contraire
   donner un ecart NON nul -- c'est la quantite que la campagne veut
   mesurer. Un ecart nul ici signalerait que la normalisation par tete
   n'est pas appliquee.

4. INDICE DE LISSAGE BORNE. Pour 'sym', le spectre de L est dans [0, 2],
   donc le quotient de Rayleigh matriciel ``energy / ||X||_F^2`` doit
   rester dans [0, 2] a la precision machine, pour tout signal.

Le signal est tire au hasard : la linearite et l'identite de formule ne
dependent pas de ce que X represente, et un signal aleatoire sollicite
toutes les directions propres plutot que les seules basses frequences d'un
indicateur.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
from spectral_trust.dirichlet import (dirichlet_energy,
                                      dirichlet_energy_all_layers,
                                      head_aggregation_gap,
                                      per_head_laplacians)

DEFAULT_NPZ = os.path.join(
    "results", "opi", "region_energy", "gemma-2-2b", "raw",
    "combine__sentiment__spam__r718", "C4.npz")

TOL_FORMULA = 1e-10      # float32 impose par SpectralAnalyzer.analyze_layer
TOL_LINEARITY = 1e-12    # identite exacte en float64
MIN_SYM_GAP = 1e-6       # l'ecart 'sym' doit etre franchement non nul


def ctor(norm: str) -> GraphConstructor:
    return GraphConstructor(GSPConfig(normalization=norm,
                                      symmetrization="symmetric",
                                      remove_self_loops=False))


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    npz = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NPZ
    if not os.path.exists(npz):
        print(f"cache absent : {npz}")
        return 2
    z = np.load(npz, allow_pickle=True)
    layers = sorted(k for k in z.files if k.startswith("layer_"))
    A0 = z[layers[0]].astype(np.float64)
    H, T, _ = A0.shape
    d = 16
    X = np.random.default_rng(2027).standard_normal((T, d))
    print(f"cache {npz}\n  {len(layers)} couches, {H} tetes, {T} tokens, "
          f"signal [{T}, {d}]\n")

    ok = True

    # -- 1. formule identique a celle de la bibliotheque, tete par tete
    an = SpectralAnalyzer(GSPConfig())
    worst = 0.0
    for norm in ("none", "sym"):
        c = ctor(norm)
        mine = dirichlet_energy(c, A0, X)
        L = per_head_laplacians(c, A0).cpu().numpy()
        for h in range(H):
            ref = an.compute_dirichlet_energy(X, L[h])
            worst = max(worst, abs(mine[h] - ref) / max(abs(ref), 1e-30))
    print(f"1. formule vs SpectralAnalyzer.compute_dirichlet_energy")
    print(f"   ecart relatif maximal sur {2 * H} tetes : {worst:.3e}"
          f"   (tolerance {TOL_FORMULA:.0e})")
    if worst > TOL_FORMULA:
        ok = False
        print("   ECHEC")

    # -- 2 et 3. ce que le passage par tete change, selon la normalisation
    for norm, expect in (("none", "nul"), ("sym", "non nul")):
        c = ctor(norm)
        rels = [head_aggregation_gap(c, z[k].astype(np.float64), X)["rel"]
                for k in layers]
        rels = np.asarray(rels)
        print(f"\n{2 if norm == 'none' else 3}. normalization={norm!r} : "
              f"moyenne des energies par tete vs energie du graphe moyen")
        print(f"   ecart relatif : max {rels.max():.3e}   median "
              f"{np.median(rels):.3e}   (attendu {expect})")
        if norm == "none":
            if rels.max() > TOL_LINEARITY:
                ok = False
                print(f"   ECHEC : l'identite L = D - W lineaire en W est "
                      f"violee au-dela de {TOL_LINEARITY:.0e}")
            else:
                print("   identite verifiee : passer par les tetes ne change "
                      "rien pour le laplacien combinatoire")
        else:
            if rels.max() < MIN_SYM_GAP:
                ok = False
                print("   ECHEC : ecart quasi nul, la normalisation par tete "
                      "ne semble pas appliquee")
            else:
                print("   ecart reel : c'est ce que la moyenne prealable des "
                      "matrices efface")

    # -- 4. borne du quotient de Rayleigh pour 'sym'
    att = [z[k].astype(np.float64) for k in layers]
    sig = [X] * len(layers)
    r = dirichlet_energy_all_layers(ctor("sym"), att, sig)
    lo, hi = float(r.smoothness.min()), float(r.smoothness.max())
    print(f"\n4. indice de lissage sur 'sym', carte [{r.smoothness.shape[0]}, "
          f"{r.smoothness.shape[1]}]")
    print(f"   intervalle observe : [{lo:.6f}, {hi:.6f}]   (borne theorique "
          f"[0, 2])")
    if lo < -1e-9 or hi > 2 + 1e-9:
        ok = False
        print("   ECHEC : hors de [0, 2]")

    hs = r.head_std["smoothness"]
    hm = r.head_mean["smoothness"]
    print(f"\n   dispersion entre tetes, mediane de l'ecart-type : "
          f"{np.median(hs):.6f}  pour une moyenne de {np.median(hm):.6f}"
          f"   ({100 * np.median(hs) / max(np.median(hm), 1e-30):.1f} %)")

    print("\n" + ("TOUS LES CONTROLES PASSENT" if ok else "AU MOINS UN ECHEC"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
