"""Confirm on the test split the GSP cells selected on train.

Only the cells that survived the family-wise correction on ``train`` are
carried over, and Holm is applied over that small set. The attack categories
of the two splits are disjoint, so a cell that survives here survives out of
distribution.

A cell counts as confirmed only if it is Holm-significant AND keeps the sign
it had on train: a reversal is not a confirmation.

    python -m spectral_injection.gsp_confirm TRAIN_DIR TEST_DIR
"""

import os
import sys

import numpy as np

from .gsp_hypotheses import BLIND, NAMES
from .specificity import family_stats, holm

PREDICTED = {            # direction the hypothesis asked for, +1 / -1
    "bipartite_lmax": +1,   # 8  : closer to 2 = more bipartite
    "edge_entropy": -1,     # 9  : lower entropy
    "radius": -1,           # 15 : contraction
    "kcore_max": +1,        # 11 : denser core
    "motif_trans": +1,      # 14 : motif deviation
    "spec_wass": +1,        # 13 : larger inter-layer divergence
    "betw_ratio": +1,       # 5  : span is a bottleneck
    "betw_boundary": +1,    # 5  : boundary tokens are the bridges
    "kcore_ratio": +1,      # 11 : span forms the core
    "eigcent_ratio": +1,    # 6  : centrality migrates to the span
}


def load(path):
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    return d["values"], [str(x) for x in d["names"]]


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    Vtr, names = load(a[0])
    Vte, _ = load(a[1])
    Ntr, _, L, M = Vtr.shape
    Nte = Vte.shape[0]
    print(f"train {Ntr} paires  ·  test {Nte} paires  ·  {L} couches x {M} mesures\n")

    dif = lambda V: (V[:, 0] - V[:, 1]).reshape(V.shape[0], -1)
    Dtr, Dte = dif(Vtr), dif(Vte)
    midx = np.tile(np.arange(M), L)
    lidx = np.repeat(np.arange(L), M)

    fam = np.isfinite(Dtr).all(0) & np.isfinite(Dte).all(0)
    st = family_stats(Dtr[:, fam], n_perm=5000, seed=0)
    cells = np.flatnonzero(fam)
    keep = st["p_fwer"] < 0.05
    sel, dz_tr = cells[keep], st["dz"][keep]
    order = np.argsort(-np.abs(dz_tr))
    sel, dz_tr = sel[order], dz_tr[order]
    if sel.size == 0:
        print("aucune cellule selectionnee sur train")
        return 1

    stc = family_stats(Dte[:, sel], n_perm=5000, seed=1)
    p_holm = holm(stc["p_perm"])

    print(f"{'mesure':<16}{'type':>8}{'couche':>8}{'dz train':>10}"
          f"{'dz test':>9}{'p_holm':>9}{'sens predit':>13}{'confirme':>10}")
    print("-" * 83)
    n_conf = 0
    for k, c in enumerate(sel):
        nm = names[midx[c]]
        same = np.sign(dz_tr[k]) == np.sign(stc["dz"][k])
        conf = bool(p_holm[k] < 0.05 and same)
        n_conf += conf
        asked = "oui" if np.sign(stc["dz"][k]) == PREDICTED[nm] else "NON"
        print(f"{nm:<16}{('aveugle' if nm in BLIND else 'etiq.'):>8}"
              f"{lidx[c]:>8}{dz_tr[k]:>+10.2f}{stc['dz'][k]:>+9.2f}"
              f"{p_holm[k]:>9.4f}{asked:>13}{('  oui' if conf else '  non'):>10}")

    print(f"\n{n_conf}/{len(sel)} cellules confirmees hors distribution "
          f"(Holm < .05 et meme signe que sur train).")
    print("'sens predit' dit si l'effet confirme va dans la direction que "
          "l'hypothese annoncait, ce qui est une question distincte.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
