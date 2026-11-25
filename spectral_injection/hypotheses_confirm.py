"""Confirm on the test split the cells selected on train.

The attack categories of train and test are disjoint, so this is an
out-of-distribution confirmation by construction. Only the cells that
survived the family-wise correction on train are carried over, and Holm is
applied over that small set -- selecting on train and reporting the same
number on train would be circular.
"""

import os, sys
import numpy as np

from .hypotheses import NAMES
from .hypotheses_report import PREDICTED, TESTED, auroc
from .specificity import family_stats, holm


def load(path):
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    return d["values"], list(d["layers"])


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    train_dir, test_dir, top = a[0], a[1], int(a[2]) if len(a) > 2 else 12
    Vtr, layers = load(train_dir)
    Vte, _ = load(test_dir)
    Ntr, _, L, H, M = Vtr.shape
    Nte = Vte.shape[0]
    print(f"train {Ntr} paires · test {Nte} paires · {L}x{H} cellules\n")

    Dtr = (Vtr[:, 0] - Vtr[:, 1]).reshape(Ntr, -1)
    Dte = (Vte[:, 0] - Vte[:, 1]).reshape(Nte, -1)
    midx = np.tile(np.arange(M), L * H).reshape(L, H, M).ravel()
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)

    fam = (np.isfinite(Dtr).all(0) & np.isfinite(Dte).all(0)
           & np.isin(midx, [NAMES.index(m) for m in TESTED]))
    st = family_stats(Dtr[:, fam], n_perm=5000, seed=0)
    cells = np.flatnonzero(fam)
    # Selection on train only: FWER-significant, ranked by |dz|.
    sel = cells[(st["p_fwer"] < 0.05)]
    sel_dz = st["dz"][st["p_fwer"] < 0.05]
    order = np.argsort(-np.abs(sel_dz))[:top]
    sel, sel_dz = sel[order], sel_dz[order]
    if sel.size == 0:
        print("aucune cellule selectionnee sur train"); return 1

    stc = family_stats(Dte[:, sel], n_perm=5000, seed=1)
    p_holm = holm(stc["p_perm"])
    lab = np.r_[np.zeros(Nte), np.ones(Nte)]
    fi = Vte[:, 0].reshape(Nte, -1); fc = Vte[:, 1].reshape(Nte, -1)

    print(f"{'hypothese':<15}{'c':>3}{'t':>3}{'dz_train':>10}{'dz_test':>9}"
          f"{'p_holm':>9}{'AUROC_te':>10}{'signe ok':>10}")
    print("-" * 70)
    n_conf = 0
    for k, c in enumerate(sel):
        au = auroc(lab, np.r_[fc[:, c], fi[:, c]])
        au_or = max(au, 1 - au)
        same = np.sign(sel_dz[k]) == np.sign(stc["dz"][k])
        conf = bool(p_holm[k] < 0.05 and same)
        n_conf += conf
        print(f"{NAMES[midx[c]]:<15}{layers[lidx[c]]:>3}{hidx[c]:>3}"
              f"{sel_dz[k]:>+10.2f}{stc['dz'][k]:>+9.2f}{p_holm[k]:>9.4f}"
              f"{au_or:>10.3f}{'  oui' if conf else '  non':>10}")
    print(f"\n{n_conf}/{len(sel)} cellules confirmees hors distribution "
          f"(Holm < .05 et meme signe).")

    # Per-hypothesis summary, direction-aware, on test only.
    print("\nPar hypothese, sur test, toutes cellules (pas de selection) :")
    print(f"{'hypothese':<15}{'% bon signe':>13}{'dz median':>11}"
          f"{'AUROC max':>11}")
    for m in TESTED:
        s = np.flatnonzero(fam & (midx == NAMES.index(m)))
        stm = family_stats(Dte[:, s], n_perm=1000, seed=2)
        frac = float(np.mean(np.sign(stm["dz"]) == PREDICTED[m]))
        aus = [max(x, 1 - x) for x in
               (auroc(lab, np.r_[fc[:, c], fi[:, c]]) for c in s)]
        print(f"{m:<15}{frac*100:>12.0f}%{np.median(stm['dz']):>+11.2f}"
              f"{max(aus):>11.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
