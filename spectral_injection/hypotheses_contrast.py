"""Contrast the two control conditions on the same cells.

    injected - email_text          "an insertion of this length"  removed
    injected - benign_instruction  "an instruction of this length" removed

A cell whose effect survives only against email prose is detecting the
imperative form, not the attack. A cell that survives both is detecting
something specific to the injection.

Cells are NOT reselected here: the comparison uses the cells that were
selected on the email-text train split and confirmed on its test split, so
no new search happens and no new correction is owed beyond Holm over them.
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
    sel_train, sel_test, bi_train, bi_test = a[0], a[1], a[2], a[3]
    top = int(a[4]) if len(a) > 4 else 12

    Vtr, layers = load(sel_train)
    Vte, _ = load(sel_test)
    Btr, _ = load(bi_train)
    Bte, _ = load(bi_test)
    Ntr, _, L, H, M = Vtr.shape
    print(f"selection email_text: train {Ntr} / test {Vte.shape[0]} paires")
    print(f"contraste benign_instruction: train {Btr.shape[0]} / test "
          f"{Bte.shape[0]} paires\n")

    def diffs(V):
        return (V[:, 0] - V[:, 1]).reshape(V.shape[0], -1)

    Dtr, Dte, Bdtr, Bdte = diffs(Vtr), diffs(Vte), diffs(Btr), diffs(Bte)
    midx = np.tile(np.arange(M), L * H).reshape(L, H, M).ravel()
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)

    fam = (np.isfinite(Dtr).all(0) & np.isfinite(Dte).all(0)
           & np.isfinite(Bdtr).all(0) & np.isfinite(Bdte).all(0)
           & np.isin(midx, [NAMES.index(m) for m in TESTED]))
    st = family_stats(Dtr[:, fam], n_perm=5000, seed=0)
    cells = np.flatnonzero(fam)
    keep = st["p_fwer"] < 0.05
    sel, sel_dz = cells[keep], st["dz"][keep]
    order = np.argsort(-np.abs(sel_dz))[:top]
    sel, sel_dz = sel[order], sel_dz[order]
    if sel.size == 0:
        print("aucune cellule selectionnee"); return 1

    s_te = family_stats(Dte[:, sel], n_perm=5000, seed=1)
    b_tr = family_stats(Bdtr[:, sel], n_perm=5000, seed=2)
    b_te = family_stats(Bdte[:, sel], n_perm=5000, seed=3)
    p_bte = holm(b_te["p_perm"])

    def au_of(V, c):
        n = V.shape[0]
        fi = V[:, 0].reshape(n, -1); fc = V[:, 1].reshape(n, -1)
        x = auroc(np.r_[np.zeros(n), np.ones(n)], np.r_[fc[:, c], fi[:, c]])
        return max(x, 1 - x)

    print(f"{'hypothese':<14}{'c':>3}{'t':>3}"
          f"{'dz prose(te)':>13}{'dz instr(te)':>13}{'retenu %':>10}"
          f"{'p_holm':>9}{'AUROC':>8}{'specifique':>12}")
    print("-" * 85)
    n_spec = 0
    for k, c in enumerate(sel):
        dz_p, dz_i = s_te["dz"][k], b_te["dz"][k]
        frac = dz_i / dz_p if abs(dz_p) > 1e-9 else np.nan
        same = np.sign(dz_p) == np.sign(dz_i)
        spec = bool(p_bte[k] < 0.05 and same)
        n_spec += spec
        print(f"{NAMES[midx[c]]:<14}{layers[lidx[c]]:>3}{hidx[c]:>3}"
              f"{dz_p:>+13.2f}{dz_i:>+13.2f}{100*frac:>9.0f}%"
              f"{p_bte[k]:>9.4f}{au_of(Bte, c):>8.3f}"
              f"{'  oui' if spec else '  NON':>12}")
    print(f"\n{n_spec}/{len(sel)} cellules restent significatives contre une "
          f"instruction benigne appariee.")
    print("\nLecture : 'retenu %' est la part de l'effet qui subsiste quand le "
          "controle\nest lui-meme une instruction. Proche de 100 % = specifique "
          "a l'injection.\nProche de 0 % = la cellule detectait la forme "
          "imperative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
