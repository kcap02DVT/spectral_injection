"""The control ladder: one set of cells, three controls, one table.

Cells are selected ONCE, on the email_text train split, and then evaluated
unchanged against every control's test split. No reselection per rung, so
the only thing that varies down a column is what the attack is being
compared against.

    usage: python -m spectral_injection.hypotheses_ladder SEL_TRAIN \
               TEST_PROSE TEST_HUMAN_IMP TEST_BENIGN_INSTR [top]
"""

import os, sys
import numpy as np

from .hypotheses import NAMES
from .hypotheses_report import PREDICTED, TESTED, auroc
from .specificity import family_stats, holm

RUNGS = ["prose", "imperatif humain", "instruction modele"]


def load(path):
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    return d["values"], list(d["layers"])


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    sel_dir, rung_dirs, top = a[0], a[1:4], int(a[4]) if len(a) > 4 else 10
    Vs, layers = load(sel_dir)
    Rs = [load(p)[0] for p in rung_dirs]
    Ns, _, L, H, M = Vs.shape
    print(f"selection : {sel_dir} ({Ns} paires)")
    for name, p, V in zip(RUNGS, rung_dirs, Rs):
        print(f"  echelon {name:<20} {V.shape[0]:>3} paires  {p}")
    print()

    dif = lambda V: (V[:, 0] - V[:, 1]).reshape(V.shape[0], -1)
    Ds, Dr = dif(Vs), [dif(V) for V in Rs]
    midx = np.tile(np.arange(M), L * H).reshape(L, H, M).ravel()
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)

    fam = np.isfinite(Ds).all(0)
    for D in Dr:
        fam &= np.isfinite(D).all(0)
    fam &= np.isin(midx, [NAMES.index(m) for m in TESTED])

    st = family_stats(Ds[:, fam], n_perm=5000, seed=0)
    cells = np.flatnonzero(fam)
    keep = st["p_fwer"] < 0.05
    sel, sel_dz = cells[keep], st["dz"][keep]
    o = np.argsort(-np.abs(sel_dz))[:top]
    sel, sel_dz = sel[o], sel_dz[o]
    if sel.size == 0:
        print("aucune cellule selectionnee"); return 1

    stats = [family_stats(D[:, sel], n_perm=5000, seed=1 + i)
             for i, D in enumerate(Dr)]
    holms = [holm(s["p_perm"]) for s in stats]

    def au(V, c):
        n = V.shape[0]
        fi = V[:, 0].reshape(n, -1); fc = V[:, 1].reshape(n, -1)
        x = auroc(np.r_[np.zeros(n), np.ones(n)], np.r_[fc[:, c], fi[:, c]])
        return max(x, 1 - x)

    head = f"{'hypothese':<14}{'c':>3}{'t':>3}"
    for r in RUNGS:
        head += f"{r[:9]+' dz':>14}{'AUROC':>7}"
    print(head); print("-" * len(head))
    for k, c in enumerate(sel):
        line = f"{NAMES[midx[c]]:<14}{layers[lidx[c]]:>3}{hidx[c]:>3}"
        for i in range(3):
            mark = "*" if holms[i][k] < 0.05 and np.sign(
                stats[i]["dz"][k]) == np.sign(sel_dz[k]) else " "
            line += f"{stats[i]['dz'][k]:>+13.2f}{mark}{au(Rs[i], c):>7.3f}"
        print(line)
    print("\n* = significatif (Holm < .05) ET de meme signe que la selection.")

    print("\nPar hypothese, toutes cellules, % de bon signe par echelon :")
    print(f"{'hypothese':<16}" + "".join(f"{r[:16]:>18}" for r in RUNGS))
    for m in TESTED:
        s = np.flatnonzero(fam & (midx == NAMES.index(m)))
        line = f"{m:<16}"
        for D in Dr:
            sm = family_stats(D[:, s], n_perm=1000, seed=9)
            frac = 100 * np.mean(np.sign(sm["dz"]) == PREDICTED[m])
            line += f"{frac:>14.0f} %  "
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
