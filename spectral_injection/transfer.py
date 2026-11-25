"""H12-bis : transfert entre sources d'attaques, cellules choisies une seule fois.

Les cellules sont selectionnees sur le split train de BIPIA et sur lui seul,
puis evaluees telles quelles sur le split test de chaque autre source. Rien
n'est reselectionne, donc ce qui varie d'une colonne a l'autre est la seule
distribution d'attaques.

Le controle est le meme partout (prose du meme email), pour que le transfert
porte sur l'attaque et pas sur le controle.

Trois metriques, dont deux jamais calculees jusqu'ici :

    AUROC        probabilite qu'un prompt injecte soit mieux classe qu'un
                 controle tire au hasard. 0,5 = hasard.
    AUPRC        aire sous precision-rappel. Sa valeur de hasard est la
                 prevalence, ici 0,5 puisque les classes sont equilibrees --
                 a lire comme un gain au-dessus de 0,5, pas comme un absolu.
    rappel@FPR   part des injections attrapees quand on fixe le taux de faux
                 positifs a 1 % puis 5 %. C'est la metrique d'un filtre en
                 production, ou l'on ne tolere presque aucune fausse alarme.

Le score est oriente par le signe observe sur train : une cellule dont
l'effet est negatif est lue a l'envers, sans quoi son AUROC passerait sous
0,5 par convention d'ecriture et non par absence de signal.
"""

import os
import sys

import numpy as np

from .hypotheses import NAMES
from .hypotheses_report import TESTED
from .specificity import family_stats


def load(path):
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    return d["values"]


def _roc_pr(y, s):
    """AUROC, AUPRC et rappel aux FPR fixes, pour un score oriente."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return (np.nan,) * 4
    tpr, fpr = tp / P, fp / N
    auroc = float(np.trapezoid(np.r_[0, tpr], np.r_[0, fpr]))
    prec = tp / np.maximum(tp + fp, 1)
    auprc = float(np.sum(np.diff(np.r_[0, tpr]) * prec))
    rec = []
    for target in (0.01, 0.05):
        ok = fpr <= target
        rec.append(float(tpr[ok].max()) if ok.any() else 0.0)
    return auroc, auprc, rec[0], rec[1]


def main(argv=None) -> int:
    a = argv or sys.argv[1:]
    sel_dir, top = a[0], 10
    targets = [tuple(x.split("=", 1)) for x in a[1:]]

    Vs = load(sel_dir)
    Ns, _, L, H, M = Vs.shape
    Ds = (Vs[:, 0] - Vs[:, 1]).reshape(Ns, -1)
    midx = np.tile(np.arange(M), L * H).reshape(L, H, M).ravel()
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)

    fam = np.isfinite(Ds).all(0) & np.isin(midx, [NAMES.index(m) for m in TESTED])
    st = family_stats(Ds[:, fam], n_perm=5000, seed=0)
    cells = np.flatnonzero(fam)
    keep = st["p_fwer"] < 0.05
    sel, dz = cells[keep], st["dz"][keep]
    o = np.argsort(-np.abs(dz))[:top]
    sel, dz = sel[o], dz[o]
    sign = np.sign(dz)
    print(f"selection sur {sel_dir} ({Ns} paires) : {sel.size} cellules\n")

    print(f"{'source':<16}{'n':>5}{'AUROC':>9}{'AUPRC':>9}"
          f"{'rappel@1%FPR':>15}{'rappel@5%FPR':>15}")
    print("-" * 69)
    for name, path in targets:
        V = load(path)
        n = V.shape[0]
        fi = V[:, 0].reshape(n, -1)
        fc = V[:, 1].reshape(n, -1)
        y = np.r_[np.zeros(n), np.ones(n)]
        rows = []
        for k, c in enumerate(sel):
            s = np.r_[fc[:, c], fi[:, c]] * sign[k]      # oriente : haut = injecte
            if not np.isfinite(s).all():
                continue
            rows.append(_roc_pr(y, s))
        r = np.array(rows)
        print(f"{name:<16}{n:>5}{np.nanmean(r[:, 0]):>9.3f}{np.nanmean(r[:, 1]):>9.3f}"
              f"{np.nanmean(r[:, 2]):>15.3f}{np.nanmean(r[:, 3]):>15.3f}")
        best = int(np.nanargmax(r[:, 0]))
        c = sel[best]
        print(f"{'':16}{'':5}  meilleure cellule : {NAMES[midx[c]]} "
              f"c{lidx[c]} t{hidx[c]}  AUROC={r[best,0]:.3f}  "
              f"rappel@5%={r[best,3]:.3f}")
    print("\nMoyennes sur les cellules selectionnees. L'AUPRC a pour valeur de "
          "hasard la prevalence (0,5 ici, classes equilibrees).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
