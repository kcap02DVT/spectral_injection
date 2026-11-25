"""Analyse the H1-H4 run: paired contrast injected - control.

The family for the family-wise correction is the UNION of all hypotheses x
layers x heads, not each hypothesis separately: four hypotheses tested at
once is four times the opportunity to find something, and the correction has
to know that. ``recip_literal`` is excluded from the family -- it is
identically zero by construction and carries no test.
"""

import os, sys
from typing import Dict, List

import numpy as np

from .hypotheses import NAMES
from .specificity import bh, family_stats

# Direction pre-registered before the run. +1 means "injected > control".
PREDICTED = {"entropy_post": -1, "distance_post": +1, "recip_literal": 0,
             "recip_rank": +1, "cut_pre_post": -1}
TESTED = ["entropy_post", "distance_post", "recip_rank", "cut_pre_post"]


def auroc(labels, scores) -> float:
    lab = np.asarray(labels); sc = np.asarray(scores)
    order = np.argsort(sc, kind="mergesort")
    ranks = np.empty(len(sc), float)
    sorted_sc = sc[order]
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sorted_sc[j + 1] == sorted_sc[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    n1 = int((lab == 1).sum()); n0 = len(lab) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None) -> int:
    path = (argv or sys.argv[1:])[0]
    d = np.load(os.path.join(path, "values.npz"), allow_pickle=True)
    V = d["values"]                                  # [N, 2, L, H, 5]
    layers = list(d["layers"])
    N, _, L, H, M = V.shape
    print(f"{N} paires · {L} couches · {H} tetes\n")

    inj, ctl = V[:, 0], V[:, 1]                      # [N, L, H, 5]
    D = (inj - ctl).reshape(N, -1)                   # [N, L*H*5]
    finite = np.isfinite(D).all(0)
    midx = np.tile(np.arange(M), L * H).reshape(L, H, M).ravel()
    lidx = np.repeat(np.arange(L), H * M)
    hidx = np.tile(np.repeat(np.arange(H), M), L)

    # Family = the four real hypotheses, all layers, all heads, at once.
    fam = finite & np.isin(midx, [NAMES.index(m) for m in TESTED])
    st = family_stats(D[:, fam], n_perm=5000, seed=0)
    full = {k: np.full(D.shape[1], np.nan) for k in st}
    for k, v in st.items():
        full[k][fam] = v

    # AUROC injected vs control, per cell.
    au = np.full(D.shape[1], np.nan)
    flat_i = inj.reshape(N, -1); flat_c = ctl.reshape(N, -1)
    lab = np.r_[np.zeros(N), np.ones(N)]
    for c in np.flatnonzero(fam):
        au[c] = auroc(lab, np.r_[flat_c[:, c], flat_i[:, c]])

    print(f"{'hypothese':<16}{'signe':>6}{'meilleur dz':>13}{'couche':>8}"
          f"{'tete':>6}{'AUROC':>8}{'q<.05':>7}{'FWER<.05':>10}{'% bon signe':>13}")
    print("-" * 90)
    rows = []
    for m in NAMES:
        sel = np.flatnonzero((midx == NAMES.index(m)) & finite)
        if m == "recip_literal":
            print(f"{m:<16}{'--':>6}{'degenere':>13}  max|val| = "
                  f"{np.abs(V[:, :, :, :, NAMES.index(m)]).max():.2e}  "
                  f"(zero structurel, exclu de la famille)")
            continue
        dz = full["dz"][sel]
        want = PREDICTED[m]
        best = sel[np.nanargmax(np.abs(dz))]
        n_q = int(np.nansum(full["q_bh"][sel] < 0.05))
        n_f = int(np.nansum(full["p_fwer"][sel] < 0.05))
        frac = float(np.nanmean(np.sign(full["dz"][sel]) == want))
        print(f"{m:<16}{('+' if want>0 else '-'):>6}{full['dz'][best]:>13.3f}"
              f"{layers[lidx[best]]:>8}{hidx[best]:>6}{au[best]:>8.3f}"
              f"{n_q:>7}{n_f:>10}{frac*100:>12.0f}%")
        rows.append((m, best, n_q, n_f))

    print("\nProfil par couche (dz median sur les tetes, signe oriente par la "
          "prediction) :")
    print(f"{'couche':>7}" + "".join(f"{m[:11]:>13}" for m in TESTED))
    for li in range(L):
        line = f"{layers[li]:>7}"
        for m in TESTED:
            mi = NAMES.index(m)
            sel = np.flatnonzero((lidx == li) & (midx == mi) & finite)
            v = np.nanmedian(full["dz"][sel]) * PREDICTED[m]
            line += f"{v:>13.2f}"
        print(line)

    print("\nCellules survivant au FWER (max-|t|, famille = 4 hypotheses x "
          f"{L} couches x {H} tetes) :")
    surv = np.flatnonzero(np.nan_to_num(full["p_fwer"], nan=1.0) < 0.05)
    if surv.size == 0:
        print("  aucune.")
    else:
        order = surv[np.argsort(-np.abs(full["dz"][surv]))]
        for c in order[:15]:
            print(f"  {NAMES[midx[c]]:<15} couche {layers[lidx[c]]:>2} "
                  f"tete {hidx[c]:>2}  dz={full['dz'][c]:+.3f}  "
                  f"AUROC={au[c]:.3f}  p_fwer={full['p_fwer'][c]:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
