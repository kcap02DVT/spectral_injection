"""Tests apparies PAR CELLULE (couche, tete), sans aucune moyenne entre tetes.

Chaque cellule est traitee comme une << metrique >> de la famille et passee a
``opi.stats.paired_differences`` / ``sign_flip_test``, code deja valide :
un tirage inverse le signe de PAIRES entieres, donc toutes les cellules
basculent ensemble et la dependance entre tetes est preservee. La
correction max-statistic rendue par ``sign_flip_test`` est le controle FWER
(severe) ; Benjamini-Hochberg sur les p brutes est le compte principal.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from ..opi.stats import paired_differences, sign_flip_test

CONDS = ("ADV", "CTL")


def cell_array(long_df: pd.DataFrame, metric: str, n_layers: int, n_heads: int
               ) -> Tuple[np.ndarray, List[str], List]:
    """Tableau long -> ``[n_paires, 2, n_cellules]`` pour une metrique.
    Cellules ordonnees (couche, tete). Rend aussi les noms et les pair_id."""
    sub = long_df[long_df.metrique == metric]
    pairs = sorted(sub.exemple_id.unique())
    names = [f"L{l:02d}H{h:02d}" for l in range(n_layers) for h in range(n_heads)]
    A = np.full((len(pairs), 2, n_layers * n_heads), np.nan)
    piv = sub.pivot_table(index=["exemple_id", "condition"],
                          columns=["couche", "tete"], values="valeur")
    for i, p in enumerate(pairs):
        for c_i, c in enumerate(CONDS):
            if (p, c) in piv.index:
                row = piv.loc[(p, c)]
                for j, (l, h) in enumerate([(l, h) for l in range(n_layers)
                                            for h in range(n_heads)]):
                    if (l, h) in row.index:
                        A[i, c_i, j] = row[(l, h)]
    return A, names, pairs


def bh_fdr(p: np.ndarray) -> np.ndarray:
    """p ajustees de Benjamini-Hochberg (monotones)."""
    p = np.asarray(p, float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def test_cells(A: np.ndarray, names: Sequence[str], n_perm: int, seed: int,
               fdr_q: float) -> pd.DataFrame:
    """Une ligne par cellule : delta median, win rate, p brute, p BH,
    p max-stat (FWER), drapeaux et direction. Aucune moyenne entre tetes."""
    pd_ = paired_differences(A, list(names), list(CONDS), "ADV", "CTL")
    res = sign_flip_test(pd_, n_perm=n_perm, seed=seed)
    D = pd_.diff
    p_raw = np.array([r["p_raw"] for r in res])
    p_bh = bh_fdr(p_raw)
    rows = []
    for j, (nm, r) in enumerate(zip(names, res)):
        d = D[:, j]
        rows.append({"cellule": nm, "couche": int(nm[1:3]), "tete": int(nm[4:6]),
                     "n": int(np.isfinite(d).sum()),
                     "delta_median": float(np.nanmedian(d)),
                     "delta_mean": float(np.nanmean(d)),
                     "win_rate": float((d > 0).mean()),
                     "p_raw": float(r["p_raw"]), "p_bh": float(p_bh[j]),
                     "p_fwer": float(r["p_fwer"]),
                     "sig_bh": bool(p_bh[j] < fdr_q),
                     "sig_fwer": bool(r["p_fwer"] < fdr_q),
                     "direction": ("+" if np.nanmedian(d) > 0 else
                                   "-" if np.nanmedian(d) < 0 else "0")})
    out = pd.DataFrame(rows)
    out.attrs["n_pairs"] = int(D.shape[0])
    out.attrs["dropped"] = list(pd_.dropped)
    return out


def count_significant(cells: pd.DataFrame) -> Dict:
    return {"n_cells": int(len(cells)),
            "sig_bh": int(cells.sig_bh.sum()),
            "sig_bh_plus": int((cells.sig_bh & (cells.direction == "+")).sum()),
            "sig_bh_minus": int((cells.sig_bh & (cells.direction == "-")).sum()),
            "sig_fwer": int(cells.sig_fwer.sum()),
            "sig_fwer_plus": int((cells.sig_fwer & (cells.direction == "+")).sum()),
            "sig_fwer_minus": int((cells.sig_fwer & (cells.direction == "-")).sum())}


def secondary_head_summary(cells: pd.DataFrame) -> pd.DataFrame:
    """SECONDAIRE, explicitement : par couche, mediane des deltas medians
    entre tetes ET la distribution complete (min, q25, q75, max, nb de tetes
    de chaque signe). Ne jamais lire la colonne 'mediane_entre_tetes' seule."""
    g = cells.groupby("couche")["delta_median"]
    out = pd.DataFrame({
        "mediane_entre_tetes_SECONDAIRE": g.median(),
        "min": g.min(), "q25": g.quantile(.25), "q75": g.quantile(.75), "max": g.max(),
        "tetes_positives": cells.groupby("couche")["direction"].apply(lambda s: int((s == "+").sum())),
        "tetes_negatives": cells.groupby("couche")["direction"].apply(lambda s: int((s == "-").sum())),
        "tetes_sig_bh": cells.groupby("couche")["sig_bh"].sum().astype(int),
    })
    return out
