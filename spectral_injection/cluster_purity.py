"""Does K-way spectral clustering isolate the injected span?

The question ARI cannot answer. With 11 injected tokens out of 250, a small
pure cluster can exist while the Adjusted Rand Index of the whole partition
sits at zero, because ARI scores the entire partition and the other 239
tokens dominate it.

So: run generalised spectral clustering (Hamilton 2.3.3 -- k-means on the K
lowest non-trivial eigenvectors of L_sym), then ask of the resulting
clusters, for each K:

    purity   of the cluster that best matches the span: what share of it is
             injected tokens  (= precision)
    recall   what share of the injected tokens that cluster holds
    f1       the two combined -- purity alone is gamed by a 1-node cluster
    majority 1 if some cluster is more than half injected tokens

All four need the labels, so they describe the mechanism. One blind measure
rides along, computed without ever looking at roles:

    contig   the most contiguous cluster of size >= 3: longest run of
             consecutive token positions over its size. An inserted span is
             contiguous by construction, so a blind detector could pick the
             most contiguous cluster and call it the span.
    hit      1 if that blindly-chosen cluster is in fact the injected one
             (f1 > 0.5) -- the bridge from "clustering sees it" to "a
             detector can find it without labels"

Everything is computed in BOTH conditions. If a cluster isolates the span
just as well under the length-matched control, then clustering localises
*insertions*, not injections -- which would still be useful, as it would
hand the blind pipeline the span it currently lacks.

K is swept over a fixed, pre-registered set; every value is reported, none
is selected.
"""

from typing import Dict, List, Sequence

import numpy as np

from .graphs import build_graph, laplacian, spectrum
from .metrics import _kmeans
from .prompts import ROLE_INJECTION

K_VALUES = (2, 3, 5, 8, 12)
NAMES = ["purity", "recall", "f1", "majority", "contig", "hit"]


def _best_cluster(labels: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """Cluster with the highest F1 against the injected set."""
    best = {"purity": 0.0, "recall": 0.0, "f1": 0.0}
    n_inj = int(truth.sum())
    if n_inj == 0:
        return {**best, "majority": 0.0}
    maj = 0.0
    for c in np.unique(labels):
        m = labels == c
        inter = float((m & truth).sum())
        prec = inter / max(m.sum(), 1)
        rec = inter / n_inj
        if prec > 0.5:
            maj = 1.0
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        if f1 > best["f1"]:
            best = {"purity": prec, "recall": rec, "f1": f1}
    return {**best, "majority": maj}


def _contiguity(labels: np.ndarray, positions: np.ndarray) -> tuple:
    """Most contiguous cluster of size >= 3: (its contiguity, its mask).

    Uses only node order, never the roles.
    """
    best_score, best_mask = 0.0, None
    for c in np.unique(labels):
        m = labels == c
        if m.sum() < 3:
            continue
        p = np.sort(positions[m])
        runs, cur = 1, 1
        for a, b in zip(p, p[1:]):
            cur = cur + 1 if b == a + 1 else 1
            runs = max(runs, cur)
        score = runs / m.sum()
        if score > best_score:
            best_score, best_mask = score, m
    return best_score, best_mask


def layer_values(W: np.ndarray, roles: np.ndarray, keep: np.ndarray,
                 seed: int = 0) -> np.ndarray:
    """[len(K_VALUES), 6] for one graph."""
    truth = roles == ROLE_INJECTION
    positions = np.flatnonzero(keep)          # original token indices kept
    L = laplacian(W, "sym", True)
    _, vecs = spectrum(L, "sym")
    out = np.full((len(K_VALUES), len(NAMES)), np.nan)
    for i, k in enumerate(K_VALUES):
        emb = vecs[:, 1:k]                    # drop the trivial eigenvector
        if emb.shape[1] == 0:
            emb = vecs[:, 1:2]
        labels = _kmeans(emb, k=k, seed=seed)
        b = _best_cluster(labels, truth)
        cscore, cmask = _contiguity(labels, positions)
        hit = 0.0
        if cmask is not None:
            inter = float((cmask & truth).sum())
            prec = inter / max(cmask.sum(), 1)
            rec = inter / max(truth.sum(), 1)
            f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
            hit = 1.0 if f1 > 0.5 else 0.0
        out[i] = [b["purity"], b["recall"], b["f1"], b["majority"],
                  cscore, hit]
    return out


def stack(attentions: Sequence[np.ndarray], roles: np.ndarray,
          layers: Sequence[int], config) -> np.ndarray:
    """[len(layers), len(K_VALUES), 6] for one prompt."""
    out = []
    for l in layers:
        g = build_graph(attentions[l], roles,
                        symmetrization=config.symmetrization,
                        head_aggregation=config.head_aggregation,
                        drop_template=config.drop_template,
                        drop_sink=config.drop_sink)
        out.append(layer_values(g["W"], g["roles"], g["keep"]))
    return np.stack(out)
