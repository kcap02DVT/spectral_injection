"""Per-head topology measures, from Hamilton's *Graph Representation Learning* ch. 2.

Two families, kept apart on purpose.

**Labelled** measures use the known injection span. They describe the
*mechanism* and are compared against a length-matched control insertion; they
can never be a detector, since a real prompt comes with no labels.

    dens_intra_inj / dens_intra_host / dens_cross
                      mean off-diagonal weight inside each family and between
                      them -- the tutor's "two clusters, little attention
                      between them" (ch. 2.3.2, graph cuts)
    separation        sqrt(d_inj * d_host) / d_cross, >1 when the families
                      hold together more than they talk to each other
    conductance_true  cut weight at the true boundary over the smaller volume
    fiedler_ari       agreement between the sign cut of the Fiedler vector
                      (ch. 2.3.2, eq. 2.49) and the true boundary
    fiedler_acc       same, as accuracy over the better orientation
    centrality_ratio  mean eigenvector centrality (ch. 2.1.1, eq. 2.3) of
                      injected tokens over that of host tokens
    clustering_ratio  same for the local clustering coefficient (eq. 2.5)
    overlap_ratio     neighbourhood overlap (ch. 2.2) injected-host over
                      host-host: low means the span keeps its own neighbours
    ppr_enrichment    mass a personalized PageRank started at the last token
                      (ch. 2.2, eq. 2.23) puts on the span, over the mass a
                      uniform distribution would give it

**Blind** measures never look at the labels, so they are detector candidates.
They are computed in all three conditions (benign, injected, control).

    lambda2           algebraic connectivity of the symmetric normalized
                      Laplacian
    eigengap          lambda3 - lambda2: large when two clusters is genuinely
                      the right split (ch. 2.3.3)
    cut_conductance   conductance of the cut discovered by the Fiedler sign
    ppr_entropy       normalized entropy of the personalized PageRank from the
                      last token: low when its mass concentrates somewhere
    ppr_window        largest PageRank mass on a contiguous window of tokens
    centrality_peak   max eigenvector centrality over its median
    clustering_mean   mean local clustering coefficient
    clustering_window largest mean clustering over a contiguous window

Everything is batched over heads: one eigendecomposition per head and a few
matrix products, which is what keeps a 32-head layer affordable.

The clustering coefficient and the neighbourhood overlap need a sparse graph:
attention is dense, where every pair has some tiny weight and a binary degree
would be the whole sequence. Both are therefore computed on the graph kept by
``EDGE_QUANTILE`` -- the strongest 10 % of off-diagonal weights per head.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .prompts import ROLE_INJECTION

LABELLED = ["dens_intra_inj", "dens_intra_host", "dens_cross", "separation",
            "conductance_true", "fiedler_ari", "fiedler_acc",
            "centrality_ratio", "clustering_ratio", "overlap_ratio",
            "ppr_enrichment"]
BLIND = ["lambda2", "eigengap", "cut_conductance", "ppr_entropy", "ppr_window",
         "centrality_peak", "clustering_mean", "clustering_window"]

EDGE_QUANTILE = 0.90     # sparsified graph for clustering / overlap
PPR_RESTART = 0.15       # probability of teleporting back to the start node
WINDOW = 20              # contiguous window, in tokens


def _symmetric(attn_layer: np.ndarray) -> np.ndarray:
    """(A + A^T) / 2 per head, zero diagonal: [H, T, T]."""
    A = np.asarray(attn_layer, dtype=np.float64)
    W = 0.5 * (A + np.transpose(A, (0, 2, 1)))
    idx = np.arange(W.shape[-1])
    W[:, idx, idx] = 0.0
    return W


def _laplacian_spectrum(W: np.ndarray):
    """Eigenvalues of L_sym and the Fiedler vector, per head."""
    deg = W.sum(-1)
    inv = np.where(deg > 0, 1.0 / np.sqrt(np.maximum(deg, 1e-12)), 0.0)
    L = -W * inv[:, :, None] * inv[:, None, :]
    idx = np.arange(W.shape[-1])
    L[:, idx, idx] += 1.0
    vals, vecs = np.linalg.eigh(L)
    return vals, vecs[:, :, 1], deg


def _conductance(W: np.ndarray, deg: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """cut(S, S_bar) / min(vol(S), vol(S_bar)) per head; mask is [H, T] or [T]."""
    m = np.broadcast_to(mask, (W.shape[0], W.shape[-1])).astype(float)
    cut = np.einsum("hi,hij,hj->h", m, W, 1.0 - m)
    vol_a = (deg * m).sum(-1)
    vol_b = (deg * (1.0 - m)).sum(-1)
    small = np.minimum(vol_a, vol_b)
    return np.where(small > 0, cut / np.maximum(small, 1e-12), np.nan)


def _eigen_centrality(W: np.ndarray, iters: int = 60) -> np.ndarray:
    """Leading eigenvector of W per head, by power iteration (eq. 2.4).

    Matrix products, not einsum: these go through BLAS, which is what makes a
    whole campaign affordable (einsum here was an order of magnitude slower).
    """
    v = np.ones((W.shape[0], W.shape[-1], 1)) / W.shape[-1]
    for _ in range(iters):
        v = W @ v
        norm = np.linalg.norm(v, axis=-2, keepdims=True)
        v = np.where(norm > 0, v / np.maximum(norm, 1e-12), v)
    return np.abs(v[:, :, 0])


def _sparse_graph(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-decile weighted graph and its binary version, per head."""
    T = W.shape[-1]
    off = W.reshape(W.shape[0], -1)
    thr = np.quantile(off, EDGE_QUANTILE, axis=-1)[:, None, None]
    keep = W >= thr
    keep = keep | np.transpose(keep, (0, 2, 1))        # keep it symmetric
    idx = np.arange(T)
    keep[:, idx, idx] = False
    return W * keep, keep.astype(float)


def _clustering(Ws: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Onnela weighted local clustering coefficient per node (eq. 2.5)."""
    mx = Ws.max(axis=(-2, -1), keepdims=True)
    Wn = np.cbrt(np.where(mx > 0, Ws / np.maximum(mx, 1e-12), 0.0))
    tri = ((Wn @ Wn) * Wn).sum(-1)          # diag(Wn^3), via BLAS
    k = B.sum(-1)
    denom = k * (k - 1.0)
    return np.where(denom > 0, tri / np.maximum(denom, 1e-12), 0.0)


def _overlap(W: np.ndarray) -> np.ndarray:
    """Cosine neighbourhood overlap between nodes, per head (ch. 2.2).

    On the weighted graph rather than the sparsified one: with only the top
    decile of edges kept, two tokens from different families share no
    neighbour at all and every overlap collapses to zero.
    """
    norm = np.linalg.norm(W, axis=-1, keepdims=True)
    Wn = W / np.maximum(norm, 1e-12)
    return Wn @ np.transpose(Wn, (0, 2, 1))


def _ppr(W: np.ndarray, start: int, iters: int = 60) -> np.ndarray:
    """Personalized PageRank from one node, per head (eq. 2.23).

    60 iterations leave a 0.85^60 ~ 5e-5 residual, below everything measured
    on top of it.
    """
    col = W.sum(-2)
    P = W / np.maximum(col, 1e-12)[:, None, :]
    e = np.zeros((1, W.shape[-1], 1))
    e[0, start, 0] = 1.0
    q = np.broadcast_to(e, (W.shape[0], W.shape[-1], 1)).copy()
    for _ in range(iters):
        q = (1.0 - PPR_RESTART) * (P @ q) + PPR_RESTART * e
    q = q[:, :, 0]
    return q / np.maximum(q.sum(-1, keepdims=True), 1e-12)


def _window_max(x: np.ndarray, width: int, mean: bool) -> np.ndarray:
    """Largest sum (or mean) over a contiguous window of ``width`` tokens."""
    w = min(width, x.shape[-1])
    c = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x], axis=-1), -1)
    sums = c[:, w:] - c[:, :-w]
    return (sums / w if mean else sums).max(-1)


def head_measures(attn_layer: np.ndarray, roles: np.ndarray,
                  last_index: Optional[int] = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """``(labelled [H, 11], blind [H, 8])`` for one layer.

    ``roles`` marks the inserted span (the attack, or the control text).
    Without one, the labelled half is all-NaN -- the benign condition.
    """
    W = _symmetric(attn_layer)
    H, T, _ = W.shape
    start = T - 1 if last_index is None else int(last_index)
    vals, fiedler, deg = _laplacian_spectrum(W)
    cent = _eigen_centrality(W)
    Ws, B = _sparse_graph(W)
    clus = _clustering(Ws, B)
    ppr = _ppr(W, start)

    blind = np.stack([
        vals[:, 1],
        vals[:, 2] - vals[:, 1],
        _conductance(W, deg, fiedler >= 0),
        -(ppr * np.log(np.maximum(ppr, 1e-12))).sum(-1) / np.log(T),
        _window_max(ppr, WINDOW, mean=False),
        np.where(np.median(cent, -1) > 0,
                 cent.max(-1) / np.maximum(np.median(cent, -1), 1e-12), np.nan),
        clus.mean(-1),
        _window_max(clus, WINDOW, mean=True),
    ], axis=-1)

    ins = np.flatnonzero(roles == ROLE_INJECTION)
    host = np.flatnonzero(roles != ROLE_INJECTION)
    if ins.size < 2 or host.size < 2:
        return np.full((H, len(LABELLED)), np.nan), blind

    def block(M, a, b, diagonal):
        sub = M[:, a][:, :, b]
        if diagonal:                       # drop the diagonal of a square block
            n = len(a)
            mask = ~np.eye(n, dtype=bool)
            return sub[:, mask].mean(-1)
        return sub.mean(axis=(-2, -1))

    d_ii = block(W, ins, ins, True)
    d_hh = block(W, host, host, True)
    d_ih = block(W, ins, host, False)
    mask = np.zeros(T, dtype=bool)
    mask[ins] = True
    cut = fiedler >= 0
    agree = np.maximum((cut[:, ins].mean(-1) + (~cut)[:, host].mean(-1)) / 2,
                       ((~cut)[:, ins].mean(-1) + cut[:, host].mean(-1)) / 2)
    ov = _overlap(W)

    labelled = np.stack([
        d_ii, d_hh, d_ih,
        np.where(d_ih > 0, np.sqrt(np.maximum(d_ii * d_hh, 0.0)) / d_ih, np.nan),
        _conductance(W, deg, mask),
        np.array([_ari(c, mask) for c in cut]),
        agree,
        np.where(cent[:, host].mean(-1) > 0,
                 cent[:, ins].mean(-1) / cent[:, host].mean(-1), np.nan),
        np.where(clus[:, host].mean(-1) > 0,
                 clus[:, ins].mean(-1) / np.maximum(clus[:, host].mean(-1),
                                                    1e-12), np.nan),
        np.where(block(ov, host, host, True) > 0,
                 block(ov, ins, host, False)
                 / np.maximum(block(ov, host, host, True), 1e-12), np.nan),
        ppr[:, ins].sum(-1) / (len(ins) / T),
    ], axis=-1)
    return labelled, blind


def _ari(cut: np.ndarray, truth: np.ndarray) -> float:
    """Adjusted Rand index between two binary partitions."""
    from scipy.special import comb

    n = len(truth)
    a = int((cut & truth).sum())
    b = int((cut & ~truth).sum())
    c = int((~cut & truth).sum())
    d = int((~cut & ~truth).sum())
    index = comb(a, 2) + comb(b, 2) + comb(c, 2) + comb(d, 2)
    s1 = comb(a + b, 2) + comb(c + d, 2)
    s2 = comb(a + c, 2) + comb(b + d, 2)
    exp = s1 * s2 / comb(n, 2)
    mx = (s1 + s2) / 2
    return float((index - exp) / (mx - exp)) if mx > exp else 0.0


def stack(attentions: Sequence[np.ndarray], roles: np.ndarray,
          layers: Sequence[int], last_index: Optional[int] = None
          ) -> Tuple[np.ndarray, np.ndarray]:
    """``(labelled [L, H, 11], blind [L, H, 8])`` over the requested layers."""
    lab, blind = [], []
    for layer in layers:
        a, b = head_measures(attentions[layer], roles, last_index)
        lab.append(a)
        blind.append(b)
    return np.stack(lab), np.stack(blind)
