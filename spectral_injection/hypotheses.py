"""Four routing hypotheses, measured on raw causal attention.

Pre-registered before the first run. Each statistic is one scalar per
(pair, condition, layer, head); the contrast that counts is
``injected - control`` (same inserted token count, same total prompt length),
never ``injected - benign``, which is length-confounded by construction.

    H1  entropy_post    Shannon entropy of the attention rows of the tokens
                        that come *after* the inserted span, normalised by
                        log(number of keys available to that row).
                        Predicted: DOWN under injection.
    H2  distance_post   mean attention distance of those same rows,
                        normalised by the row's maximum possible distance.
                        Predicted: UP under injection.
    H3a recip_literal   sum over span pairs of min(A_ij, A_ji).
                        STRUCTURALLY ZERO on a causal decoder -- kept as a
                        self-check, not as a hypothesis.
    H3b recip_rank      rank reciprocity inside the span: i puts a large
                        share of its own mass on j, AND i is a large share
                        of the mass j receives.
                        Predicted: UP under injection.
    H4  cut_pre_post    mean mass that each post-span token sends to the
                        pre-span tokens -- does the injection wall off the
                        two halves of the host?
                        Predicted: DOWN under injection.

Every row is normalised by what that row could possibly reach. Without it,
all four statistics measure token position: row i has only i+1 non-zero
entries, so raw entropy and raw distance both grow with position, and an
insertion shifts every later token to the right.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .prompts import ROLE_INJECTION

NAMES = ["entropy_post", "distance_post", "recip_literal", "recip_rank",
         "cut_pre_post"]
EPS = 1e-12


def _span(roles: np.ndarray) -> Tuple[int, int]:
    """First and last index of the inserted span (inclusive)."""
    idx = np.flatnonzero(roles == ROLE_INJECTION)
    if idx.size == 0:
        raise ValueError("no inserted span in roles")
    lo, hi = int(idx[0]), int(idx[-1])
    if idx.size != hi - lo + 1:
        raise ValueError("inserted span is not contiguous")
    return lo, hi


def _row_entropy(A: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Normalised Shannon entropy of selected rows: [H] -> mean over rows.

    A is [H, T, T], causal. Row i has i+1 valid keys, so the entropy is
    divided by log(i+1): a value of 1 means uniform over everything the row
    can see, 0 means all the mass on one token.
    """
    P = A[:, rows, :]                                   # [H, R, T]
    ent = -(P * np.log(np.clip(P, EPS, None))).sum(-1)  # [H, R]
    denom = np.log(np.maximum(rows + 1, 2)).astype(np.float64)
    return (ent / denom[None, :]).mean(-1)


def _row_distance(A: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean normalised look-back distance of selected rows: [H]."""
    T = A.shape[-1]
    j = np.arange(T)
    dist = rows[:, None] - j[None, :]                   # [R, T], >=0 causally
    dist = np.clip(dist, 0, None).astype(np.float64)
    dist /= np.maximum(rows, 1)[:, None]                # row max distance = i
    d = (A[:, rows, :] * dist[None, :, :]).sum(-1)      # [H, R]
    return d.mean(-1)


def _recip_literal(A: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """sum_{i<j in span} min(A_ij, A_ji), normalised by span pairs: [H].

    On a causal decoder this is identically zero. Computed anyway: if it ever
    comes back non-zero, the attention tensor is not what we think it is.
    """
    S = A[:, lo:hi + 1, lo:hi + 1]
    n = S.shape[-1]
    if n < 2:
        return np.zeros(A.shape[0])
    M = np.minimum(S, np.transpose(S, (0, 2, 1)))
    iu = np.triu_indices(n, 1)
    return M[:, iu[0], iu[1]].sum(-1) / (n * (n - 1) / 2)


def _recip_rank(A: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Rank reciprocity inside the span: [H].

    The literal notion is unavailable causally, so: R_ij is the share of row
    i's own in-span mass that lands on j, C_ij is the share of the in-span
    mass arriving at j that comes from i. min(R, C) is large only when the
    preference runs both ways in the only sense a causal graph allows.
    """
    S = A[:, lo:hi + 1, lo:hi + 1].astype(np.float64)
    n = S.shape[-1]
    if n < 2:
        return np.zeros(A.shape[0])
    R = S / np.clip(S.sum(-1, keepdims=True), EPS, None)   # row-stochastic
    C = S / np.clip(S.sum(-2, keepdims=True), EPS, None)   # column-stochastic
    M = np.minimum(R, C)
    il = np.tril_indices(n, -1)                            # causal pairs only
    return M[:, il[0], il[1]].mean(-1)


def _cut_pre_post(A: np.ndarray, lo: int, hi: int, keep: np.ndarray
                  ) -> np.ndarray:
    """Mean mass each post-span token sends to pre-span tokens: [H].

    Pre -> post is causally impossible, so the cut is one-directional. The
    mean is per post-span token, not a raw sum, so it does not scale with the
    size of either block.
    """
    pre = np.flatnonzero(keep & (np.arange(A.shape[-1]) < lo))
    post = np.flatnonzero(keep & (np.arange(A.shape[-1]) > hi))
    if pre.size == 0 or post.size == 0:
        return np.full(A.shape[0], np.nan)
    return A[:, post, :][:, :, pre].sum(-1).mean(-1)


def head_values(attn_layer: np.ndarray, roles: np.ndarray) -> np.ndarray:
    """All five statistics for one layer: [heads, 5]."""
    A = np.asarray(attn_layer, dtype=np.float64)
    if A.ndim != 3:
        raise ValueError(f"expected [H, T, T], got {A.shape}")
    lo, hi = _span(roles)
    T = A.shape[-1]
    # Rows that come after the span and are not themselves inserted.
    not_inserted = roles != ROLE_INJECTION
    post_rows = np.flatnonzero(not_inserted & (np.arange(T) > hi))
    if post_rows.size == 0:
        raise ValueError("no non-inserted token after the span")

    return np.stack([
        _row_entropy(A, post_rows),
        _row_distance(A, post_rows),
        _recip_literal(A, lo, hi),
        _recip_rank(A, lo, hi),
        _cut_pre_post(A, lo, hi, not_inserted),
    ], axis=-1)


def stack(attentions: Sequence[np.ndarray], roles: np.ndarray,
          layers: Sequence[int]) -> np.ndarray:
    """[len(layers), heads, 5] for one prompt."""
    return np.stack([head_values(attentions[l], roles) for l in layers])
