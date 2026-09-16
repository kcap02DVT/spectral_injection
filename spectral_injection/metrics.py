"""Spectral metrics and cut statistics on the attention graph.

Two groups of numbers come out of this module.

1. **Per-layer spectral metrics** -- family "attention" in spectral-trust
   terms: they are functions of the attention weights alone, so results stay
   valid under attention-only access. The hybrid metrics (energy,
   smoothness_index, spectral_entropy, hfer as computed from residual-stream
   states) are deliberately *not* used here: they need hidden states and must
   not be reported as attention-only.

2. **Cut statistics** -- how strongly the benign tokens and the injected
   tokens hold together, and how weakly they connect to each other. This is
   what a single averaged Fiedler value cannot express: a uniformly
   lukewarm graph and two dense clusters barely touching can share the same
   algebraic connectivity.

Every density is normalised by the number of *possible* pairs. Without that,
the larger group wins mechanically and the whole comparison is an artefact of
the injection being shorter than the host text.
"""

from typing import Dict, List, Optional

import numpy as np

from .graphs import EPS, _require, fiedler, gsp_config, laplacian, spectrum
from .prompts import ROLE_BENIGN, ROLE_INJECTION

# Same names and order as spectral_trust.per_head.PER_HEAD_METRIC_NAMES.
METRIC_NAMES = [
    "fiedler_value",
    "connectivity_ratio",
    "spectral_entropy_norm",
    "hfer",
    "spectral_radius",
]


# --------------------------------------------------------------------------
# Whole-graph (head-aggregated) metrics
# --------------------------------------------------------------------------
def layer_metrics(
    W: np.ndarray,
    normalization: str = "sym",
    remove_self_loops: bool = True,
) -> Dict[str, float]:
    """The five attention-only metrics for one graph.

    Definitions match spectral_trust v0.3.0 so numbers are directly
    comparable with the library's per-head output.
    """
    n = W.shape[0]
    if n < 3:
        return {k: 0.0 for k in METRIC_NAMES}

    torch, _, _, _ = _require()
    from spectral_trust import per_head_metrics

    # The aggregated graph is handed to the library as a single "head": same
    # five definitions, same code path as the per-head export, so the two can
    # be compared without an asterisk. per_head_metrics uses a full
    # torch.linalg.eigvalsh, so there is no truncated-spectrum trap here.
    cfg = gsp_config(normalization=normalization,
                     remove_self_loops=remove_self_loops)
    tensor = torch.from_numpy(np.ascontiguousarray(W, dtype=np.float32))[None]
    values = per_head_metrics(tensor, config=cfg).values[0]
    return {name: float(v) for name, v in zip(METRIC_NAMES, values)}


# --------------------------------------------------------------------------
# Per-head metrics (delegated to spectral-trust)
# --------------------------------------------------------------------------
def per_head_available() -> Optional[str]:
    """None if per-head metrics can run, else a one-line reason why not.

    Worth reporting rather than swallowing: the usual cause is running the
    wrong interpreter -- torch and transformers are often installed
    system-wide while spectral-trust lives only in the project venv, so the
    run proceeds and only the per-head half goes missing.
    """
    import sys
    try:
        import torch                                        # noqa: F401
        from spectral_trust import per_head_metrics         # noqa: F401
    except Exception as exc:                                 # noqa: BLE001
        return f"{type(exc).__name__}: {exc} (interpréteur : {sys.executable})"
    return None


def per_head_layer_metrics(
    attn_layer: np.ndarray,
    config,
    token_span: Optional[tuple] = None,
) -> Optional[np.ndarray]:
    """Per-head metrics for one layer: [heads, 5], or None if unavailable.

    Head averaging is a signal destroyer -- the routing anomaly is usually
    carried by a handful of heads and gets diluted into a near-uniform
    background when heads are summed. For any detection experiment, feed these
    per-head values to the probe and let it learn which heads matter.

    ``token_span`` takes the induced subgraph *before* the Laplacian is built,
    so the diagnostic is local to that span rather than to the whole sequence.
    """
    try:
        import torch
        from spectral_trust import GSPConfig, per_head_metrics
    except Exception:
        return None

    cfg = GSPConfig(
        normalization=config.normalization,
        symmetrization=config.symmetrization,
        head_aggregation=config.head_aggregation,
        remove_self_loops=config.remove_self_loops,
    )
    tensor = torch.from_numpy(np.ascontiguousarray(attn_layer, dtype=np.float32))
    diag = per_head_metrics(tensor, config=cfg, token_span=token_span)
    return np.asarray(diag.values)


# --------------------------------------------------------------------------
# Cut statistics
# --------------------------------------------------------------------------
def _pair_normalised_density(W: np.ndarray, idx_a: np.ndarray,
                             idx_b: np.ndarray, same: bool) -> float:
    """Mean edge weight over all possible pairs between two index sets."""
    if idx_a.size == 0 or idx_b.size == 0:
        return float("nan")
    block = W[np.ix_(idx_a, idx_b)]
    if same:
        if idx_a.size < 2:
            return float("nan")
        total = block.sum() - np.trace(block)
        n_pairs = idx_a.size * (idx_a.size - 1)
    else:
        total = block.sum()
        n_pairs = idx_a.size * idx_b.size
    return float(total / max(n_pairs, 1))


def cut_statistics(W: np.ndarray, roles: np.ndarray) -> Dict[str, float]:
    """Quantify the benign / injection split of a graph.

    Returns NaNs for the cross terms when there is no injection (the benign
    control), which is correct: there is nothing to cut.
    """
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)

    b = np.flatnonzero(roles == ROLE_BENIGN)
    i = np.flatnonzero(roles == ROLE_INJECTION)

    stats = {
        "n_benign": int(b.size),
        "n_injection": int(i.size),
        "density_benign": _pair_normalised_density(A, b, b, True),
        "density_injection": _pair_normalised_density(A, i, i, True),
        "density_cross": _pair_normalised_density(A, b, i, False),
    }

    if i.size == 0 or b.size == 0:
        stats.update(cut_weight=float("nan"), conductance=float("nan"),
                     normalized_cut=float("nan"), modularity=float("nan"),
                     separation=float("nan"))
        return stats

    cut = float(A[np.ix_(b, i)].sum()) * 2.0     # undirected: count both ways
    vol_b = float(A[b].sum())
    vol_i = float(A[i].sum())
    total = float(A.sum())

    stats["cut_weight"] = cut
    stats["conductance"] = cut / max(min(vol_b, vol_i), EPS)
    stats["normalized_cut"] = cut / max(vol_b, EPS) + cut / max(vol_i, EPS)
    stats["modularity"] = _modularity(A, roles == ROLE_INJECTION, total)
    # >1 means the two families hold together more than they talk to each
    # other; this is the bicluster hypothesis in one number.
    denom = np.sqrt(max(stats["density_benign"], EPS)
                    * max(stats["density_injection"], EPS))
    stats["separation"] = float(denom / max(stats["density_cross"], EPS))
    return stats


def _modularity(A: np.ndarray, mask: np.ndarray, total: float) -> float:
    """Newman modularity of a bipartition (weighted, undirected)."""
    if total <= EPS:
        return float("nan")
    deg = A.sum(axis=1)
    two_m = total
    q = 0.0
    for group in (mask, ~mask):
        idx = np.flatnonzero(group)
        if idx.size == 0:
            continue
        e_in = A[np.ix_(idx, idx)].sum()
        d_sum = deg[idx].sum()
        q += e_in / two_m - (d_sum / two_m) ** 2
    return float(q)


# --------------------------------------------------------------------------
# Discovered partition: the label-free detector
# --------------------------------------------------------------------------
def fiedler_partition(W: np.ndarray, normalization: str = "sym",
                      remove_self_loops: bool = True) -> np.ndarray:
    """Bipartition from the sign of the Fiedler vector.

    This is what a detector can actually compute: it never sees where the
    injection is, it discovers a cut and measures how good that cut is.
    """
    L = laplacian(W, normalization, remove_self_loops)
    _, vec = fiedler(L, normalization)
    return vec >= 0


def partition_quality(W: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    """Conductance / modularity of an arbitrary bipartition.

    Computable on a benign prompt too -- there the best available cut is a bad
    one, which is exactly the signal. Comparing this quantity across benign
    and injected prompts is the detection experiment.
    """
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    a = np.flatnonzero(mask)
    b = np.flatnonzero(~mask)
    if a.size == 0 or b.size == 0:
        return {"cut_conductance": float("nan"), "cut_modularity": float("nan")}
    cut = float(A[np.ix_(a, b)].sum()) * 2.0
    vol_a, vol_b = float(A[a].sum()), float(A[b].sum())
    return {
        "cut_conductance": cut / max(min(vol_a, vol_b), EPS),
        "cut_modularity": _modularity(A, mask, float(A.sum())),
    }


def partition_agreement(discovered: np.ndarray, roles: np.ndarray) -> Dict[str, float]:
    """Does the discovered cut land on the injection boundary?

    This is the step that turns "the graph splits in two" into "the graph
    splits along the injection". Accuracy is taken over both label
    orientations since the sign of an eigenvector is arbitrary.
    """
    truth = roles == ROLE_INJECTION
    if truth.sum() == 0 or truth.sum() == truth.size:
        return {"accuracy": float("nan"), "ari": float("nan")}
    acc = max((discovered == truth).mean(), (discovered != truth).mean())
    return {"accuracy": float(acc), "ari": _ari(discovered, truth)}


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    """Adjusted Rand Index for two binary partitions (no sklearn needed)."""
    from math import comb

    n = a.size
    table = np.zeros((2, 2), dtype=np.int64)
    for i in (0, 1):
        for j in (0, 1):
            table[i, j] = np.sum((a == bool(i)) & (b == bool(j)))
    sum_comb = sum(comb(int(v), 2) for v in table.ravel())
    sum_a = sum(comb(int(v), 2) for v in table.sum(axis=1))
    sum_b = sum(comb(int(v), 2) for v in table.sum(axis=0))
    total = comb(n, 2)
    if total == 0:
        return float("nan")
    expected = sum_a * sum_b / total
    max_index = 0.5 * (sum_a + sum_b)
    if abs(max_index - expected) < EPS:
        return 0.0
    return float((sum_comb - expected) / (max_index - expected))


# --------------------------------------------------------------------------
# Spectral clustering used for the figure's node colours
# --------------------------------------------------------------------------
def spectral_clusters(W: np.ndarray, k: int = 3, seed: int = 42,
                      normalization: str = "sym") -> np.ndarray:
    """k-means on the low Laplacian eigenvectors (u2, u3).

    Reproduces the colouring of the reference figure. Note this is plain
    spectral clustering, not modularity or Leiden community detection.
    """
    L = laplacian(W, normalization, remove_self_loops=True)
    _, vecs = spectrum(L, normalization)
    emb = vecs[:, 1:1 + max(k - 1, 1)]
    return _kmeans(emb, k=k, seed=seed)


def _kmeans(X: np.ndarray, k: int, seed: int = 42, iters: int = 100) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)]
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=-1)
        new = d.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for j in range(k):
            pts = X[labels == j]
            if pts.size:
                centers[j] = pts.mean(axis=0)
    return labels


def metrics_table(attentions: List[np.ndarray], roles: np.ndarray, config) -> List[Dict]:
    """Per-layer metrics + cut statistics for one prompt.

    ``attentions`` is the list of [H, T, T] arrays, one per layer.
    """
    from .graphs import bandwidth, build_graph

    rows = []
    for layer_idx, attn in enumerate(attentions):
        g = build_graph(
            attn, roles,
            symmetrization=config.symmetrization,
            head_aggregation=config.head_aggregation,
            drop_template=config.drop_template,
            drop_sink=config.drop_sink,
        )
        W, kept_roles = g["W"], g["roles"]
        row = {"layer": layer_idx, "n_nodes": W.shape[0]}
        row.update(layer_metrics(W, config.normalization, config.remove_self_loops))
        row.update(cut_statistics(W, kept_roles))
        discovered = fiedler_partition(W, config.normalization, config.remove_self_loops)
        row.update(partition_quality(W, discovered))
        row.update(partition_agreement(discovered, kept_roles))
        row["bandwidth"] = bandwidth(W)
        rows.append(row)
    return rows
