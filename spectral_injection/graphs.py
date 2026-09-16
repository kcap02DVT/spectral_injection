"""Attention-graph construction, delegated to spectral-trust.

Symmetrisation, head aggregation, the Laplacian and the eigendecomposition are
all computed by ``spectral_trust`` (v0.3.0) rather than reimplemented here, so
there is exactly one definition of the graph in play and the figures, the cut
statistics and the per-head metrics cannot drift apart.

What stays local is what the library has no concept of: the benign/injection
roles, and therefore the node filtering that depends on them.

**spectral-trust is a hard requirement.** There is no numpy fallback on
purpose: a fallback that quietly takes over would mean two runs producing
different numbers with nothing to distinguish them.

Why each step matters:

* **Symmetrisation** -- a decoder-only model's attention is lower-triangular
  (causal). Treated as an undirected graph without symmetrising, every
  spectral quantity is meaningless.
* **sym normalisation** -- the symmetric normalised Laplacian has spectrum in
  [0, 2] whatever the sequence length, so a benign prompt and its longer
  injected variant remain comparable. This is the v0.3.0 default and it is
  what makes the length confound disappear.
* **Sink removal** -- the first token (and whatever else absorbs most of the
  attention mass) is a hub connected to everything. Keeping it bridges the two
  token families and erases the very cut we are looking for.
"""

from typing import Dict, Optional, Tuple

import numpy as np

from .prompts import ROLE_BENIGN, ROLE_INJECTION, ROLE_TEMPLATE

EPS = 1e-12


# --------------------------------------------------------------------------
# spectral-trust plumbing
# --------------------------------------------------------------------------
def _require():
    """Import spectral-trust, or explain precisely why it is missing."""
    try:
        import torch
        from spectral_trust import GSPConfig, GraphConstructor, SpectralAnalyzer
    except Exception as exc:                                # noqa: BLE001
        import sys
        raise RuntimeError(
            f"spectral-trust (et torch) sont requis : {type(exc).__name__}: "
            f"{exc}\n  interpréteur : {sys.executable}\n"
            f"  activez le venv du projet : "
            f".\\spectral_injection\\env\\Scripts\\Activate.ps1"
        ) from exc
    return torch, GSPConfig, GraphConstructor, SpectralAnalyzer


def gsp_config(symmetrization: str = "symmetric",
               head_aggregation: str = "uniform",
               normalization: str = "sym",
               remove_self_loops: bool = True,
               eigen_solver: str = "dense"):
    """A GSPConfig carrying only the fields that change a graph."""
    _, GSPConfig, _, _ = _require()
    return GSPConfig(
        symmetrization=symmetrization, head_aggregation=head_aggregation,
        normalization=normalization, remove_self_loops=remove_self_loops,
        eigen_solver=eigen_solver,
    )


def _tensor(array: np.ndarray):
    torch, _, _, _ = _require()
    return torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))


# --------------------------------------------------------------------------
# Head handling
# --------------------------------------------------------------------------
def symmetrize(attn: np.ndarray, method: str = "symmetric") -> np.ndarray:
    """Symmetrise an attention tensor. Works on [..., T, T]."""
    _, _, GraphConstructor, _ = _require()
    gc = GraphConstructor(gsp_config(symmetrization=method))
    return gc.symmetrize_attention(_tensor(attn)).numpy()


def aggregate_heads(attn: np.ndarray, method: str = "uniform") -> np.ndarray:
    """Collapse [H, T, T] to [T, T].

    Only for visualisation and qualitative layer trajectories. For detection,
    keep heads separate (see metrics.per_head_layer_metrics): head averaging
    dilutes the few heads that actually carry the routing anomaly.
    """
    if attn.ndim != 3:
        raise ValueError(f"expected [H, T, T], got {attn.shape}")
    _, _, GraphConstructor, _ = _require()
    gc = GraphConstructor(gsp_config(head_aggregation=method))
    # The library insists on a batch axis: [B, H, Q, K].
    return gc.aggregate_heads(_tensor(attn)[None])[0].numpy()


# --------------------------------------------------------------------------
# Node filtering
# --------------------------------------------------------------------------
def detect_sink(W: np.ndarray) -> int:
    """Index of the token receiving the most attention mass."""
    return int(W.sum(axis=0).argmax())


def keep_mask(
    roles: np.ndarray,
    W: np.ndarray,
    drop_template: bool = True,
    drop_sink: bool = True,
) -> np.ndarray:
    """Boolean mask of nodes to keep in the graph."""
    keep = np.ones(len(roles), dtype=bool)
    if drop_template:
        keep &= roles != ROLE_TEMPLATE
    if drop_sink:
        sink = detect_sink(W)
        keep[sink] = False
        keep[0] = False          # BOS is a sink even when not the argmax
    if keep.sum() < 3:
        raise RuntimeError(
            "Fewer than 3 nodes left after filtering; the prompt is too short "
            "or drop_template removed everything."
        )
    return keep


# --------------------------------------------------------------------------
# Laplacian and spectrum
# --------------------------------------------------------------------------
def laplacian(
    W: np.ndarray,
    normalization: str = "sym",
    remove_self_loops: bool = True,
) -> np.ndarray:
    """Graph Laplacian of a symmetric non-negative weight matrix.

    Note the library builds ``I - D^-1/2 W D^-1/2`` directly, so an isolated
    node gets 1 on the diagonal where a ``D - A`` construction would leave 0.
    """
    _, _, GraphConstructor, _ = _require()
    gc = GraphConstructor(gsp_config(normalization=normalization,
                                     remove_self_loops=remove_self_loops))
    # construct_laplacian adds a batch axis for sym/rw; feed it one so the
    # shape is the same whatever the normalization, then take it back off.
    L = gc.construct_laplacian(_tensor(W)[None])[0]
    return L.numpy().astype(np.float64)


def spectrum(L: np.ndarray, normalization: str = "sym"
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Eigenvalues (ascending, clamped at 0) and eigenvectors of a Laplacian.

    Delegated to ``SpectralAnalyzer``, which dispatches on the operator: ``rw``
    Laplacians are not symmetric and get a general solver, rather than being
    symmetrised first and silently diagonalising a different matrix.

    ``eigen_solver="dense"`` is not optional here -- see the note on
    ``RunConfig.eigen_solver``.
    """
    _, _, _, SpectralAnalyzer = _require()
    analyzer = SpectralAnalyzer(gsp_config(normalization=normalization,
                                           eigen_solver="dense"))
    vals, vecs = analyzer.compute_eigendecomposition(
        np.ascontiguousarray(L, dtype=np.float64))
    order = np.argsort(vals)
    return np.clip(vals[order], 0.0, None), vecs[:, order]


def fiedler(L: np.ndarray, normalization: str = "sym"
            ) -> Tuple[float, np.ndarray]:
    """Algebraic connectivity (lambda_2) and its eigenvector."""
    vals, vecs = spectrum(L, normalization)
    return float(vals[1]), vecs[:, 1]


def build_graph(
    attn_layer: np.ndarray,
    roles: np.ndarray,
    *,
    symmetrization: str = "symmetric",
    head_aggregation: str = "uniform",
    drop_template: bool = True,
    drop_sink: bool = True,
) -> Dict[str, np.ndarray]:
    """Attention tensor [H, T, T] -> filtered symmetric graph.

    Returns a dict with the weight matrix ``W``, the surviving ``roles``, and
    ``keep`` (the boolean mask, so token strings can be filtered identically).
    """
    W_full = aggregate_heads(symmetrize(attn_layer, symmetrization), head_aggregation)
    W_full = np.asarray(W_full, dtype=np.float64)
    keep = keep_mask(roles, W_full, drop_template, drop_sink)
    return {
        "W": W_full[np.ix_(keep, keep)],
        "roles": roles[keep],
        "keep": keep,
        "sink_index": detect_sink(W_full),
    }


def bandwidth(W: np.ndarray) -> float:
    """Weight-averaged |i - j|: how local the attention is.

    Small values mean a band around the diagonal (local attention), large
    values mean long-range links. Reported under each adjacency panel.
    """
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    total = A.sum()
    if total <= EPS:
        return 0.0
    n = A.shape[0]
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return float((A * dist).sum() / total)
