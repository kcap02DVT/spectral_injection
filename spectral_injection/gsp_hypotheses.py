"""Seven graph-signal and topology hypotheses, measured on the attention graph.

What survived triage of a list of fifteen. The eight dropped ones were either
already answered by the existing pipeline (lambda2, modularity, hfer,
PageRank enrichment, eigenvector centrality), dead by construction on a
causal decoder (a directed cut from an earlier block to a later one is
identically zero; the graph is a DAG so no directed cycle exists), or
premised on things absent from BIPIA (system-prompt tokens are dropped by
``drop_template``; no attack in 120 contains a lexical pivot such as
"ignore" or "instead of").

BLIND -- no labels, so a detector could use them:

    bipartite_lmax  largest eigenvalue of L_sym. It equals 2 if and only if
                    the graph is bipartite, so the gap to 2 measures how far
                    the attention polarises into two sides that only talk to
                    each other.                                   (hyp. 8)
    edge_entropy    normalised Shannon entropy of the off-diagonal weight
                    distribution. Low = the head concentrates its mass on a
                    few edges.                                    (hyp. 9)
    radius          mean weighted shortest-path length, cost -log(w/w_max),
                    on the sparsified graph: the effective radius.(hyp. 15)
    kcore_max       largest core number of the sparsified graph.  (hyp. 11)
    motif_trans     density of transitive triangles i<-j<-k, i<-k in the
                    directed graph. Cyclic motifs are omitted: the causal
                    graph is a DAG and their count is always zero.(hyp. 14)
    spec_wass       Wasserstein-1 distance between this layer's Laplacian
                    spectrum and layer 0's: how far the layer has drifted
                    from the input.                               (hyp. 13)

LABELLED -- need the span, so they describe the mechanism, never detect:

    betw_ratio      mean betweenness of span nodes over mean of host nodes.
                    Approximated from sampled sources.             (hyp. 5)
    betw_boundary   betweenness of the two span-boundary nodes over the
                    graph median: are they the bridges?            (hyp. 5)
    kcore_ratio     mean coreness of span over host.              (hyp. 11)
    eigcent_ratio   mean eigenvector centrality of span over host. (hyp. 6,
                    recomputed here so every measure shares one graph)
"""

from typing import Dict, List, Sequence

import numpy as np

from .graphs import build_graph, laplacian, spectrum
from .prompts import ROLE_INJECTION

BLIND = ["bipartite_lmax", "edge_entropy", "radius", "kcore_max",
         "motif_trans", "spec_wass"]
LABELLED = ["betw_ratio", "betw_boundary", "kcore_ratio", "eigcent_ratio"]
NAMES = BLIND + LABELLED

EDGE_QUANTILE = 0.90     # keep the strongest 10 % of off-diagonal weights
BETW_SOURCES = 24        # sampled sources for approximate betweenness
EPS = 1e-12


def _sparsify(W: np.ndarray):
    """Binary adjacency of the strongest off-diagonal weights."""
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    off = A[A > 0]
    if off.size == 0:
        return np.zeros_like(A, dtype=bool), A
    thr = np.quantile(off, EDGE_QUANTILE)
    return A >= max(thr, EPS), A


def _edge_entropy(W: np.ndarray) -> float:
    A = np.array(W, dtype=np.float64, copy=True)
    np.fill_diagonal(A, 0.0)
    w = A[np.triu_indices_from(A, 1)]
    w = w[w > 0]
    if w.size < 2:
        return 0.0
    p = w / w.sum()
    return float(-(p * np.log(p)).sum() / np.log(w.size))


def _radius(B: np.ndarray, A: np.ndarray) -> float:
    """Mean finite weighted shortest-path length on the sparsified graph."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    w = A * B
    wmax = w.max()
    if wmax <= EPS:
        return float("nan")
    cost = np.where(B, -np.log(np.clip(w / wmax, 1e-9, 1.0)) + 1e-3, 0.0)
    d = shortest_path(csr_matrix(cost), method="D", directed=False)
    finite = np.isfinite(d) & (d > 0)
    return float(d[finite].mean()) if finite.any() else float("nan")


def _kcore(B: np.ndarray) -> np.ndarray:
    """Core number per node, by repeated peeling (unweighted)."""
    deg = B.sum(1).astype(int)
    alive = np.ones(len(deg), bool)
    core = np.zeros(len(deg), int)
    k = 0
    while alive.any():
        k = max(k, int(deg[alive].min()))
        while True:
            drop = alive & (deg <= k)
            if not drop.any():
                break
            core[drop] = k
            alive &= ~drop
            deg = (B & alive[None, :] & alive[:, None]).sum(1).astype(int)
    return core


def _motif_trans(B: np.ndarray) -> float:
    """Transitive triangles per possible triple, on the causal direction."""
    n = B.shape[0]
    if n < 3:
        return 0.0
    L = np.tril(B, -1).astype(np.float64)      # i receives from j < i
    tri = float(np.trace(L @ L @ L.T))
    return tri / max(n * (n - 1) * (n - 2) / 6.0, 1.0)


def _betweenness(B: np.ndarray, rng) -> np.ndarray:
    """Brandes from sampled sources, unweighted, on the sparsified graph."""
    import networkx as nx
    G = nx.from_numpy_array(B.astype(np.int8))
    k = min(BETW_SOURCES, G.number_of_nodes())
    bc = nx.betweenness_centrality(G, k=k, seed=int(rng.integers(1 << 30)),
                                   normalized=True)
    return np.array([bc[i] for i in range(B.shape[0])])


def _eigcent(W: np.ndarray, iters: int = 60) -> np.ndarray:
    v = np.ones(W.shape[0]) / W.shape[0]
    for _ in range(iters):
        v = W @ v
        n = np.linalg.norm(v)
        if n <= EPS:
            break
        v = v / n
    return np.abs(v)


def _ratio(x: np.ndarray, mask: np.ndarray) -> float:
    a, b = x[mask], x[~mask]
    if a.size == 0 or b.size == 0 or abs(b.mean()) < EPS:
        return float("nan")
    return float(a.mean() / b.mean())


def layer_values(W: np.ndarray, roles: np.ndarray, ref_spec: np.ndarray,
                 rng) -> np.ndarray:
    """One layer -> the 10 measures."""
    from scipy.stats import wasserstein_distance

    inj = roles == ROLE_INJECTION
    B, A = _sparsify(W)
    vals, _ = spectrum(laplacian(W, "sym", True), "sym")

    out = {}
    out["bipartite_lmax"] = float(vals[-1])
    out["edge_entropy"] = _edge_entropy(W)
    out["radius"] = _radius(B, A)
    core = _kcore(B)
    out["kcore_max"] = float(core.max())
    out["motif_trans"] = _motif_trans(B)
    out["spec_wass"] = float(wasserstein_distance(vals, ref_spec))

    bc = _betweenness(B, rng)
    out["betw_ratio"] = _ratio(bc, inj)
    idx = np.flatnonzero(inj)
    if idx.size and np.median(bc) > EPS:
        edge_nodes = bc[[idx[0], idx[-1]]].mean()
        out["betw_boundary"] = float(edge_nodes / np.median(bc))
    else:
        out["betw_boundary"] = float("nan")
    out["kcore_ratio"] = _ratio(core.astype(float), inj)
    out["eigcent_ratio"] = _ratio(_eigcent(W), inj)
    return np.array([out[n] for n in NAMES])


def stack(attentions: Sequence[np.ndarray], roles: np.ndarray,
          layers: Sequence[int], config) -> np.ndarray:
    """[len(layers), 10] for one prompt."""
    rng = np.random.default_rng(0)
    graphs = []
    for l in layers:
        g = build_graph(attentions[l], roles,
                        symmetrization=config.symmetrization,
                        head_aggregation=config.head_aggregation,
                        drop_template=config.drop_template,
                        drop_sink=config.drop_sink)
        graphs.append(g)
    ref, _ = spectrum(laplacian(graphs[0]["W"], "sym", True), "sym")
    return np.stack([layer_values(g["W"], g["roles"], ref, rng)
                     for g in graphs])
