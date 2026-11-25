"""Figures: circular node-link attention graph + adjacency heatmap.

Layout follows the reference figure. Each dot is a token placed in reading
order around the circle, each line is an attention link coloured by strength,
and dot size is how much attention a token receives.

Two colourings:

``role``     blue = benign / host tokens, red = injected tokens. This is the
             view that answers "do the two families form separate clusters?".
``cluster``  3-way spectral clustering (k-means on the low Laplacian
             eigenvectors u2, u3), as in the reference figure. Useful to see
             what structure the graph has on its own, without labels.

One caveat worth keeping in mind when reading the circular panel: nodes are
placed in reading order, so a contiguous injection occupies a contiguous arc
whatever the attention does. Visual grouping there is partly a property of the
layout. The reordered adjacency heatmap and the cut statistics are the
unambiguous evidence; the circle is for intuition.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PathCollection
from matplotlib.colors import LogNorm, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.path import Path

from .graphs import bandwidth
from .metrics import spectral_clusters
from .prompts import ROLE_BENIGN, ROLE_INJECTION, pretty_token

COLOR_BENIGN = "#1f77b4"
COLOR_INJECTION = "#d62728"
CLUSTER_COLORS = ["#2ca02c", "#ff7f0e", "#1f77b4", "#9467bd", "#8c564b"]

#: Defaults for the edge ramp, mirroring the RunConfig fields of the same name.
EDGE_STYLE_DEFAULTS = dict(
    edge_scale="log", edge_gamma=3.0, edge_alpha_min=0.003,
    edge_alpha_max=0.90, edge_width_min=0.10, edge_width_max=2.4,
    edge_ref_percentile=99.5, edge_floor_percentile=2.0,
    edge_ink_budget=250.0, edge_curvature=0.75, edge_outward=0.55,
    edge_outward_span=6.0, edge_color_by="family", node_gap_frac=0.04,
)


def edge_style_from(config) -> Dict[str, float]:
    """Pull the edge-ramp settings off a RunConfig, with defaults."""
    return {k: getattr(config, k, v) for k, v in EDGE_STYLE_DEFAULTS.items()}


#: Edge families, by the roles of the two endpoints. The cross family gets a
#: hue outside the blue/red pair so it cannot be mistaken for either side.
COLOR_EDGE_BB = "#1f77b4"      # host <-> host
COLOR_EDGE_II = "#d62728"      # injection <-> injection
COLOR_EDGE_BI = "#00a878"      # host <-> injection: the cut being measured


def _ring_angles(roles: np.ndarray, gap_frac: float = 0.04) -> np.ndarray:
    """Angles around the circle, with a gap opened at every role boundary.

    Reading order is preserved -- node i still comes after node i-1 -- but a
    blank arc is inserted wherever the role changes. Without it a 9-token
    injection inside 176 host tokens occupies 17 degrees of an unbroken ring
    of dots and cannot be picked out at all.

    The gap is expressed in node-widths, so it stays visible as n grows.
    """
    n = len(roles)
    if n == 0:
        return np.zeros(0)
    r = np.asarray(roles)
    extra = np.zeros(n, dtype=float)
    if n > 1:
        changes = np.flatnonzero(r[1:] != r[:-1]) + 1
        gap = max(gap_frac * n, 1.5)
        extra[changes] = gap
        wrap = gap if r[0] != r[-1] else 0.0
    else:
        wrap = 0.0
    pos = np.cumsum(extra + 1.0) - 1.0
    total = pos[-1] + 1.0 + wrap
    return np.pi / 2 + 2.0 * np.pi * pos / max(total, 1e-9)


def _node_colors(roles: np.ndarray, W: np.ndarray, mode: str, seed: int) -> List[str]:
    if mode == "cluster":
        labels = spectral_clusters(W, k=3, seed=seed)
        return [CLUSTER_COLORS[l % len(CLUSTER_COLORS)] for l in labels]
    return [COLOR_INJECTION if r == ROLE_INJECTION else COLOR_BENIGN for r in roles]


def _edge_ramp(weights: np.ndarray, gamma: float, ref_percentile: float,
               scale: str = "log", floor_percentile: float = 2.0) -> np.ndarray:
    """Map attention weights to a [0, 1] visual intensity.

    Every ramp here is strictly monotone in the weight: a stronger edge is
    never drawn fainter, thinner or paler than a weaker one. What the scale
    chooses is how the *gaps* are spread across the visible range.

    That choice matters because attention spans decades -- on the example
    prompt the strongest pair is ~300x the median, and the sink's edges are
    ~30000x the weakest. On a linear ramp the sink saturates the top and the
    other 3000-odd edges land together on the alpha floor: the picture becomes
    a star, and every distinction *among* those edges is lost. The hierarchy
    is not more visible on a linear scale, it is less.

    ``log``     default. Weights are mapped between a low and a high
                percentile in log space, so each decade gets equal room. The
                hub stays plainly dominant; the local structure under it
                becomes readable.
    ``rank``    intensity is the edge's rank. Maximum contrast, but magnitude
                is gone -- two edges a decade apart can look adjacent, and the
                disc fills in. Offered for completeness; rarely what you want.
    ``linear``  raw proportionality, the pre-existing behaviour.

    ``gamma`` then bends whichever ramp was produced. With thousands of
    overlapping lines alpha accumulates, so raising gamma pushes the weak bulk
    back down when the disc starts filling in.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return w

    hi = float(np.percentile(w, ref_percentile))
    if not np.isfinite(hi) or hi <= 0.0:
        hi = float(w.max())
    if hi <= 0.0:
        return np.zeros_like(w)

    if scale == "rank":
        s = np.empty(w.size, dtype=float)
        order = np.argsort(w, kind="mergesort")
        s[order] = np.linspace(0.0, 1.0, w.size)
    elif scale == "log":
        lo = float(np.percentile(w, floor_percentile))
        # A dense graph can have exact zeros, and a degenerate graph can have
        # lo == hi; fall back to the smallest positive weight in either case.
        if not np.isfinite(lo) or lo <= 0.0 or lo >= hi:
            positive = w[w > 0.0]
            lo = float(positive.min()) if positive.size else hi * 1e-6
        if lo >= hi:
            lo = hi * 1e-6
        s = ((np.log(np.clip(w, lo, hi)) - np.log(lo))
             / (np.log(hi) - np.log(lo)))
    elif scale == "linear":
        s = np.clip(w / hi, 0.0, 1.0)
    else:
        raise ValueError(
            f"unknown edge scale: {scale!r} (expected log, rank or linear)")

    return np.clip(s, 0.0, 1.0) ** float(gamma)


def plot_node_link(ax, W: np.ndarray, roles: np.ndarray, tokens: Sequence[str],
                   title: str = "", top_k_edges: Optional[int] = None,
                   max_label_tokens: int = 24, color_by: str = "role",
                   seed: int = 42, edge_scale: str = "log",
                   edge_gamma: float = 3.0,
                   edge_alpha_min: float = 0.003, edge_alpha_max: float = 0.90,
                   edge_width_min: float = 0.10, edge_width_max: float = 2.4,
                   edge_ref_percentile: float = 99.5,
                   edge_floor_percentile: float = 2.0,
                   edge_ink_budget: Optional[float] = 250.0,
                   edge_curvature: float = 0.75,
                   edge_outward: float = 0.55,
                   edge_outward_span: float = 6.0,
                   edge_color_by: str = "family",
                   node_gap_frac: float = 0.04) -> Dict[str, float]:
    """Circular node-link view.

    ``top_k_edges=None`` draws every non-zero edge -- n(n-1)/2 of them, so
    17000 on a 185-token prompt. Three things keep that readable, and all
    three are needed:

    * one collection instead of one ``ax.plot`` per edge, or the render takes
      minutes;
    * ``edge_ink_budget``, which rescales alpha so the *total* opacity laid
      down is the same whatever the graph size. A curve tuned on 3000 edges
      saturates at 17000 -- ink grows with the edge count, not with the
      picture, so it has to be normalised rather than fixed;
    * ``edge_curvature``, which bows each chord towards the centre. Straight
      chords between nearby nodes pile onto the same lines and merge into a
      solid band; bowed ones separate and the crossings stay distinguishable.

    Pass ``edge_curvature=0`` for straight chords and ``edge_ink_budget=None``
    to let the raw ramp through.
    """
    n = W.shape[0]
    angles = _ring_angles(roles, node_gap_frac)
    xs, ys = np.cos(angles), np.sin(angles)

    in_deg = W.sum(axis=0)

    edges = np.array(W, dtype=float, copy=True)
    np.fill_diagonal(edges, 0.0)

    ii, jj = np.nonzero(np.triu(edges, k=1) > 0.0)
    weights = edges[ii, jj]
    n_total = int(weights.size)

    if top_k_edges is not None and 0 < top_k_edges < n_total:
        keep = np.argpartition(weights, n_total - top_k_edges)[n_total - top_k_edges:]
        ii, jj, weights = ii[keep], jj[keep], weights[keep]

    n_drawn = int(weights.size)
    if n_drawn:
        ramp = _edge_ramp(weights, edge_gamma, edge_ref_percentile,
                          scale=edge_scale,
                          floor_percentile=edge_floor_percentile)
        # Weak links first so the strong ones land on top of the pile. In
        # family mode the cross edges go last whatever their weight: they are
        # the ones being looked for, and they are usually the weak ones, so
        # left in weight order they end up buried under the two blocks.
        if edge_color_by == "family":
            inj_all = np.asarray(roles) == ROLE_INJECTION
            is_cross = inj_all[ii] ^ inj_all[jj]
            order = np.lexsort((weights, is_cross))
        else:
            order = np.argsort(weights, kind="mergesort")
        ii, jj, ramp = ii[order], jj[order], ramp[order]

        alpha = edge_alpha_min + (edge_alpha_max - edge_alpha_min) * ramp
        # Normalise total ink so the same settings hold from 3k to 17k edges.
        ink = float(alpha.sum())
        if edge_ink_budget is not None and ink > edge_ink_budget > 0:
            alpha = alpha * (edge_ink_budget / ink)
        alpha = np.clip(alpha, 0.0, 1.0)

        if edge_color_by == "family":
            # Hue says which families the edge joins, intensity still says how
            # strong it is. The cross edges -- the cut the whole analysis is
            # about -- become findable instead of being one shade among 17000.
            inj = np.asarray(roles) == ROLE_INJECTION
            a_inj, b_inj = inj[ii], inj[jj]
            base = np.empty((ramp.size, 4), dtype=float)
            base[:] = to_rgba(COLOR_EDGE_BB)
            base[a_inj & b_inj] = to_rgba(COLOR_EDGE_II)
            base[a_inj ^ b_inj] = to_rgba(COLOR_EDGE_BI)
            # Weak edges fade towards white so intensity still reads as weight.
            rgba = np.ones((ramp.size, 4), dtype=float)
            tint = 0.35 + 0.65 * ramp[:, None]
            rgba[:, :3] = base[:, :3] * tint + (1.0 - tint)
        else:
            rgba = plt.cm.YlOrRd(0.30 + 0.70 * ramp)
        rgba[:, 3] = alpha
        widths = edge_width_min + (edge_width_max - edge_width_min) * ramp

        if edge_curvature > 0.0 or edge_outward > 0.0:
            # Quadratic Bezier whose control point sits on the angular
            # bisector, at a radius that depends on how far apart the two
            # endpoints are.
            #
            # A single uniform pull towards the centre does the wrong thing:
            # the deviation it produces is proportional to the distance from
            # the centre to the chord's midpoint, which is *largest* for
            # neighbouring tokens. A link spanning 0.03 of the rim ends up
            # looping 0.30 inwards -- short-range structure is destroyed
            # precisely where it should be clearest.
            #
            # So short-range links arc *outwards*, clear of their own markers,
            # and long chords bow inwards where there is room.
            #
            # The outward height has to grow with the span, the way arcs nest
            # in an arc diagram: a flat bulge puts spans 1, 2, 5 and 20 at the
            # same radius and they overlap into one fuzzy annulus. Here span 1
            # hugs the ring, span 2 sits just above it, and so on, so a link
            # between consecutive tokens is finally a shape of its own.
            #
            # u is 0 for adjacent nodes, 1 for opposite ones; t is the span
            # measured in units of edge_outward_span, where the bulge peaks.
            delta = angles[jj] - angles[ii]
            delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
            mid_angle = angles[ii] + 0.5 * delta
            u = np.abs(delta) / np.pi
            u_ref = float(np.clip(2.0 * edge_outward_span / max(n, 1),
                                  1e-3, 1.0))
            t = u / u_ref
            r_ctrl = (1.0 - edge_curvature * u
                      + edge_outward * t * np.exp(1.0 - t))
            cx, cy = r_ctrl * np.cos(mid_angle), r_ctrl * np.sin(mid_angle)

            codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
            paths = [Path([(xs[a], ys[a]), (cx[k], cy[k]), (xs[b], ys[b])],
                          codes)
                     for k, (a, b) in enumerate(zip(ii, jj))]
            ax.add_collection(PathCollection(
                paths, facecolors="none", edgecolors=rgba, linewidths=widths,
                zorder=1, capstyle="round"))
        else:
            segments = np.stack(
                [np.column_stack([xs[ii], ys[ii]]),
                 np.column_stack([xs[jj], ys[jj]])], axis=1)
            ax.add_collection(LineCollection(
                segments, colors=rgba, linewidths=widths, zorder=1,
                capstyle="round"))

    # Dots have to shrink as the ring fills up, or neighbours touch and the
    # short-range links between them are hidden under their own markers.
    node_scale = float(np.clip(80.0 / max(n, 1), 0.28, 1.0))
    sizes = (30 + 260 * (in_deg / (in_deg.max() + 1e-9))) * node_scale
    colors = _node_colors(roles, W, color_by, seed)
    ax.scatter(xs, ys, s=sizes, c=colors, zorder=3,
               edgecolors="white", linewidths=0.5)

    step = max(1, n // max(max_label_tokens, 1))
    for i in range(0, n if max_label_tokens > 0 else 0, step):
        label = pretty_token(tokens[i])
        if not label:
            continue
        # Pushed clear of the outward bulge of the short-range arcs (~1.18).
        ax.text(1.28 * xs[i], 1.28 * ys[i], label[:12], fontsize=6,
                ha="center", va="center", color="black",
                path_effects=[pe.withStroke(linewidth=1.6, foreground="white")])

    hub = int(in_deg.argmax())
    ax.scatter([xs[hub]], [ys[hub]], s=sizes[hub] * 1.8, c=[colors[hub]],
               zorder=4, edgecolors="black", linewidths=1.2)

    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-1.55, 1.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)

    return {"hub_token": pretty_token(tokens[hub]) or "?",
            "bandwidth": bandwidth(W),
            "n_edges_drawn": n_drawn,
            "n_edges_total": n_total}


def adjacency_color_range(W: np.ndarray, floor_percentile: float = 40.0
                          ) -> Optional[Tuple[float, float]]:
    """(vmin, vmax) of the log colour scale for an adjacency matrix.

    vmax is the largest off-diagonal weight, vmin a percentile of the positive
    ones, so the bottom of the range is not spent on near-zero noise.
    """
    A = np.asarray(W, dtype=float)
    positive = A[(A > 0.0) & ~np.eye(A.shape[0], dtype=bool)]
    if not positive.size:
        return None
    vmax = float(positive.max())
    vmin = float(np.percentile(positive, floor_percentile))
    if not np.isfinite(vmin) or vmin <= 0.0 or vmin >= vmax:
        vmin = vmax * 1e-4
    return vmin, vmax


def plot_adjacency(ax, W: np.ndarray, roles: np.ndarray, title: str = "",
                   reorder: bool = False, caption: str = "",
                   aspect: str = "equal", log_color: bool = True,
                   floor_percentile: float = 40.0,
                   color_range: Optional[Tuple[float, float]] = None) -> None:
    """Adjacency heatmap. ``reorder`` groups benign then injection tokens.

    The reordered version is the decisive picture: if the bicluster hypothesis
    holds, two dense blocks appear on the diagonal and the off-diagonal
    corners stay dark. Nothing about that depends on the layout.

    ``log_color`` matters for the same reason it matters on the node-link
    panel: the attention sink's row and column are orders of magnitude above
    everything else, and on a linear colour scale they take the whole range
    while the blocks this panel exists to show stay at the bottom, rendered
    black. A log norm gives each decade equal colour, so the block structure
    becomes visible next to the hub instead of underneath it.

    ``color_range`` imposes (vmin, vmax) instead of deriving them from ``W``,
    so two panels share one scale and their colours can be compared.
    """
    A = np.array(W, dtype=float, copy=True)
    order = np.arange(A.shape[0])
    if reorder:
        order = np.concatenate([np.flatnonzero(roles == ROLE_BENIGN),
                                np.flatnonzero(roles == ROLE_INJECTION)])
        A = A[np.ix_(order, order)]

    masked = np.ma.masked_array(A, mask=np.eye(A.shape[0], dtype=bool))
    cmap = plt.cm.magma.copy()
    cmap.set_bad("black")

    norm = None
    if log_color and color_range is not None:
        norm = LogNorm(vmin=color_range[0], vmax=color_range[1], clip=True)
    elif log_color:
        vrange = adjacency_color_range(A, floor_percentile)
        if vrange is not None:
            norm = LogNorm(vmin=vrange[0], vmax=vrange[1], clip=True)

    ax.imshow(masked, cmap=cmap, aspect=aspect, interpolation="nearest",
              norm=norm)

    reordered_roles = roles[order]
    inj = np.flatnonzero(reordered_roles == ROLE_INJECTION)
    if inj.size:
            lo, hi = inj.min(), inj.max() + 1
            ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                                linewidth=2.5, edgecolor="cyan",
                                facecolor="none", linestyle="-",
                                clip_on=False, zorder=10))

    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if caption:
        ax.set_xlabel(caption, fontsize=7)


def aligned_abs_diff(benign: Dict, injected: Dict) -> Tuple[np.ndarray, int]:
    """|W_injected - W_benign| in the injected prompt's token space.

    The two prompts do not have the same length, so an index-by-index
    subtraction would compare token k of one with an unrelated token k of the
    other as soon as the insertion point is passed -- a diagonal smear that is
    pure misalignment. Tokens are matched by sequence alignment instead
    (difflib on the token strings, autojunk off so frequent tokens are not
    discarded). Entries involving a token with no benign counterpart -- the
    attack, plus any token re-split at its boundaries -- have nothing to be
    subtracted from and are left NaN rather than filled with a made-up zero.

    Returns the matrix and the number of injected-prompt tokens matched.
    """
    return aligned_diff(benign, injected, signed=False)


def aligned_diff(ref: Dict, cond: Dict, signed: bool = True,
                 pair_inserted: bool = False) -> Tuple[np.ndarray, int]:
    """``W_cond - W_ref`` (or its absolute value) in ``cond``'s token space.

    Tokens are matched by sequence alignment, as in ``aligned_abs_diff``.
    ``pair_inserted`` additionally pairs the inserted spans position by
    position when both prompts have one of the same length -- injected vs
    control: token k of the attack against token k of the control text, same
    slot, different words. Their rows and columns then say whether the attack
    draws more or less attention than neutral text at that place. With spans
    of different lengths they stay unmatched (NaN).
    """
    from difflib import SequenceMatcher

    Wr = np.asarray(ref["W"], dtype=float)
    Wc = np.asarray(cond["W"], dtype=float)
    to_ref = np.full(Wc.shape[0], -1, dtype=int)
    sm = SequenceMatcher(None, list(ref["tokens"]), list(cond["tokens"]),
                         autojunk=False)
    for blk in sm.get_matching_blocks():
        to_ref[blk.b:blk.b + blk.size] = np.arange(blk.a, blk.a + blk.size)

    if pair_inserted:
        ins_r = np.flatnonzero(ref["roles"] == ROLE_INJECTION)
        ins_c = np.flatnonzero(cond["roles"] == ROLE_INJECTION)
        if ins_r.size and ins_r.size == ins_c.size:
            to_ref[ins_c] = ins_r

    D = np.full(Wc.shape, np.nan)
    m = np.flatnonzero(to_ref >= 0)
    delta = Wc[np.ix_(m, m)] - Wr[np.ix_(to_ref[m], to_ref[m])]
    D[np.ix_(m, m)] = delta if signed else np.abs(delta)
    return D, int(m.size)


def plot_adjacency_signed_diff(ax, D: np.ndarray, roles: np.ndarray,
                               title: str = "", caption: str = "",
                               floor_percentile: float = 40.0) -> None:
    """Signed difference heatmap: red = first condition higher, blue = lower.

    Same ordering and injection rectangle as the other adjacency panels. The
    colour scale is symmetric around zero and logarithmic on both sides
    (SymLogNorm), linear only below a threshold set at a percentile of |D|:
    differences span several decades, and a linear diverging scale would show
    only the handful of largest ones.
    """
    from matplotlib.colors import SymLogNorm

    order = np.concatenate([np.flatnonzero(roles == ROLE_BENIGN),
                            np.flatnonzero(roles == ROLE_INJECTION)])
    A = D[np.ix_(order, order)]
    off = ~np.eye(A.shape[0], dtype=bool)
    masked = np.ma.masked_array(A, mask=~np.isfinite(A) | ~off)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("#bdbdbd")

    mag = np.abs(A[np.isfinite(A) & off])
    mag = mag[mag > 0.0]
    norm = None
    if mag.size:
        vmax = float(mag.max())
        lin = float(np.percentile(mag, floor_percentile))
        if not np.isfinite(lin) or lin <= 0.0 or lin >= vmax:
            lin = vmax * 1e-4
        norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)
    im = ax.imshow(masked, cmap=cmap, aspect="equal", interpolation="nearest",
                   norm=norm)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    if norm is not None:
        # Default symlog ticks crowd around the linear zone; keep 0 and whole
        # decades clearly above the threshold, on both sides.
        k_lo = int(np.floor(np.log10(lin))) + 1
        k_hi = int(np.floor(np.log10(vmax)))
        decades = [10.0 ** k for k in range(k_lo, k_hi + 1)][-3:]
        cbar.set_ticks(sorted([-d for d in decades] + [0.0] + decades))
    cbar.ax.tick_params(labelsize=6)

    inj = np.flatnonzero(roles[order] == ROLE_INJECTION)
    if inj.size:
        lo, hi = inj.min(), inj.max() + 1
        ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                               linewidth=2.5, edgecolor="black",
                               facecolor="none", clip_on=False, zorder=10))
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if caption:
        ax.set_xlabel(caption, fontsize=7)


def diff_color_range(*Ds: np.ndarray, floor_percentile: float = 40.0
                     ) -> Optional[Tuple[float, float]]:
    """One (vmin, vmax) over several |difference| matrices, NaN ignored."""
    vals = [D[np.isfinite(D) & (D > 0.0) & ~np.eye(D.shape[0], dtype=bool)]
            for D in Ds]
    vals = np.concatenate(vals) if vals else np.array([])
    if not vals.size:
        return None
    vmax = float(vals.max())
    vmin = float(np.percentile(vals, floor_percentile))
    if not np.isfinite(vmin) or vmin <= 0.0 or vmin >= vmax:
        vmin = vmax * 1e-4
    return vmin, vmax


def plot_adjacency_diff(ax, D: np.ndarray, roles: np.ndarray, title: str = "",
                        caption: str = "", floor_percentile: float = 40.0,
                        color_range: Optional[Tuple[float, float]] = None
                        ) -> None:
    """Heatmap of an aligned |difference| matrix, ordered like the injected one.

    Same reordering, same log colour logic and same injection rectangle as
    ``plot_adjacency``, so the three panels line up cell for cell. A different
    colormap (viridis) keeps it from being read as a third attention matrix;
    unmatched entries are grey.
    """
    order = np.concatenate([np.flatnonzero(roles == ROLE_BENIGN),
                            np.flatnonzero(roles == ROLE_INJECTION)])
    A = D[np.ix_(order, order)]
    off = ~np.eye(A.shape[0], dtype=bool)
    masked = np.ma.masked_array(A, mask=~np.isfinite(A) | ~off)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#bdbdbd")

    vrange = color_range or diff_color_range(A, floor_percentile=floor_percentile)
    norm = (LogNorm(vmin=vrange[0], vmax=vrange[1], clip=True)
            if vrange is not None else None)
    im = ax.imshow(masked, cmap=cmap, aspect="equal", interpolation="nearest",
                   norm=norm)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)

    inj = np.flatnonzero(roles[order] == ROLE_INJECTION)
    if inj.size:
        lo, hi = inj.min(), inj.max() + 1
        ax.add_patch(Rectangle((lo - 0.5, lo - 0.5), hi - lo, hi - lo,
                               linewidth=2.5, edgecolor="cyan",
                               facecolor="none", clip_on=False, zorder=10))
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    if caption:
        ax.set_xlabel(caption, fontsize=7)


def _flatten_segments(segments: Sequence[Tuple[str, str]]):
    """``[(text, role)]`` -> one string plus a role per character."""
    text = "".join(t for t, _ in segments)
    roles: List[str] = []
    for chunk, role in segments:
        roles.extend([role] * len(chunk))
    return text, roles


def _wrap_spans(text: str, width: int, max_lines: int):
    """Greedy wrap to ``width`` columns; returns (start, end) per line."""
    lines, i, n = [], 0, len(text)
    while i < n and len(lines) < max_lines:
        newline = text.find("\n", i)
        end = min(i + width, n)
        if newline != -1 and newline < end:
            lines.append((i, newline))
            i = newline + 1
            continue
        if end < n:
            space = text.rfind(" ", i, end)
            if space > i:
                end = space
        lines.append((i, end))
        i = end
        while i < n and text[i] in " \n":
            i += 1
    return lines, i < n


def plot_prompt_text(ax, segments: Sequence[Tuple[str, str]], title: str = "",
                     width: int = 118, max_lines: int = 24,
                     fontsize: float = 5.6) -> None:
    """The prompt itself, with the injected span in red.

    The graph panels show *that* two families separate; this shows *what* they
    are. Without it a reader has to take on faith that the red arc is the
    attack and the blue one the host, which is the one thing a figure about
    injection detection should not ask them to assume.

    Monospace is not cosmetic here: character advance has to be constant for
    the coloured runs to line up, since each run is placed by counting
    columns rather than by measuring text.
    """
    ax.axis("off")
    if not segments:
        return
    text, roles = _flatten_segments(segments)
    lines, truncated = _wrap_spans(text, width, max_lines)

    char_w = 1.0 / width
    # Floor at half the line budget: a two-line prompt (the attack alone)
    # would otherwise be spread over the whole panel height.
    line_h = 1.0 / max(len(lines) + 1, max_lines // 2)

    for li, (start, end) in enumerate(lines):
        y = 1.0 - (li + 1) * line_h
        j = start
        while j < end:
            k = j
            while k < end and roles[k] == roles[j]:
                k += 1
            # parse_math=False: emails are full of '$', and mathtext would
            # swallow everything between two of them into italic gibberish.
            ax.text((j - start) * char_w, y, text[j:k].replace("\n", " "),
                    transform=ax.transAxes, family="monospace",
                    fontsize=fontsize, va="center", ha="left",
                    parse_math=False,
                    color=(COLOR_INJECTION if roles[j] == ROLE_INJECTION
                           else "#333333"),
                    fontweight=("bold" if roles[j] == ROLE_INJECTION
                                else "normal"))
            j = k
    if truncated:
        ax.text(0.0, 1.0 - (len(lines) + 1) * line_h,
                f"[... prompt tronqué à {max_lines} lignes ...]",
                transform=ax.transAxes, family="monospace", fontsize=fontsize,
                va="center", color="#999999", style="italic")
    if title:
        ax.set_title(title, fontsize=8, pad=3, loc="left")


def figure_pair(benign: Dict, injected: Optional[Dict], model_slug: str,
                layer: int, n_layers: int, config, out_path: str,
                benign_segments: Optional[Sequence[Tuple[str, str]]] = None,
                injected_segments: Optional[Sequence[Tuple[str, str]]] = None,
                subtitle: str = "",
                isolated: Optional[Dict] = None,
                isolated_segments: Optional[Sequence[Tuple[str, str]]] = None,
                control: Optional[Dict] = None,
                control_segments: Optional[Sequence[Tuple[str, str]]] = None,
                control_note: str = "",
                ) -> str:
    """Benign | injected [| attack alone  or  | control insertion].

    Each ``Dict`` holds ``W``, ``roles`` and ``tokens`` for one condition.

    Two conditions: the adjacency row holds benign, |difference|, injected.

    A third column -- ``isolated`` (``--compare triptych``) or ``control``
    (``--compare control``) -- turns every row into three columns, and the
    three adjacency panels share one colour scale. The |difference| row then
    holds:

    triptych  |injected - benign| under the first two columns (the attack
              alone shares no tokens with the others);
    control   |injected - benign| under the injected column and
              |control - benign| under the control column, on one shared
              scale -- the comparison the control exists for: if the two look
              alike, the injection picture is an insertion/length effect.
    """
    if isolated is not None and control is not None:
        raise ValueError("isolated et control sont exclusifs")
    conditions = [("Bénin (référence)", benign, benign_segments)]
    if injected is not None:
        conditions.append(("Avec injection", injected, injected_segments))
    third = None
    if injected is not None and isolated is not None:
        third = "isolated"
        conditions.append(("Injection seule", isolated, isolated_segments))
    elif injected is not None and control is not None:
        third = "control"
        conditions.append(("Contrôle inséré (texte d'email)", control,
                           control_segments))
    triptych = third is not None
    ncols = len(conditions)
    with_text = any(seg for _, _, seg in conditions)

    color_range = None
    if triptych:
        ranges = [r for r in (adjacency_color_range(d["W"])
                              for _, d, _ in conditions) if r is not None]
        if ranges:
            color_range = (min(r[0] for r in ranges), max(r[1] for r in ranges))

    if triptych:
        rows = ["graph", "adj", "diff"] + (["text"] if with_text else [])
        heights = [2.4, 1.35, 1.35] + ([1.6] if with_text else [])
        fig = plt.figure(figsize=(8.2 * ncols, 18.5 if with_text else 15))
    else:
        rows = ["graph", "adj"] + (["text"] if with_text else [])
        heights = [2.4, 1.35] + ([1.6] if with_text else [])
        fig = plt.figure(figsize=(8.2 * ncols, 14.0 if with_text else 11))
    gs = fig.add_gridspec(len(rows), ncols, height_ratios=heights,
                          hspace=0.16, wspace=0.12)
    r_graph, r_adj = rows.index("graph"), rows.index("adj")
    r_text = rows.index("text") if with_text else None

    # (slot, inserted-condition dict, title) for each |difference| panel.
    diffs = []
    if third == "isolated":
        adj_slot = {c: gs[r_adj, c] for c in range(3)}
        diffs = [(gs[rows.index("diff"), 0:2], injected,
                  "|injecté − bénin| (tokens alignés, même ordre)")]
    elif third == "control":
        adj_slot = {c: gs[r_adj, c] for c in range(3)}
        r_diff = rows.index("diff")
        diffs = [(gs[r_diff, 1], injected,
                  "|injecté − bénin| (tokens alignés)"),
                 (gs[r_diff, 2], control,
                  "|contrôle − bénin| (tokens alignés)")]
    elif injected is not None:
        # Two conditions: the adjacency row gets three panels -- benign,
        # |difference|, injected -- while graphs and texts keep two columns.
        sub = gs[r_adj, :].subgridspec(1, 3, wspace=0.10)
        adj_slot = {0: sub[0, 0], 1: sub[0, 2]}
        diffs = [(sub[0, 1], injected,
                  "|injecté − bénin| (tokens alignés, même ordre)")]
    else:
        adj_slot = {0: gs[r_adj, 0]}

    for col, (label, data, segments) in enumerate(conditions):
        W, roles, tokens = data["W"], data["roles"], data["tokens"]
        n_inj = int((roles == ROLE_INJECTION).sum())
        is_control = third == "control" and col == 2
        head = f"{label} — {W.shape[0]} tokens"
        if n_inj:
            head += (f"  [bleu={W.shape[0] - n_inj} · "
                     f"{'inséré' if is_control else 'rouge'}={n_inj}]")
        if is_control and control_note:
            head += f"\n{control_note}"

        info = plot_node_link(
            fig.add_subplot(gs[r_graph, col]), W, roles, tokens, title=head,
            top_k_edges=config.top_k_edges,
            max_label_tokens=config.max_label_tokens,
            color_by=config.color_by, seed=config.seed,
            **edge_style_from(config),
        )
        plot_adjacency(
            fig.add_subplot(adj_slot[col]), W, roles,
            title="Adjacence (réordonnée par rôle)" if n_inj else "Adjacence",
            reorder=bool(n_inj),
            caption=f"hub={info['hub_token']} · bw={info['bandwidth']:.1f} · "
                    f"{info['n_edges_drawn']} arêtes"
                    + (" · échelle commune aux 3" if color_range else ""),
            color_range=color_range,
        )
        if with_text:
            # The segments are the *user* turn. The chat template and system
            # prompt wrap around them and are identical in both conditions, so
            # they are not reproduced here -- saying so beats implying that
            # this is the literal input.
            plot_prompt_text(
                fig.add_subplot(gs[r_text, col]), segments or [],
                title="Texte utilisateur — template de chat non montré"
                      + ((" · contrôle inséré en rouge" if is_control
                          else " · injection en rouge") if n_inj else ""),
            )

    if diffs:
        computed = [(slot, cond, title, *aligned_abs_diff(benign, cond))
                    for slot, cond, title in diffs]
        shared = (diff_color_range(*(c[3] for c in computed))
                  if len(computed) > 1 else None)
        ref = np.asarray(benign["W"], float)
        ref = ref[~np.eye(ref.shape[0], dtype=bool)]
        for slot, cond, title, D, n_match in computed:
            roles_c = cond["roles"]
            host = np.flatnonzero(roles_c == ROLE_BENIGN)
            hh = D[np.ix_(host, host)]
            hh = hh[np.isfinite(hh) & ~np.eye(len(host), dtype=bool)]
            plot_adjacency_diff(
                fig.add_subplot(slot), D, roles_c, title=title,
                caption=(f"{n_match}/{D.shape[0]} tokens appariés · gris = "
                         f"sans équivalent bénin\n|Δ| moyen hôte×hôte "
                         f"{hh.mean() if hh.size else float('nan'):.2e} "
                         f"(poids moyen bénin {ref.mean():.2e})"
                         + (" · échelle commune" if shared else "")),
                color_range=shared,
            )

    if third == "control":
        # Signed injected - control, in the empty slot of the difference row.
        # Both prompts have the same length and host, so what is left is the
        # content of the inserted text. Kept signed: whether the attack draws
        # *more* attention than neutral text is the question, not just how
        # much the two differ.
        S, n_match = aligned_diff(control, injected, signed=True,
                                  pair_inserted=True)
        roles_i = injected["roles"]
        host = np.flatnonzero(roles_i == ROLE_BENIGN)
        ins = np.flatnonzero(roles_i == ROLE_INJECTION)

        def _mean(block):
            v = block[np.isfinite(block)]
            return float(v.mean()) if v.size else float("nan")

        hh = S[np.ix_(host, host)].copy()
        np.fill_diagonal(hh, np.nan)
        ii = S[np.ix_(ins, ins)].copy() if ins.size else np.full((0, 0), np.nan)
        np.fill_diagonal(ii, np.nan)
        cross = S[np.ix_(host, ins)] if ins.size else np.array([np.nan])
        plot_adjacency_signed_diff(
            fig.add_subplot(gs[rows.index("diff"), 0]), S, roles_i,
            title="injecté − contrôle (signé : rouge = plus sous injection)",
            caption=(f"{n_match}/{S.shape[0]} tokens appariés · insérés "
                     f"appariés par position\nmoyennes Δ : hôte×hôte "
                     f"{_mean(hh):+.2e} · hôte×inséré {_mean(cross):+.2e} · "
                     f"inséré×inséré {_mean(ii):+.2e}"),
        )

    red = ("Tokens insérés (injection, ou texte de contrôle en col. 3)"
           if third == "control" else "Tokens injectés")
    handles = [Patch(facecolor=COLOR_BENIGN, label="Tokens bénins (hôte)"),
               Patch(facecolor=COLOR_INJECTION, label=red)]
    if config.color_by == "cluster":
        handles = [Patch(facecolor=c, label=f"Cluster {i + 1}")
                   for i, c in enumerate(CLUSTER_COLORS[:3])]
    if getattr(config, "edge_color_by", "family") == "family":
        handles += [
            Line2D([], [], color=COLOR_EDGE_BB, lw=2.4, label="Arête hôte–hôte"),
            Line2D([], [], color=COLOR_EDGE_II, lw=2.4,
                   label="Arête insérée–insérée" if third == "control"
                   else "Arête injection–injection"),
            Line2D([], [], color=COLOR_EDGE_BI, lw=2.4,
                   label="Arête croisée (la coupe mesurée)"),
        ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, 0.015))

    dropped = []
    if config.drop_sink:
        dropped.append("sink")
    if config.drop_template:
        dropped.append("template/système")
    edge_note = ("toutes les arêtes" if config.top_k_edges is None
                 else f"top-{config.top_k_edges} arêtes")
    fig.suptitle(
        f"Graphe d'attention — {model_slug} — couche {layer}/{n_layers - 1}"
        + (f"\n{subtitle}" if subtitle else "") + "\n"
        f"Laplacien {config.normalization}, symétrisé, "
        f"têtes={config.head_aggregation}, {edge_note} "
        f"(opacité ∝ poids, échelle {getattr(config, 'edge_scale', 'log')}, "
        f"γ={getattr(config, 'edge_gamma', 2.5)})"
        + (f", retirés : {', '.join(dropped)}" if dropped else ""),
        fontsize=11, fontweight="bold", y=0.975,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_layer_grid(views: Dict[int, Dict], model_slug: str, config,
                      out_path: str, panel: str = "adjacency",
                      ncols: int = 6, condition: str = "") -> str:
    """One panel per layer, laid out as a grid.

    ``views`` maps a layer index to ``{"W", "roles", "tokens"}``.

    ``panel="adjacency"`` (default) draws the role-reordered adjacency of each
    layer. This is the view that scales: with the tokens grouped by role, the
    two diagonal blocks either appear or they don't, and you can read twenty
    layers at a glance to see at what depth the structure emerges.

    ``panel="graph"`` draws the circular node-link view instead. It is much
    heavier to render and the token labels are dropped -- readable up to about
    a dozen layers, then it becomes decorative.
    """
    layers = sorted(views)
    if not layers:
        raise ValueError("no layer to plot")
    ncols = max(1, min(ncols, len(layers)))
    nrows = int(np.ceil(len(layers) / ncols))

    cell = 3.4 if panel == "graph" else 2.6
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(cell * ncols, cell * nrows + 1.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, layer in zip(axes, layers):
        data = views[layer]
        W, roles, tokens = data["W"], data["roles"], data["tokens"]
        if panel == "graph":
            info = plot_node_link(ax, W, roles, tokens,
                                  title=f"L{layer}",
                                  top_k_edges=config.top_k_edges,
                                  max_label_tokens=0,
                                  color_by=config.color_by, seed=config.seed,
                                  **edge_style_from(config))
            ax.set_title(f"L{layer} · bw={info['bandwidth']:.1f}",
                         fontsize=8, fontweight="bold", pad=3)
        else:
            has_inj = bool((roles == ROLE_INJECTION).any())
            # bw goes in the title, not the xlabel: an xlabel would collide
            # with the title of the panel on the next row.
            # aspect="auto" so each panel fills its grid cell: with "equal"
            # the axes shrink inside the cell and titles drift onto the panel
            # above.
            plot_adjacency(ax, W, roles,
                           title=f"L{layer} · bw={bandwidth(W):.1f}",
                           reorder=has_inj, aspect="auto")
    for ax in axes[len(layers):]:
        ax.axis("off")

    handles = [Patch(facecolor=COLOR_BENIGN, label="Tokens bénins (hôte)"),
               Patch(facecolor=COLOR_INJECTION, label="Tokens injectés")]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9,
               frameon=True, bbox_to_anchor=(0.5, 0.002))

    suffix = f" — {condition}" if condition else ""
    note = ("adjacence réordonnée par rôle" if panel == "adjacency"
            else "graphe circulaire, ordre de lecture")
    fig.suptitle(
        f"Graphe d'attention par couche — {model_slug}{suffix}\n"
        f"{len(layers)} couches · {note} · Laplacien {config.normalization}",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out_path


def figure_head_grid(attn_layer: np.ndarray, roles: np.ndarray, config,
                     out_path: str, layer: int, model_slug: str,
                     condition: str = "", ncols: int = 8,
                     panel: str = "adjacency",
                     tokens: Optional[Sequence[str]] = None,
                     top_k_edges: Optional[int] = None,
                     top: Optional[int] = None,
                     min_score: Optional[float] = None) -> str:
    """One adjacency panel per attention head, for a single layer.

    This is the picture that matches what ``--per-head`` computes. The other
    figures draw the head-*averaged* graph, which is exactly the object the
    per-head argument exists to avoid: if two heads out of thirty-two react to
    the injection, the average shows a flat background and the picture agrees
    with it. Here each head keeps its own panel.

    All heads are filtered with the *same* node mask -- the one derived from
    the aggregated graph -- so the panels are comparable to each other and to
    the main figure. Filtering each head on its own sink would give thirty-two
    different node sets.

    Each panel is captioned with that head's own ``separation``: it turns the
    grid into a ranking you can read, rather than thirty-two textures.

    ``panel="graph"`` draws the circular node-link view per head instead.
    Be warned: at thumbnail size a 329-node head carries ~54000 edges, so the
    render is slow and the individual links are below one pixel. The
    adjacency panel is the one that stays legible at this scale;
    ``top_k_edges`` exists to make the graph variant tractable.
    """
    from .graphs import aggregate_heads, keep_mask, symmetrize
    from .metrics import cut_statistics

    sym = symmetrize(attn_layer, config.symmetrization)          # [H, T, T]
    W_agg = aggregate_heads(sym, config.head_aggregation)
    keep = keep_mask(roles, np.asarray(W_agg, dtype=np.float64),
                     config.drop_template, config.drop_sink)
    kept_roles = roles[keep]
    n_heads = sym.shape[0]

    scores = []
    for h in range(n_heads):
        W_h = np.asarray(sym[h], dtype=np.float64)[np.ix_(keep, keep)]
        sep = cut_statistics(W_h, kept_roles).get("separation", float("nan"))
        scores.append((h, W_h, sep))

    finite = [s for _, _, s in scores if np.isfinite(s)]
    best = max(finite) if finite else float("nan")
    n_total = len(scores)

    # Keep only the heads that actually show two blocks. Drawing all thirty-two
    # buries the handful that separate among a majority that does not -- the
    # same dilution the per-head argument is about, transposed to the figure.
    if min_score is not None:
        scores = [s for s in scores if np.isfinite(s[2]) and s[2] >= min_score]
    if top is not None and top > 0:
        scores = sorted(scores, key=lambda s: (-s[2] if np.isfinite(s[2])
                                               else np.inf))[:top]
        scores.sort(key=lambda s: s[0])        # back to head order for reading
    if not scores:
        plt.close("all")
        raise ValueError(
            f"aucune tête ne passe le critère (min_score={min_score}) ; "
            f"meilleure separation observée = {best:.2f}")
    n_heads = len(scores)

    kept_tokens = ([t for t, k in zip(tokens, keep) if k] if tokens is not None
                   else [f"t{i}" for i in range(int(keep.sum()))])

    ncols = max(1, min(ncols, n_heads))
    nrows = int(np.ceil(n_heads / ncols))
    cell = 3.0 if panel == "graph" else 2.3
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols,
                                                   (cell + 0.2) * nrows + 1.0))
    axes = np.atleast_1d(axes).ravel()

    for ax, (h, W_h, sep) in zip(axes, scores):
        if panel == "graph":
            plot_node_link(ax, W_h, kept_roles, kept_tokens,
                           top_k_edges=top_k_edges, max_label_tokens=0,
                           color_by=config.color_by, seed=config.seed,
                           **edge_style_from(config))
        else:
            plot_adjacency(ax, W_h, kept_roles, reorder=True, aspect="auto")
        label = f"H{h}" + (f" · sep={sep:.1f}" if np.isfinite(sep) else "")
        # The best head is what the per-head export would single out.
        is_best = np.isfinite(sep) and np.isfinite(best) and sep >= best
        ax.set_title(label, fontsize=8,
                     fontweight="bold" if is_best else "normal",
                     color=COLOR_INJECTION if is_best else "black", pad=3)
    for ax in axes[n_heads:]:
        ax.axis("off")

    suffix = f" — {condition}" if condition else ""
    what = ("graphe circulaire, ordre de lecture" if panel == "graph"
            else "adjacence réordonnée par rôle")
    cap = (f", top-{top_k_edges} arêtes" if top_k_edges else ", toutes arêtes")
    kept = (f"{n_heads} têtes sur {n_total}" if n_heads < n_total
            else f"{n_heads} têtes")
    why = ""
    if n_heads < n_total:
        why = (f" (separation ≥ {min_score})" if min_score is not None
               else f" (les {n_heads} meilleures)")
    fig.suptitle(
        f"Têtes d'attention — {model_slug} — couche {layer}{suffix}\n"
        f"{kept}{why} · {what}{cap if panel == 'graph' else ''} · "
        f"separation par tête (meilleure en rouge)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_layer_trajectory(rows: List[Dict], metric: str, out_path: str,
                          title: str = "") -> str:
    """Layer-wise trajectory of one quantity (benign vs injected)."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, sub in rows:
        layers = [r["layer"] for r in sub]
        values = [r.get(metric, np.nan) for r in sub]
        ax.plot(layers, values, marker="o", markersize=3, label=label)
    ax.set_xlabel("Couche")
    ax.set_ylabel(metric)
    ax.set_title(title or metric, fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
