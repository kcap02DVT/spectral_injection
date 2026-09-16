"""Run configuration for the attention-graph analysis."""

from dataclasses import dataclass, field
from typing import Optional

# Short names -> Hugging Face identifiers.
MODEL_ALIASES = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama3.2-1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
    "smollm2-1.7b": "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "phi-4-mini": "microsoft/Phi-4-mini-instruct",
    "gemma-3-4b": "google/gemma-3-4b-it",
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe. Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content."
)


@dataclass
class RunConfig:
    """Everything that changes a number, in one place.

    The graph-construction fields mirror ``spectral_trust.GSPConfig`` so that
    the per-head metrics and the local (numpy) computations are built on the
    same graph. ``normalization="sym"`` is the v0.3.0 default: the spectrum
    lies in [0, 2] independently of sequence length, which is what removes the
    length confound between a benign prompt and its longer injected variant.
    """

    model: str = "tinyllama"
    device: str = "auto"
    dtype: str = "float32"

    # --- Graph construction -------------------------------------------------
    normalization: str = "sym"          # sym | rw | none
    symmetrization: str = "symmetric"   # symmetric | row_norm | col_norm
    head_aggregation: str = "uniform"   # uniform | attention_weighted
    remove_self_loops: bool = True
    # "dense" on purpose, and it must stay that way. spectral-trust defaults to
    # "sparse", which above 50 nodes returns only ``num_eigenvalues`` modes --
    # the extremes -- and drops the middle of the spectrum. lambda_2 and the
    # spectral radius survive that, but hfer and spectral_entropy_norm are
    # ratios over the *whole* spectrum and become wrong (measured: 12% off on
    # hfer for a 329-token prompt). At these sizes the dense solver is also as
    # fast: 92 ms vs 88 ms at n=329.
    eigen_solver: str = "dense"         # dense | sparse

    # --- Node filtering -----------------------------------------------------
    # Both default to False in raw-text mode, and the reasons are different.
    #
    # drop_sink: the prompt carries no BOS any more, so the sink would be a
    # *content* token, found by argmax independently in each condition. The
    # benign and injected graphs would then lose different nodes, and the
    # comparison between them would no longer be like-for-like.
    drop_sink: bool = False
    # drop_template: nothing is labelled 'template' any longer, so this has
    # nothing to remove. Kept as a switch for result files predating raw-text
    # mode.
    drop_template: bool = False

    # --- Analysis -----------------------------------------------------------
    layer: Optional[int] = None   # None -> middle layer
    all_layers: bool = True       # compute metrics for every layer
    per_head: bool = True         # per-head metrics via spectral-trust
    max_tokens: int = 512

    # --- Output -------------------------------------------------------------
    out_dir: str = "results"
    color_by: str = "role"        # role (blue/red) | cluster (3-way spectral)
    # None -> draw every edge. The graph is dense (~n^2/2 edges), so legibility
    # comes from the alpha/width ramp below rather than from a cut-off.
    top_k_edges: Optional[int] = None
    max_label_tokens: int = 24
    seed: int = 42

    # --- Edge rendering -----------------------------------------------------
    # Alpha and width are indexed on the attention weight. Every scale below is
    # monotone -- a stronger edge is never drawn fainter than a weaker one --
    # they differ only in how the gaps are spread across the visible range.
    #
    # "log" is the default because attention spans decades: on a linear ramp
    # the attention sink saturates the top and the remaining thousands of edges
    # all land on the alpha floor, indistinguishable from each other. The
    # picture becomes a star. Log space gives each decade equal room, so the
    # hub stays dominant and the structure underneath becomes readable.
    # "linear" restores the pre-raw-text behaviour; "rank" maximises contrast
    # but discards magnitude.
    edge_scale: str = "log"          # log | linear | rank
    # edge_gamma > 1 bends the ramp downwards. With thousands of overlapping
    # lines alpha accumulates, so this is what keeps the disc from filling in.
    edge_gamma: float = 3.0
    edge_alpha_min: float = 0.003
    edge_alpha_max: float = 0.90
    edge_width_min: float = 0.10
    edge_width_max: float = 2.4
    # Total opacity laid down, summed over every drawn edge. Ink grows with the
    # edge count -- a ramp tuned on an 80-token prompt (3k edges) saturates on
    # a 185-token one (17k edges) -- so it is normalised rather than fixed.
    # None lets the raw ramp through.
    edge_ink_budget: Optional[float] = 250.0
    # How far long-range chords bow towards the centre, where there is room.
    edge_curvature: float = 0.75
    # How far short-range chords bulge *outwards*, past the ring. Links between
    # neighbouring tokens are only a few points long and sit right under their
    # own markers; arcing them outwards is what makes the local structure --
    # the diagonal band of the adjacency matrix -- visible on the circle at
    # all. 0 keeps them inside.
    #
    # These links are not a detail: the median weight between consecutive
    # tokens sits at the 95th percentile of the whole graph.
    edge_outward: float = 0.55
    # Span, in tokens, at which the outward bulge peaks. Arc height grows with
    # span below it, so spans 1, 2 and 5 nest instead of landing on the same
    # radius and merging into one fuzzy annulus. Kept small on purpose: it is
    # the first handful of spans that has to be separable, and spreading them
    # over the whole outward range is what makes an individual i -> i+1 link
    # a shape you can point at.
    edge_outward_span: float = 6.0
    # "family": hue says which families an edge joins (host-host, cross,
    # injection-injection) while intensity still says how strong it is. Cross
    # edges are the cut the analysis measures and are usually weak, so without
    # a hue of their own they are invisible among thousands of others.
    # "weight" restores the single YlOrRd ramp.
    edge_color_by: str = "family"     # family | weight
    # Blank arc inserted at every role boundary, in node-widths per 100 nodes.
    # A 9-token injection inside 176 host tokens is otherwise an unbroken run
    # of dots that cannot be picked out.
    node_gap_frac: float = 0.04
    # Percentile of the weights mapped to full intensity, and (log scale only)
    # the one mapped to zero.
    edge_ref_percentile: float = 99.5
    edge_floor_percentile: float = 2.0

    system_prompt: str = field(default=DEFAULT_SYSTEM_PROMPT)

    @property
    def model_id(self) -> str:
        return MODEL_ALIASES.get(self.model.lower(), self.model)

    @property
    def model_slug(self) -> str:
        return self.model_id.split("/")[-1]
