"""Attention-graph diagnostics for prompt-injection detection."""

from .config import MODEL_ALIASES, RunConfig
from .graphs import build_graph, fiedler, laplacian, symmetrize
from .metrics import (
    METRIC_NAMES,
    cut_statistics,
    fiedler_partition,
    layer_metrics,
    metrics_table,
    partition_agreement,
    partition_quality,
    per_head_layer_metrics,
)
from .bipia import BipiaPair, load_pairs, make_pair
from .evaluate import (
    auroc,
    best_layer,
    evaluate_pairs,
    per_head_stack,
    summarise_by_category,
    summarise_by_layer,
    summarise_by_position,
    summarise_per_head,
)
from .prompts import build_labelled_prompt, build_labelled_prompt_from_segments

__version__ = "0.1.0"

__all__ = [
    "RunConfig", "MODEL_ALIASES", "build_labelled_prompt",
    "build_labelled_prompt_from_segments", "build_graph",
    "symmetrize", "laplacian", "fiedler", "layer_metrics", "cut_statistics",
    "fiedler_partition", "partition_quality", "partition_agreement",
    "per_head_layer_metrics", "metrics_table", "METRIC_NAMES",
    "BipiaPair", "load_pairs", "make_pair", "evaluate_pairs", "auroc",
    "summarise_by_layer", "summarise_by_position", "summarise_by_category",
    "per_head_stack", "summarise_per_head", "best_layer", "__version__",
]
