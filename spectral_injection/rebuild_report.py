"""Rebuild ``rapport.tex`` from the CSVs of a finished run.

    python -m spectral_injection.rebuild_report results/bipia/<modèle>/<split>

A campaign costs tens of minutes of forward passes; the report is a few
milliseconds of formatting. Anything that already lives in the CSVs must be
re-renderable without touching the model -- otherwise improving the report
means paying for the run again, and runs predating the report module can never
get one at all.
"""

import argparse
import csv
import os
import sys
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .config import RunConfig
from .evaluate import summarise_by_category, summarise_by_position
from .report import write_report

_INT = {"layer", "label", "n_benign", "n_injection", "n_nodes", "n_tokens",
        "example_id", "head"}


def _typed(row: dict) -> dict:
    """CSV cells are strings; the report compares them numerically."""
    out = {}
    for key, value in row.items():
        if value is None or value == "":
            out[key] = float("nan")
            continue
        try:
            out[key] = int(value) if key in _INT else float(value)
        except ValueError:
            out[key] = value
    return out


def _read(path: str):
    with open(path, newline="", encoding="utf-8") as handle:
        return [_typed(r) for r in csv.DictReader(handle)]


def rebuild(out_dir: str, model: Optional[str] = None) -> str:
    rows = _read(os.path.join(out_dir, "per_pair_metrics.csv"))
    summary = _read(os.path.join(out_dir, "summary_by_layer.csv"))
    if not rows or not summary:
        raise SystemExit(f"CSV vides ou absents dans {out_dir}")

    split = os.path.basename(os.path.normpath(out_dir))
    slug = os.path.basename(os.path.dirname(os.path.normpath(out_dir)))
    config = RunConfig(model=model or slug)

    n_layers = len(summary)
    layer = n_layers // 2
    n_pairs = len({r["example_id"] for r in rows})

    per_head, ranking = None, None
    npy = os.path.join(out_dir, "per_head_values.npy")
    auroc_csv = os.path.join(out_dir, "per_head_auroc.csv")
    if os.path.exists(npy) and os.path.getsize(npy) > 1024:
        values = np.load(npy)
        index = _read(os.path.join(out_dir, "per_head_index.csv"))
        per_head = {
            "values": values,
            "labels": np.array([r["label"] for r in index], dtype=int),
            "index": index,
            "layers": list(range(values.shape[1])),
            "metric_names": ["fiedler_value", "connectivity_ratio",
                             "spectral_entropy_norm", "hfer",
                             "spectral_radius"],
            "n_heads": int(values.shape[2]),
        }
    if os.path.exists(auroc_csv):
        ranking = _read(auroc_csv)

    args = SimpleNamespace(split=split, per_head=per_head is not None)
    figures = []
    fig_dir = os.path.join(out_dir, "figures")
    if os.path.isdir(fig_dir):
        figures = sorted(f for f in os.listdir(fig_dir) if f.endswith(".png"))

    return write_report(
        os.path.join(out_dir, "rapport.tex"), config, args, rows, summary,
        layer, n_layers, range(n_pairs),
        summarise_by_position(rows, layer), summarise_by_category(rows, layer),
        per_head=per_head, head_ranking=ranking, figures=figures)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Régénère rapport.tex depuis les CSV d'un run terminé.")
    p.add_argument("out_dir", help="dossier contenant per_pair_metrics.csv")
    p.add_argument("--model", default=None,
                   help="alias ou identifiant HF, si le nom du dossier ne "
                        "suffit pas")
    args = p.parse_args(argv)
    path = rebuild(args.out_dir, args.model)
    print(f"rapport régénéré -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
