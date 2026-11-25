"""Batch evaluation over BIPIA pairs.

Runs every pair in both conditions, collects per-layer metrics, and aggregates
into the numbers that go in the paper:

* **AUROC of the discovered cut** -- benign versus injected, using
  ``cut_conductance``. Label-free: the score never sees where the injection
  is, it only measures how good the best available cut is. This is the
  detection result.
* **Agreement** -- how often the discovered cut lands on the real injection
  boundary. Computable on injected prompts only, since a benign prompt has no
  boundary. This is the mechanism result.
* **Separation** -- whether the two token families hold together more than
  they talk to each other, per layer.

The two are complementary and should be reported together: AUROC says the
method works, agreement says *why* it works.
"""

from typing import Dict, List, Optional, Sequence

import numpy as np

from .bipia import BipiaPair
from .metrics import (
    METRIC_NAMES, metrics_table, per_head_available, per_head_layer_metrics,
)


def per_head_stack(attentions: Sequence[np.ndarray], config,
                   layers: Optional[Sequence[int]] = None
                   ) -> Optional[np.ndarray]:
    """Per-head metrics for one prompt: ``[n_layers, n_heads, 5]``, or None.

    ``token_span`` is deliberately left at None, i.e. the whole sequence. The
    single-prompt CLI restricts it to the injection span, which is fine for a
    qualitative look but would be label leakage here: the benign condition has
    no injection span, so the two conditions would be measured on different
    windows and the resulting AUROC would be meaningless.
    """
    wanted = list(layers) if layers is not None else range(len(attentions))
    out = []
    for layer in wanted:
        values = per_head_layer_metrics(attentions[layer], config,
                                        token_span=None)
        if values is None:            # spectral-trust unavailable
            return None
        out.append(values)
    return np.stack(out) if out else None


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC (Mann-Whitney U), ties averaged.

    Returns NaN if either class is empty or all scores are NaN.
    """
    y = np.asarray(labels, dtype=float)
    s = np.asarray(scores, dtype=float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    # Average ranks within ties so the score is not sensitive to sort order.
    s_sorted = s[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1

    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
                 / (n_pos * n_neg))


def directed_auroc(labels, scores) -> Dict[str, float]:
    """AUROC plus the direction that makes it a detector.

    ``direction = -1`` means low scores indicate injection, which is what we
    expect for conductance: a sharper cut is a smaller number.
    """
    a = auroc(labels, scores)
    if not np.isfinite(a):
        return {"auroc": float("nan"), "direction": 0.0}
    if a >= 0.5:
        return {"auroc": a, "direction": 1.0}
    return {"auroc": 1.0 - a, "direction": -1.0}


def evaluate_pairs(runner, pairs: List[BipiaPair], config,
                   layers: Optional[Sequence[int]] = None,
                   verbose: bool = True, return_per_head: bool = False,
                   figure_hook=None):
    """Run every pair in both conditions; one row per (pair, condition, layer).

    Failures on a single pair are recorded and skipped rather than aborting a
    long run -- but they are counted and reported, never swallowed silently.

    With ``return_per_head=True`` the return value becomes
    ``(rows, per_head)``, where ``per_head`` is a dict holding

        values   float32 [n_observations, n_layers, n_heads, 5]
        labels   int [n_observations]   1 = injected
        index    one metadata dict per observation
        layers   the layer indices the second axis refers to
        metric_names, n_heads

    or None when spectral-trust is unavailable. One observation is one
    (pair, condition); the forward passes are shared with the aggregate
    metrics, so collecting this costs an eigendecomposition, not another run
    of the model.

    ``figure_hook(index, pair, lab_b, attn_b, lab_i, attn_i, table_b,
    table_i)`` is called once per successful pair, with the attentions and the
    per-layer metrics already in hand. Plotting from here rather than
    afterwards is what keeps per-pair figures free of a second forward pass --
    at ~13 s a pair, re-running the model just to draw would double the cost
    of a campaign -- and it lets the hook pick layers by what the metrics say
    rather than by index.
    """
    rows: List[Dict] = []
    ph_values: List[np.ndarray] = []
    ph_index: List[Dict] = []
    per_head_failed = False
    failures = 0

    for idx, pair in enumerate(pairs):
        try:
            lab_b, attn_b = runner.analyse_segments(pair.benign_segments)
            lab_i, attn_i = runner.analyse_segments(pair.injected_segments)
        except Exception as exc:                      # noqa: BLE001
            failures += 1
            if verbose:
                print(f"  [{idx + 1}/{len(pairs)}] échec: {exc}")
            continue

        table_b = metrics_table(attn_b, lab_b.roles, config)
        table_i = metrics_table(attn_i, lab_i.roles, config)
        wanted = layers if layers is not None else range(len(table_b))

        for layer in wanted:
            for condition, table, labelled in (("benign", table_b, lab_b),
                                               ("injected", table_i, lab_i)):
                row = dict(table[layer])
                row.update(pair.meta())
                row["condition"] = condition
                row["label"] = 1 if condition == "injected" else 0
                row["n_tokens"] = len(labelled)
                rows.append(row)

        if return_per_head and not per_head_failed:
            for condition, attns, labelled in (("benign", attn_b, lab_b),
                                               ("injected", attn_i, lab_i)):
                stack = per_head_stack(attns, config, layers)
                if stack is None:
                    per_head_failed = True
                    if verbose:
                        reason = per_head_available() or "raison inconnue"
                        print(f"  [per-head] collecte abandonnée — {reason}")
                        print("  [per-head] activez le venv du projet : "
                              ".\\spectral_injection\\env\\Scripts\\"
                              "Activate.ps1")
                    break
                ph_values.append(stack.astype(np.float32))
                meta = dict(pair.meta())
                meta["condition"] = condition
                meta["label"] = 1 if condition == "injected" else 0
                meta["n_tokens"] = len(labelled)
                ph_index.append(meta)

        if figure_hook is not None:
            try:
                # table_i is handed over so the hook can choose which layers
                # to draw from the cut statistics rather than by index.
                figure_hook(idx, pair, lab_b, attn_b, lab_i, attn_i,
                            table_b, table_i)
            except Exception as exc:                  # noqa: BLE001
                # A broken plot must not throw away a run that may have taken
                # half an hour of forward passes -- but it must not vanish
                # either: this goes to stderr whatever --quiet says, or a
                # missing figure looks exactly like a figure not requested.
                import sys as _sys
                print(f"  [figure] paire {idx + 1} ignorée: "
                      f"{type(exc).__name__}: {exc}", file=_sys.stderr)

        if verbose:
            span = lab_i.count("injection")
            print(f"  [{idx + 1}/{len(pairs)}] {pair.position:<6} "
                  f"{pair.attack_category[:24]:<24} "
                  f"bénin={len(lab_b)}tok injecté={len(lab_i)}tok "
                  f"(rouge={span})")

    if failures and verbose:
        print(f"  {failures} paire(s) en échec sur {len(pairs)}")

    if not return_per_head:
        return rows

    per_head = None
    if ph_values and not per_head_failed:
        values = np.stack(ph_values)
        n_layers = values.shape[1]
        per_head = {
            "values": values,
            "labels": np.array([m["label"] for m in ph_index], dtype=int),
            "index": ph_index,
            "layers": (list(layers) if layers is not None
                       else list(range(n_layers))),
            "metric_names": list(METRIC_NAMES),
            "n_heads": int(values.shape[2]),
        }
    return rows, per_head


def summarise_per_head(per_head: Dict, top: int = 15) -> List[Dict]:
    """AUROC of every (layer, head, metric) triple, best first.

    This is the question head-averaging cannot answer: not "does the spectrum
    shift?" but "which heads shift?". A single head with a high AUROC is the
    routing anomaly the averaged graph dilutes into its background -- and it
    is also the feature a downstream probe would actually be given.

    Reported AUROC is direction-corrected, so 0.5 means uninformative and the
    ``direction`` column says which way the metric moves under injection.
    """
    values = per_head["values"]                     # [obs, layer, head, metric]
    labels = per_head["labels"]
    out: List[Dict] = []
    for li, layer in enumerate(per_head["layers"]):
        for head in range(per_head["n_heads"]):
            for mi, metric in enumerate(per_head["metric_names"]):
                res = directed_auroc(labels, values[:, li, head, mi])
                out.append({
                    "layer": int(layer), "head": int(head), "metric": metric,
                    "auroc": res["auroc"], "direction": res["direction"],
                })
    out.sort(key=lambda d: np.nan_to_num(d["auroc"], nan=0.0), reverse=True)
    return out[:top] if top else out


def summarise_by_layer(rows: List[Dict],
                       score_keys: Sequence[str] = ("cut_conductance",
                                                    "cut_modularity",
                                                    "fiedler_value",
                                                    "spectral_entropy_norm")
                       ) -> List[Dict]:
    """Per-layer detection AUROC and mechanism statistics."""
    layers = sorted({r["layer"] for r in rows})
    out = []
    for layer in layers:
        sub = [r for r in rows if r["layer"] == layer]
        inj = [r for r in sub if r["label"] == 1]
        entry = {"layer": layer, "n_benign": sum(1 for r in sub if r["label"] == 0),
                 "n_injected": len(inj)}

        for key in score_keys:
            res = directed_auroc([r["label"] for r in sub],
                                 [r.get(key, np.nan) for r in sub])
            entry[f"auroc_{key}"] = res["auroc"]
            entry[f"dir_{key}"] = res["direction"]

        for key in ("separation", "conductance", "modularity", "accuracy", "ari"):
            values = [r[key] for r in inj if np.isfinite(r.get(key, np.nan))]
            entry[f"mean_{key}"] = float(np.mean(values)) if values else float("nan")
        out.append(entry)
    return out


def summarise_by_position(rows: List[Dict], layer: int) -> List[Dict]:
    """Does the discovered cut follow the injection when it moves?

    A stable agreement across start / middle / end is evidence that the cut
    tracks the injection rather than a fixed region of the sequence -- the
    difference between a causal claim and a positional artefact.
    """
    sub = [r for r in rows if r["layer"] == layer and r["label"] == 1]
    out = []
    for position in sorted({r["position"] for r in sub}):
        part = [r for r in sub if r["position"] == position]
        out.append({
            "position": position,
            "n": len(part),
            "mean_accuracy": _safe_mean(part, "accuracy"),
            "mean_ari": _safe_mean(part, "ari"),
            "mean_separation": _safe_mean(part, "separation"),
            "mean_conductance": _safe_mean(part, "conductance"),
        })
    return out


def summarise_by_category(rows: List[Dict], layer: int, top: int = 10) -> List[Dict]:
    """Agreement broken down by attack category, worst first.

    Useful to spot categories where the structure does not appear -- typically
    the ones whose instruction reads like ordinary host text.
    """
    sub = [r for r in rows if r["layer"] == layer and r["label"] == 1]
    out = []
    for category in sorted({r["attack_category"] for r in sub}):
        part = [r for r in sub if r["attack_category"] == category]
        out.append({
            "attack_category": category,
            "n": len(part),
            "mean_accuracy": _safe_mean(part, "accuracy"),
            "mean_separation": _safe_mean(part, "separation"),
        })
    out.sort(key=lambda d: (np.nan_to_num(d["mean_accuracy"], nan=1.0)))
    return out[:top]


def best_layer(summary: List[Dict], key: str = "auroc_cut_conductance") -> Dict:
    """Layer with the highest AUROC on one score.

    Selecting a layer on the same rows used to report the number is selection
    on the evaluation set. For the paper, pick the layer on the BIPIA *train*
    split (whose attack categories are disjoint) and report it on *test*.
    """
    finite = [s for s in summary if np.isfinite(s.get(key, np.nan))]
    if not finite:
        return {}
    return max(finite, key=lambda s: s[key])


def _safe_mean(rows: List[Dict], key: str) -> float:
    values = [r[key] for r in rows if np.isfinite(r.get(key, np.nan))]
    return float(np.mean(values)) if values else float("nan")
