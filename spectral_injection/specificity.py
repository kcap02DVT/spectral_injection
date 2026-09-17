"""What is specific to an injection, head by head and layer by layer.

Every pair is run in three conditions that share the same email, question and
insertion slot:

    benign     the email alone
    injected   the attack inserted
    control    text from another email inserted, same token count, same slot
               (``controls.EmailTextControl``)

``injected - control`` removes what any insertion of that length does and
leaves what is due to the inserted text being an injection. The pipeline
measures that difference for every (layer, head), tests it across pairs, and
corrects for the hundreds of cells tested at once.

Attention measures -- on the *raw* causal attention ``A[i, j]`` (row i attends
to earlier token j, rows sum to 1), not on the symmetrised graph, so the
direction is known:

    cohesion          mean over inserted tokens (the first excluded: it has
                      no inserted predecessor) of the attention mass they put
                      on the inserted span. "The attack is bound to itself."
    salience_post     mean over the non-inserted tokens that come *after* the
                      span of the mass they put on it. "The rest of the prompt
                      looks at the attack."
    salience_readout  the same, restricted to the question and the chat
                      template closing the prompt -- the positions from which
                      the answer is produced.

Length is removed at two levels.

design    the control has exactly the attack's inserted token count AND the
          same total prompt length (a retokenised junction would otherwise
          shift the question/answer positions by one), and among such
          candidates the closest character count; a pair failing the token
          conditions is excluded.
analysis  the residual character and word gaps enter a Freedman-Lane model
          with the attack's token count as nuisance covariate, so ``*_adj``
          statistics are the injected - control effect at zero length gap.
          The contrasts against benign are not adjusted: they are length
          confounded by construction, which is what they are there to show.

Spectral measure -- ``fiedler_value`` (and the other four spectral-trust
metrics, stored) per head and on the head-aggregated graph, in all three
conditions, so ``injected - benign`` can be split into ``control - benign``
(insertion) and ``injected - control`` (content).

Statistics, for one family = one measure x one contrast x one level (heads or
layers), over N pairs with d = injected - control:

    mean, median, dz = mean / sd, n_pos / n_neg, sign test, Wilcoxon,
    p_perm   per-cell sign-flip permutation p (two-sided, on t)
    p_fwer   max-|t| sign-flip permutation: flipping a whole pair at once
             keeps the correlation between cells, and taking the maximum over
             the family controls the family-wise error rate
    q_bh     Benjamini-Hochberg over the family

Selecting the best cells on one split and reporting them on the same split is
circular; ``confirm`` tests cells chosen on train on a fresh split, with Holm
correction over the few cells carried over.
"""

import csv
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .metrics import METRIC_NAMES
from .prompts import ROLE_BENIGN, ROLE_INJECTION, build_labelled_prompt_from_segments

ATTN_MEASURES = ["cohesion", "salience_post", "salience_readout"]
CONDITIONS = ["benign", "injected", "control"]
CONTRASTS = {                       # name -> (condition a, condition b): a - b
    "injected-control": ("injected", "control"),
    "injected-benign": ("injected", "benign"),
    "control-benign": ("control", "benign"),
}


# --------------------------------------------------------------------------
# Measures for one prompt
# --------------------------------------------------------------------------
def readout_start(tokenizer, segments: Sequence[Tuple[str, str]],
                  system_prompt: str) -> int:
    """Index of the first token of the question segment (the last segment).

    Found by re-labelling rather than by string search: the question segment
    is marked as the only ``injection`` span and the same labeller that
    produced the real roles locates it, so both use identical token offsets.
    """
    relabelled = [(t, ROLE_BENIGN) for t, _ in segments]
    relabelled[-1] = (segments[-1][0], ROLE_INJECTION)
    roles = build_labelled_prompt_from_segments(
        tokenizer, relabelled, system_prompt).roles
    idx = np.flatnonzero(roles == ROLE_INJECTION)
    if not idx.size:
        raise RuntimeError("segment de question introuvable dans le prompt")
    return int(idx[0])


def attention_measures(attentions: Sequence[np.ndarray], roles: np.ndarray,
                       readout: int, layers: Sequence[int]) -> np.ndarray:
    """``[n_layers, n_heads, 3]`` -- cohesion, salience_post, salience_readout."""
    ins = np.flatnonzero(roles == ROLE_INJECTION)
    if ins.size < 2:
        raise ValueError("moins de deux tokens insérés")
    after = np.arange(ins.max() + 1, len(roles))
    after = after[roles[after] != ROLE_INJECTION]
    read = np.arange(max(readout, ins.max() + 1), len(roles))
    out = []
    for layer in layers:
        a = np.asarray(attentions[layer], dtype=np.float64)       # [H, T, T]
        coh = a[:, ins[1:]][:, :, ins].sum(-1).mean(-1)
        post = (a[:, after][:, :, ins].sum(-1).mean(-1) if after.size
                else np.full(a.shape[0], np.nan))
        rd = (a[:, read][:, :, ins].sum(-1).mean(-1) if read.size
              else np.full(a.shape[0], np.nan))
        out.append(np.stack([coh, post, rd], axis=-1))
    return np.stack(out)


def spectral_measures(attentions: Sequence[np.ndarray], labelled, config,
                      layers: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Per head ``[n_layers, n_heads, 5]`` and aggregated ``[n_layers, 5]``."""
    from .evaluate import per_head_stack
    from .graphs import build_graph
    from .metrics import layer_metrics

    heads = per_head_stack(attentions, config, layers)
    if heads is None:
        raise RuntimeError("spectral-trust indisponible pour les métriques par tête")
    agg = []
    for layer in layers:
        g = build_graph(attentions[layer], labelled.roles,
                        symmetrization=config.symmetrization,
                        head_aggregation=config.head_aggregation,
                        drop_template=config.drop_template,
                        drop_sink=config.drop_sink)
        m = layer_metrics(np.asarray(g["W"], float), config.normalization,
                          config.remove_self_loops)
        agg.append([m[k] for k in METRIC_NAMES])
    return heads, np.asarray(agg, dtype=float)


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def _t_stats(D: np.ndarray) -> np.ndarray:
    n = D.shape[0]
    mean = D.mean(0)
    sd = D.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(sd > 0, mean / (sd / np.sqrt(n)), 0.0)


def signflip(D: np.ndarray, n_perm: int, seed: int
             ) -> Tuple[np.ndarray, np.ndarray]:
    """Per-cell and max-|t| (family-wise) sign-flip permutation p-values.

    ``D`` is ``[n_pairs, n_cells]`` of paired differences. Under the null of
    no systematic difference each pair's sign is exchangeable; flipping whole
    rows keeps the dependence between cells, which is what makes the maximum
    over cells a valid family-wise threshold.
    """
    n, _ = D.shape
    rng = np.random.default_rng(seed)
    t_obs = np.abs(_t_stats(D))
    ss = (D ** 2).sum(0)
    count_cell = np.zeros(D.shape[1])
    count_max = np.zeros(D.shape[1])
    done = 0
    while done < n_perm:                      # chunks keep memory bounded
        b = min(1000, n_perm - done)
        flips = rng.choice([-1.0, 1.0], size=(b, n))
        mean = flips @ D / n                                   # [b, cells]
        var = (ss[None, :] - n * mean ** 2) / (n - 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(var > 0, np.abs(mean) / np.sqrt(var / n), 0.0)
        count_cell += (t >= t_obs[None, :] - 1e-12).sum(0)
        count_max += (t.max(1)[:, None] >= t_obs[None, :] - 1e-12).sum(0)
        done += b
    return (count_cell + 1) / (n_perm + 1), (count_max + 1) / (n_perm + 1)


def signflip_adjusted(D: np.ndarray, Z_zero: np.ndarray, Z_center: np.ndarray,
                      n_perm: int, seed: int) -> Dict[str, np.ndarray]:
    """Effect at zero residual length gap: Freedman-Lane sign-flip test.

    Model, per cell: ``d = b0 + Z_zero . g + Z_center . h + e``.

    ``Z_zero`` holds the residual gaps between attack and control (characters,
    words), left uncentred, so ``b0`` is the injected - control difference
    *extrapolated to no gap at all* -- not the mean difference. ``Z_center``
    holds nuisance covariates whose level is not of interest (the attack's
    token count, which varies across pairs), centred so they only absorb
    variance.

    Freedman-Lane: fit the covariates-only model, sign-flip its residuals, add
    the fitted part back, refit the full model, record t(b0). Flipping whole
    pairs keeps the dependence between cells, so the max over cells again
    gives a family-wise p-value.

    Covariates with no variance are dropped (a gap that is always zero needs
    no adjustment); if nothing is left the ordinary test applies.
    """
    n, c = D.shape
    Zz =[Z_zero[:, j] for j in range(Z_zero.shape[1])] if Z_zero.size else []
    Zc = [Z_center[:, j] - Z_center[:, j].mean()
          for j in range(Z_center.shape[1])] if Z_center.size else []
    cols = [z for z in Zz + Zc if np.std(z) > 1e-12]
    Z = np.column_stack(cols) if cols else np.zeros((n, 0))
    X = np.column_stack([np.ones(n), Z])
    dof = n - X.shape[1]
    if dof < 3:
        raise ValueError("trop de covariables pour le nombre de paires")
    XtX_inv = np.linalg.pinv(X.T @ X)
    H = XtX_inv @ X.T                                         # [p, n]
    if Z.shape[1]:
        fitted = Z @ (np.linalg.pinv(Z) @ D)                  # [n, c]
    else:
        fitted = np.zeros_like(D)
    resid = D - fitted

    def t_b0(Ds):                                             # [..., n, c]
        b = np.einsum("pn,...nc->...pc", H, Ds)
        e = Ds - np.einsum("np,...pc->...nc", X, b)
        s2 = (e ** 2).sum(-2) / dof
        se = np.sqrt(s2 * XtX_inv[0, 0])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(se > 0, b[..., 0, :] / se, 0.0)
        return b[..., 0, :], np.sqrt(s2), t

    b0, sd, t_obs = t_b0(D)
    t_abs = np.abs(t_obs)
    rng = np.random.default_rng(seed)
    count_cell = np.zeros(c)
    count_max = np.zeros(c)
    chunk = max(1, int(2e7 // max(n * c, 1)))
    done = 0
    while done < n_perm:
        b = min(chunk, n_perm - done)
        flips = rng.choice([-1.0, 1.0], size=(b, n))
        Ds = fitted[None] + flips[:, :, None] * resid[None]
        _, _, t = t_b0(Ds)
        t = np.abs(t)
        count_cell += (t >= t_abs[None] - 1e-12).sum(0)
        count_max += (t.max(1)[:, None] >= t_abs[None] - 1e-12).sum(0)
        done += b
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = np.where(sd > 0, b0 / sd, 0.0)
    return {"mean_adj": b0, "dz_adj": dz,
            "p_perm_adj": (count_cell + 1) / (n_perm + 1),
            "p_fwer_adj": (count_max + 1) / (n_perm + 1),
            "covariates_used": np.full(c, Z.shape[1])}


def open_ending(text: str) -> int:
    """1 if the text stops without sentence-final punctuation, else 0.

    Closing quotes, brackets and parentheses are stripped first, so a text is
    judged on what they enclose. Measured on the text itself rather than on
    whether the control was truncated: attacks end open too, and what the
    model reads is the form, not how the text was obtained.
    """
    t = text.rstrip().rstrip("\"'”’)]}»").rstrip()
    return int(not t.endswith((".", "!", "?", "…")))


def length_covariates(pairs_csv: str) -> Optional[Dict[str, np.ndarray]]:
    """Residual length gaps and nuisance lengths per pair, from pairs.csv.

    Recomputed from the stored texts when the count columns are absent, so a
    run made before they existed can still be adjusted.

    ``open_gap`` = open_ending(attack) - open_ending(control), in {-1, 0, 1}:
    matching token and character counts exactly often means cutting the
    control mid-sentence, and an abrupt ending is its own anomaly. It is not
    a length but a by-product of removing length, so it is adjusted the same
    way -- at zero gap.
    """
    if not os.path.exists(pairs_csv):
        return None
    rows = read_rows(pairs_csv)
    if not rows or "attack" not in rows[0]:
        return None

    def num(r, key, fallback):
        v = r.get(key)
        return float(v) if v not in (None, "") else float(fallback)

    ca = np.array([num(r, "chars_attack", len(r["attack"])) for r in rows])
    cc = np.array([num(r, "chars_control", len(r["control"])) for r in rows])
    wa = np.array([num(r, "words_attack", len(r["attack"].split())) for r in rows])
    wc = np.array([num(r, "words_control", len(r["control"].split())) for r in rows])
    tok = np.array([num(r, "tokens_inserted", 0) for r in rows])
    oa = np.array([open_ending(r["attack"]) for r in rows], dtype=float)
    oc = np.array([open_ending(r["control"]) for r in rows], dtype=float)
    return {"chars_gap": ca - cc, "words_gap": wa - wc, "open_gap": oa - oc,
            "tokens_inserted": tok,
            "chars_attack": ca, "chars_control": cc,
            "words_attack": wa, "words_control": wc,
            "open_attack": oa, "open_control": oc}


def balance_report(cov: Dict[str, np.ndarray]) -> List[str]:
    """Human-readable residual length balance between attack and control."""
    lines = []
    ca, cc = cov["chars_attack"], cov["chars_control"]
    ratio = cc / np.maximum(ca, 1)
    lines.append(f"tokens insérés : identiques par construction "
                 f"(médiane {np.median(cov['tokens_inserted']):.0f})")
    lines.append(f"caractères : attaque {np.median(ca):.0f} · contrôle "
                 f"{np.median(cc):.0f} (médianes) · écart médian "
                 f"{np.median(np.abs(ca - cc)):.0f} · contrôle/attaque "
                 f"{np.median(ratio):.2f} [{np.percentile(ratio, 25):.2f}–"
                 f"{np.percentile(ratio, 75):.2f}] · à ±10 % : "
                 f"{int((np.abs(ratio - 1) <= 0.10).sum())}/{len(ca)}")
    lines.append(f"mots : attaque {np.median(cov['words_attack']):.0f} · "
                 f"contrôle {np.median(cov['words_control']):.0f} (médianes) · "
                 f"écart médian {np.median(np.abs(cov['words_gap'])):.0f}")
    lines.append(f"fin abrupte (sans ponctuation finale) : attaque "
                 f"{int(cov['open_attack'].sum())}/{len(ca)} · contrôle "
                 f"{int(cov['open_control'].sum())}/{len(ca)} · paires "
                 f"discordantes {int((cov['open_gap'] != 0).sum())}")
    return lines


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if not ok.any():
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.maximum.accumulate(p[order] * (m - np.arange(m)))
    out = np.empty_like(p)
    out[order] = np.minimum(adj, 1.0)
    return out


def family_stats(D: np.ndarray, n_perm: int, seed: int) -> Dict[str, np.ndarray]:
    """All statistics for one family ``D [n_pairs, n_cells]`` (NaN-free)."""
    from scipy.stats import binomtest, wilcoxon

    n = D.shape[0]
    mean, median = D.mean(0), np.median(D, 0)
    sd = D.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        dz = np.where(sd > 0, mean / sd, 0.0)
    npos, nneg = (D > 0).sum(0), (D < 0).sum(0)
    p_sign = np.array([binomtest(int(a), int(a + b)).pvalue if a + b else 1.0
                       for a, b in zip(npos, nneg)])
    p_wil = np.empty(D.shape[1])
    for c in range(D.shape[1]):
        col = D[:, c]
        p_wil[c] = (wilcoxon(col).pvalue if np.any(col != 0) else 1.0)
    p_perm, p_fwer = signflip(D, n_perm, seed)
    return {"n": np.full(D.shape[1], n), "mean": mean, "median": median,
            "sd": sd, "dz": dz, "n_pos": npos, "n_neg": nneg,
            "p_sign": p_sign, "p_wilcoxon": p_wil, "p_perm": p_perm,
            "p_fwer": p_fwer, "q_bh": bh(p_perm)}


# --------------------------------------------------------------------------
# Campaign container
# --------------------------------------------------------------------------
class SpecificityData:
    """Per-pair arrays, filled one pair at a time.

    attn_heads  [pairs, 2, L, H, 3]    conditions (injected, control)
    spec_heads  [pairs, 3, L, H, 5]    conditions (benign, injected, control)
    spec_layers [pairs, 3, L, 5]
    """

    def __init__(self, layers: Sequence[int]):
        self.layers = list(layers)
        self.attn: List[np.ndarray] = []
        self.spec_heads: List[np.ndarray] = []
        self.spec_layers: List[np.ndarray] = []
        self.meta: List[Dict] = []

    def add(self, attn_i, attn_c, spec=None, meta=None):
        self.attn.append(np.stack([attn_i, attn_c]))
        if spec is not None:
            self.spec_heads.append(np.stack([s[0] for s in spec]))
            self.spec_layers.append(np.stack([s[1] for s in spec]))
        self.meta.append(meta or {})

    @property
    def n(self) -> int:
        return len(self.attn)

    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        arrays = {"layers": np.asarray(self.layers),
                  "attn_heads": np.asarray(self.attn, dtype=np.float32)}
        if self.spec_heads:
            arrays["spec_heads"] = np.asarray(self.spec_heads, dtype=np.float32)
            arrays["spec_layers"] = np.asarray(self.spec_layers, dtype=np.float32)
        np.savez_compressed(os.path.join(out_dir, "values.npz"), **arrays)
        if self.meta:
            keys = list(self.meta[0].keys())
            with open(os.path.join(out_dir, "pairs.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(self.meta)


ADJ_KEYS = ["mean_adj", "dz_adj", "p_perm_adj", "p_fwer_adj", "covariates_used"]


def analyse(npz_path: str, n_perm: int = 5000, seed: int = 0,
            adjust: bool = True) -> List[Dict]:
    """Every family, every cell: one row per (level, measure, contrast, cell).

    ``injected-control`` families also get length-adjusted statistics
    (``*_adj``, see ``signflip_adjusted``) when pairs.csv is next to the npz:
    residual character and word gaps at zero, attack token count as nuisance.
    The contrasts against benign are left unadjusted -- they are confounded
    with length by construction, which is exactly what they are there to show.
    """
    data = np.load(npz_path)
    layers = data["layers"]
    rows: List[Dict] = []
    cov = (length_covariates(os.path.join(os.path.dirname(npz_path), "pairs.csv"))
           if adjust else None)
    if cov is not None and len(cov["chars_gap"]) != data["attn_heads"].shape[0]:
        cov = None                       # pairs.csv out of step with the npz
    warned: List[bool] = []

    def emit(level, measure, contrast, D, cells):
        keep = np.all(np.isfinite(D), axis=1)
        D = D[keep]
        if D.shape[0] < 5:
            return
        st = family_stats(D, n_perm, seed)
        adjusted = None
        if cov is not None and contrast == "injected-control":
            Zz = np.column_stack([cov["chars_gap"], cov["words_gap"],
                                  cov["open_gap"]])[keep]
            Zc = cov["tokens_inserted"][keep][:, None]
            try:
                adjusted = signflip_adjusted(D, Zz, Zc, n_perm, seed)
            except ValueError as exc:
                # Too few pairs for the covariates: report unadjusted rather
                # than fail the whole analysis, and say so once.
                if not warned:
                    print(f"  [ajustement longueur omis : {exc} "
                          f"({D.shape[0]} paires)]")
                    warned.append(True)
        st.update(adjusted if adjusted is not None else
                  {k: np.full(D.shape[1], np.nan) for k in ADJ_KEYS})
        for c, (lay, head) in enumerate(cells):
            row = {"level": level, "measure": measure, "contrast": contrast,
                   "layer": int(lay), "head": head}
            row.update({k: float(v[c]) for k, v in st.items()})
            rows.append(row)

    A = data["attn_heads"]                                  # [N, 2, L, H, 3]
    N, _, L, H, _ = A.shape
    head_cells = [(layers[l], h) for l in range(L) for h in range(H)]
    layer_cells = [(layers[l], "moy") for l in range(L)]
    for m, name in enumerate(ATTN_MEASURES):
        d = A[:, 0, :, :, m] - A[:, 1, :, :, m]              # injected - control
        emit("head", name, "injected-control", d.reshape(N, -1), head_cells)
        emit("layer", name, "injected-control", d.mean(2), layer_cells)

    if "spec_heads" in data:
        S, SL = data["spec_heads"], data["spec_layers"]
        for m, name in enumerate(METRIC_NAMES):
            for contrast, (a, b) in CONTRASTS.items():
                ia, ib = CONDITIONS.index(a), CONDITIONS.index(b)
                d = S[:, ia, :, :, m] - S[:, ib, :, :, m]
                emit("head", name, contrast, d.reshape(N, -1), head_cells)
                emit("layer", name, contrast,
                     SL[:, ia, :, m] - SL[:, ib, :, m], layer_cells)
    return rows


def write_rows(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def read_rows(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def best(row: Dict, key: str) -> float:
    """The length-adjusted statistic when it exists, the plain one otherwise."""
    v = row.get(f"{key}_adj")
    try:
        v = float(v)
    except (TypeError, ValueError):
        v = float("nan")
    return v if np.isfinite(v) else float(row[key])


def select_cells(train_rows: List[Dict], top: int, alpha: float = 0.05,
                 contrast: str = "injected-control") -> List[Dict]:
    """Cells to carry to a fresh split: FWER-significant on train, top |dz|.

    Length-adjusted statistics are used whenever present. Per (level,
    measure), at most ``top`` cells. The sign observed on train is kept:
    confirmation means the same direction again, not any difference.
    """
    chosen = []
    groups: Dict[Tuple[str, str], List[Dict]] = {}
    for r in train_rows:
        if r["contrast"] != contrast or best(r, "p_fwer") >= alpha:
            continue
        groups.setdefault((r["level"], r["measure"]), []).append(r)
    for _, rs in sorted(groups.items()):
        rs.sort(key=lambda r: -abs(best(r, "dz")))
        chosen.extend(rs[:top])
    return chosen


def confirm(test_rows: List[Dict], chosen: List[Dict]) -> List[Dict]:
    """Test-split statistics of the train-selected cells, Holm-corrected.

    ``replicated`` requires the train sign and Holm-adjusted p < 0.05 on the
    per-cell permutation test -- the family here is only the carried cells,
    which is the point of selecting on another split.
    """
    index = {(r["level"], r["measure"], r["contrast"], str(r["layer"]),
              str(r["head"])): r for r in test_rows}
    out = []
    for c in chosen:
        key = (c["level"], c["measure"], c["contrast"], str(c["layer"]),
               str(c["head"]))
        t = index.get(key)
        if t is None:
            continue
        out.append({"level": c["level"], "measure": c["measure"],
                    "contrast": c["contrast"], "layer": c["layer"],
                    "head": c["head"], "dz_train": best(c, "dz"),
                    "p_fwer_train": best(c, "p_fwer"),
                    "dz_test": best(t, "dz"), "n_test": int(float(t["n"])),
                    "n_pos_test": int(float(t["n_pos"])),
                    "n_neg_test": int(float(t["n_neg"])),
                    "p_perm_test": best(t, "p_perm"),
                    "length_adjusted": int(np.isfinite(
                        float(t.get("p_perm_adj") or "nan")))})
    if out:
        adj = holm(np.array([r["p_perm_test"] for r in out]))
        for r, a in zip(out, adj):
            r["p_holm_test"] = float(a)
            r["replicated"] = int(a < 0.05 and np.sign(r["dz_test"])
                                  == np.sign(r["dz_train"]))
    return out


# --------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------
def heatmaps(rows: List[Dict], out_path: str, model: str, split: str,
             alpha: float = 0.05) -> Optional[str]:
    """Layer x head maps of dz (injected - control), FWER-significant cells dotted.

    Top row: the three attention measures and lambda_2. Bottom row: lambda_2
    decomposed -- injected-benign, control-benign, injected-control -- which
    shows at a glance how much of the injection effect is insertion.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def grid(measure, contrast):
        rs = [r for r in rows if r["level"] == "head" and r["measure"] == measure
              and r["contrast"] == contrast]
        if not rs:
            return None
        layers = sorted({int(r["layer"]) for r in rs})
        heads = sorted({int(r["head"]) for r in rs})
        dz = np.full((len(layers), len(heads)), np.nan)
        sig = np.zeros_like(dz, dtype=bool)
        for r in rs:
            i, j = layers.index(int(r["layer"])), heads.index(int(r["head"]))
            dz[i, j] = best(r, "dz")
            sig[i, j] = best(r, "p_fwer") < alpha
        lay = {int(r["layer"]): r for r in rows if r["level"] == "layer"
               and r["measure"] == measure and r["contrast"] == contrast}
        return layers, heads, dz, sig, lay

    panels = [(m, "injected-control") for m in ATTN_MEASURES]
    panels.append(("fiedler_value", "injected-control"))
    panels += [("fiedler_value", c) for c in
               ("injected-benign", "control-benign", "injected-control")]
    panels = [(m, c, grid(m, c)) for m, c in panels]
    panels = [p for p in panels if p[2] is not None]
    if not panels:
        return None

    ncols = min(4, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 5.0 * nrows),
                             squeeze=False)
    labels = {"cohesion": "cohésion (attaque → attaque)",
              "salience_post": "saillance (suite du prompt → inséré)",
              "salience_readout": "saillance (question/réponse → inséré)",
              "fiedler_value": "λ₂ par tête"}
    for ax, (m, c, (layers, heads, dz, sig, lay)) in zip(axes.flat, panels):
        vmax = max(np.nanmax(np.abs(dz)), 1e-9)
        im = ax.imshow(dz, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto",
                       origin="lower", interpolation="nearest")
        yy, xx = np.nonzero(sig)
        ax.scatter(xx, yy, s=10, c="black", marker="o", linewidths=0)
        ax.set_xlabel("tête", fontsize=8)
        ax.set_ylabel("couche", fontsize=8)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers, fontsize=5)
        ax.tick_params(axis="x", labelsize=6)
        n_sig = int(sig.sum())
        lsig = sum(1 for r in lay.values() if best(r, "p_fwer") < alpha)
        adj = c == "injected-control" and any(
            np.isfinite(float(r.get("p_fwer_adj") or "nan")) for r in lay.values())
        ax.set_title(f"{labels.get(m, m)}\n{c}"
                     f"{' · ajusté longueur' if adj else ''} · dz · "
                     f"● FWER<{alpha} : {n_sig} têtes, {lsig} couches (moy.)",
                     fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=6)
    for ax in list(axes.flat)[len(panels):]:
        ax.axis("off")
    fig.suptitle(f"Spécificité de l'injection — {model} — split {split}\n"
                 f"dz = moyenne / écart-type des différences appariées ; "
                 f"rouge = plus grand dans la 1re condition",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
