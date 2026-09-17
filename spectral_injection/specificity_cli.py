"""Injection specificity, head by head and layer by layer.

    # discovery on train: every cell tested, family-wise corrected
    python -m spectral_injection.specificity_cli --model llama3.2-1b --split train \\
        --n-pairs 40 --attack-source neuralchemy --attack-max-chars 200 \\
        --out results/specificity

    # confirmation on test of the cells found on train
    python -m spectral_injection.specificity_cli --model llama3.2-1b --split test \\
        --n-pairs 40 --attack-source neuralchemy --attack-max-chars 200 \\
        --out results/specificity \\
        --confirm results/specificity/Llama-3.2-1B-Instruct/train/cells.csv

    # statistics again from saved values, no model
    python -m spectral_injection.specificity_cli --analyse-only \\
        results/specificity/Llama-3.2-1B-Instruct/train

See ``specificity.py`` for the measures and the statistics.
"""

import argparse
import os
import sys
import time

import numpy as np

from .bipia import POSITIONS, describe, load_pairs
from .config import RunConfig
from .specificity import (
    ATTN_MEASURES, SpecificityData, analyse, attention_measures,
    balance_report, best, confirm, heatmaps, length_covariates, read_rows,
    readout_start, select_cells, spectral_measures, write_rows,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ce qui est propre à l'injection (injecté − contrôle), "
                    "tête par tête et couche par couche.")
    p.add_argument("--model", default="tinyllama")
    p.add_argument("--split", default="train", choices=["train", "test"])
    p.add_argument("--n-pairs", type=int, default=40)
    p.add_argument("--positions", nargs="+", default=list(POSITIONS),
                   choices=list(POSITIONS))
    p.add_argument("--max-context-chars", type=int, default=1200)
    p.add_argument("--attack-source", default="bipia",
                   choices=["bipia", "neuralchemy"])
    p.add_argument("--attack-category", default=None)
    p.add_argument("--attack-max-chars", type=int, default=None)
    p.add_argument("--attack-severity", nargs="+", default=None,
                   choices=["low", "medium", "high", "critical"])
    p.add_argument("--control", default="email_text", choices=["email_text"])
    p.add_argument("--max-tokens", type=int, default=768)
    p.add_argument("--layers", nargs="+", type=int, default=None,
                   help="couches analysées (défaut : toutes)")
    p.add_argument("--no-spectral", action="store_true",
                   help="mesures d'attention seulement : ni λ₂ ni passe bénigne "
                        "(environ 3x plus rapide)")
    p.add_argument("--permutations", type=int, default=5000,
                   help="permutations par inversion de signe (défaut 5000)")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--confirm", default=None, metavar="CELLS_CSV",
                   help="cells.csv d'un run train : teste ces cellules ici, "
                        "correction de Holm sur les cellules reportées")
    p.add_argument("--confirm-top", type=int, default=5,
                   help="cellules reportées par (niveau, mesure) (défaut 5)")
    p.add_argument("--analyse-only", default=None, metavar="DIR",
                   help="recalculer statistiques et figure depuis DIR/values.npz")
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    p.add_argument("--normalization", default="sym", choices=["sym", "rw", "none"])
    p.add_argument("--cache-dir", default="data/bipia")
    p.add_argument("--offline", action="store_true")
    p.add_argument("--out", default="results/specificity")
    p.add_argument("--seed", type=int, default=42)
    return p


def _summary(rows, alpha: float) -> None:
    print("\n" + "=" * 78)
    print("injecté − contrôle : cellules significatives (FWER < "
          f"{alpha}, permutation max-|t|)")
    print("=" * 78)
    measures = ATTN_MEASURES + ["fiedler_value"]
    for level in ("layer", "head"):
        for m in measures:
            rs = [r for r in rows if r["level"] == level and r["measure"] == m
                  and r["contrast"] == "injected-control"]
            if not rs:
                continue
            sig = [r for r in rs if float(r["p_fwer"]) < alpha]
            adj = [r for r in rs if np.isfinite(float(r.get("p_fwer_adj") or "nan"))]
            sig_adj = [r for r in adj if float(r["p_fwer_adj"]) < alpha]
            n_up = sum(1 for r in sig_adj if float(r["dz_adj"]) > 0)
            print(f"\n[{'couches (moy. têtes)' if level == 'layer' else 'têtes'}]"
                  f" {m} : brut {len(sig)}/{len(rs)}"
                  + (f" · ajusté longueur {len(sig_adj)}/{len(rs)} "
                     f"({n_up} en hausse, {len(sig_adj) - n_up} en baisse)"
                     if adj else " (ajustement longueur indisponible)"))
            top = sorted(rs, key=lambda r: -abs(best(r, "dz")))[:5]
            for r in top:
                cell = (f"L{r['layer']}" if level == "layer"
                        else f"L{r['layer']} H{r['head']}")
                n = int(float(r["n"]))
                line = (f"    {cell:<9} {int(float(r['n_pos']))}/{n} en hausse  "
                        f"dz brut {float(r['dz']):+.2f} p_fwer "
                        f"{float(r['p_fwer']):.3f}")
                if r in adj:
                    line += (f"  | ajusté dz {float(r['dz_adj']):+.2f} p_fwer "
                             f"{float(r['p_fwer_adj']):.3f}")
                print(line)

    lay = {}
    for r in rows:
        if r["level"] == "layer" and r["measure"] == "fiedler_value":
            lay.setdefault(int(r["layer"]), {})[r["contrast"]] = r
    if lay:
        print("\nλ₂ du graphe agrégé : part de l'effet due à l'insertion")
        print(f"  {'couche':>6} {'inj−bén':>10} {'ctl−bén':>10} {'inj−ctl':>10} "
              f"{'part insertion':>15} {'dz inj−ctl':>11} {'p_fwer':>7}")
        for L in sorted(lay):
            d = lay[L]
            if not all(k in d for k in ("injected-benign", "control-benign",
                                        "injected-control")):
                continue
            ib = float(d["injected-benign"]["mean"])
            cb = float(d["control-benign"]["mean"])
            ic = d["injected-control"]
            part = cb / ib if abs(ib) > 1e-12 else float("nan")
            print(f"  {L:>6} {ib:>+10.2e} {cb:>+10.2e} "
                  f"{float(ic['mean']):>+10.2e} {part:>14.0%} "
                  f"{float(ic['dz']):>+11.2f} {float(ic['p_fwer']):>7.3f}")


def _analyse_dir(out_dir: str, args, model: str, split: str) -> list:
    npz = os.path.join(out_dir, "values.npz")
    t0 = time.time()
    rows = analyse(npz, n_perm=args.permutations, seed=args.seed)
    write_rows(rows, os.path.join(out_dir, "cells.csv"))
    fig = heatmaps(rows, os.path.join(out_dir, "specificity.png"), model, split,
                   alpha=args.alpha)
    print(f"statistiques : {len(rows)} cellules en {time.time() - t0:.0f} s "
          f"-> cells.csv" + (f", {os.path.basename(fig)}" if fig else ""))
    cov = length_covariates(os.path.join(out_dir, "pairs.csv"))
    if cov is not None:
        print("\néquilibre des longueurs attaque / contrôle")
        for line in balance_report(cov):
            print(f"  {line}")
        print("  ajustement : écarts de caractères, de mots et de fin abrupte "
              "ramenés à zéro, longueur de l'attaque en covariable "
              "(Freedman–Lane)")
    _summary(rows, args.alpha)

    if args.confirm:
        chosen = select_cells(read_rows(args.confirm), args.confirm_top,
                              args.alpha)
        result = confirm(rows, chosen)
        path = os.path.join(out_dir, "confirmation.csv")
        write_rows(result, path)
        print("\n" + "=" * 78)
        print(f"CONFIRMATION de {len(chosen)} cellules choisies sur "
              f"{args.confirm}")
        print("=" * 78)
        if not result:
            print("  aucune cellule significative sur train : rien à confirmer")
        for r in result:
            cell = (f"L{r['layer']}" if r["level"] == "layer"
                    else f"L{r['layer']} H{r['head']}")
            print(f"  {r['measure']:<17} {cell:<9} dz train {r['dz_train']:+.2f}"
                  f" -> test {r['dz_test']:+.2f} ({r['n_pos_test']}/"
                  f"{r['n_test']} en hausse) p_holm {r['p_holm_test']:.3f} "
                  f"{'RÉPLIQUÉ' if r['replicated'] else 'non répliqué'}")
        if result:
            k = sum(r["replicated"] for r in result)
            print(f"\n  {k}/{len(result)} cellules répliquées -> {path}")
    return rows


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.analyse_only:
        parts = os.path.normpath(args.analyse_only).split(os.sep)
        _analyse_dir(args.analyse_only, args, parts[-2], parts[-1])
        return 0

    config = RunConfig(model=args.model, device=args.device, dtype=args.dtype,
                       normalization=args.normalization,
                       max_tokens=args.max_tokens, seed=args.seed)
    out_dir = os.path.join(args.out, config.model_slug, args.split)
    os.makedirs(out_dir, exist_ok=True)

    if not args.no_spectral:
        from .metrics import per_head_available
        reason = per_head_available()
        if reason:
            print(f"spectral-trust indisponible ({reason}) ; relancer dans le "
                  f"venv ou passer --no-spectral", file=sys.stderr)
            return 2

    pairs = load_pairs(
        split=args.split, n_pairs=args.n_pairs, positions=args.positions,
        max_context_chars=args.max_context_chars, cache_dir=args.cache_dir,
        offline=args.offline, seed=args.seed, attack_source=args.attack_source,
        attack_category=args.attack_category,
        attack_max_chars=args.attack_max_chars,
        attack_severity=args.attack_severity)
    print(f"Spécificité | {config.model_id} | split {args.split} | "
          f"{describe(pairs)}")

    from .bipia import _load_emails, ensure_data
    from .controls import EmailTextControl, control_segments
    from .prompts import build_labelled_prompt_from_segments
    from .runner import AttentionRunner

    runner = AttentionRunner(config)
    layers = (list(args.layers) if args.layers is not None
              else list(range(runner.n_layers)))
    emails = _load_emails(ensure_data(args.cache_dir, offline=args.offline)
                          [f"email_{args.split}"])
    pool = EmailTextControl(emails, runner.tokenizer, seed=args.seed)
    data = SpecificityData(layers)
    skipped = []
    t_start = time.time()

    for idx, pair in enumerate(pairs):
        try:
            lab_i, att_i = runner.analyse_segments(pair.injected_segments)
            n_ins = lab_i.count("injection")

            def in_prompt(text, _pair=pair):
                lab = build_labelled_prompt_from_segments(
                    runner.tokenizer, control_segments(_pair, text),
                    system_prompt=config.system_prompt)
                return lab.count("injection"), len(lab)

            # Same inserted tokens AND same total prompt length (no shift of
            # the question/answer positions), closest character count.
            ctl = pool.choose(pair, idx, in_prompt=in_prompt,
                              n_target=(n_ins, len(lab_i)), match_chars=True)
            ctl_segs = control_segments(pair, ctl.text)
            lab_c, att_c = runner.analyse_segments(ctl_segs)
            if lab_c.count("injection") != n_ins or len(lab_c) != len(lab_i):
                skipped.append((idx, f"longueurs : insérés {n_ins} vs "
                                     f"{lab_c.count('injection')}, prompt "
                                     f"{len(lab_i)} vs {len(lab_c)} tokens"))
                continue

            sp = config.system_prompt
            a_i = attention_measures(att_i, lab_i.roles, readout_start(
                runner.tokenizer, pair.injected_segments, sp), layers)
            a_c = attention_measures(att_c, lab_c.roles, readout_start(
                runner.tokenizer, ctl_segs, sp), layers)

            spec = None
            if not args.no_spectral:
                lab_b, att_b = runner.analyse_segments(pair.benign_segments)
                spec = [spectral_measures(att, lab, config, layers)
                        for att, lab in ((att_b, lab_b), (att_i, lab_i),
                                         (att_c, lab_c))]
                del att_b

            data.add(a_i, a_c, spec, meta={
                "pair": idx, "position": pair.position,
                "category": pair.attack_category,
                "host_email": pool.contexts.index(pair.context),
                "donor_email": ctl.donor_index, "tokens_inserted": n_ins,
                "prompt_tokens": len(lab_i),
                "chars_attack": len(pair.attack), "chars_control": len(ctl.text),
                "words_attack": len(pair.attack.split()),
                "words_control": len(ctl.text.split()),
                "truncated": int(ctl.truncated), "attack": pair.attack,
                "control": ctl.text})
            del att_i, att_c
        except Exception as exc:                          # noqa: BLE001
            skipped.append((idx, f"{type(exc).__name__}: {exc}"))
            print(f"  [{idx + 1}/{len(pairs)}] ignorée : {exc}", file=sys.stderr)
            continue

        elapsed = time.time() - t_start
        print(f"  [{idx + 1}/{len(pairs)}] {pair.position:<6} "
              f"{n_ins:>3} tokens insérés · {elapsed / (idx + 1):.0f} s/paire")
        if data.n % 5 == 0:                     # a crash loses 5 pairs at most
            data.save(out_dir)

    data.save(out_dir)
    print(f"\n{data.n} paires exploitables, {len(skipped)} ignorées"
          + "".join(f"\n  paire {i} : {why}" for i, why in skipped))
    if data.n < 5:
        print("moins de 5 paires : pas de statistiques", file=sys.stderr)
        return 1
    _analyse_dir(out_dir, args, config.model_slug, args.split)
    print(f"\nSorties -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
