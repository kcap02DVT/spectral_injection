"""Run the four routing hypotheses over BIPIA pairs, injected vs control.

One forward pass per condition per pair; everything else is numpy. The
control has the same inserted token count AND the same total prompt length as
the attack, so the statistics are compared at identical token positions.

    python -m spectral_injection.hypotheses_cli --model smollm2-135m \
        --split train --n-pairs 20 --out results/hyp_h1h4
"""

import argparse, csv, os, sys, time
from typing import List

import numpy as np

from .bipia import BipiaPair, ensure_data, load_pairs, _load_emails
from .config import RunConfig
from .controls import EmailTextControl, control_segments
from .gsp_hypotheses import NAMES, stack
from .prompts import build_labelled_prompt_from_segments
from .runner import AttentionRunner


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="smollm2-135m")
    p.add_argument("--split", default="train")
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/gsp")
    p.add_argument("--cache-dir", default="data/bipia")
    p.add_argument("--control", default="email_text",
                   choices=["email_text", "benign_instruction",
                            "human_imperative"])
    p.add_argument("--offline", action="store_true")
    args = p.parse_args(argv)

    cfg = RunConfig(model=args.model, max_tokens=args.max_tokens)
    runner = AttentionRunner(cfg)
    layers = list(range(runner.n_layers))
    print(f"{cfg.model_id} : {runner.n_layers} couches, "
          f"{runner.model.config.num_attention_heads} tetes", flush=True)

    pairs = load_pairs(split=args.split, n_pairs=args.n_pairs, seed=args.seed)
    emails = _load_emails(ensure_data(args.cache_dir)[f"email_{args.split}"])
    if args.control == "human_imperative":
        from .controls import (BenignInstructionControl,
                               load_human_imperatives)
        pool = BenignInstructionControl(load_human_imperatives(),
                                        runner.tokenizer, seed=args.seed)
    elif args.control == "benign_instruction":
        from .bipia import load_neuralchemy_benign
        from .controls import BenignInstructionControl
        pool = BenignInstructionControl(
            load_neuralchemy_benign(args.split, max_chars=400,
                                    offline=args.offline),
            runner.tokenizer, seed=args.seed)
    else:
        pool = EmailTextControl(emails, runner.tokenizer, seed=args.seed)
    print(f"controle {args.control} : {len(pool.windows)} candidats",
          flush=True)

    tag = "" if args.control == "email_text" else "_" + args.control
    out_dir = os.path.join(args.out, os.path.basename(cfg.model_id),
                           args.split + tag)
    os.makedirs(out_dir, exist_ok=True)

    values, meta, skipped = [], [], []
    t0 = time.time()
    for idx, pair in enumerate(pairs):
        try:
            lab_i, att_i = runner.analyse_segments(pair.injected_segments)
            n_ins = lab_i.count("injection")

            def in_prompt(text, _pair=pair):
                lab = build_labelled_prompt_from_segments(
                    runner.tokenizer, control_segments(_pair, text),
                    system_prompt=cfg.system_prompt)
                return lab.count("injection"), len(lab)

            ctl = pool.choose(pair, idx, in_prompt=in_prompt,
                              n_target=(n_ins, len(lab_i)), match_chars=True)
            ctl_segs = control_segments(pair, ctl.text)
            lab_c, att_c = runner.analyse_segments(ctl_segs)
            if lab_c.count("injection") != n_ins or len(lab_c) != len(lab_i):
                skipped.append((idx, "longueurs non appariees"))
                continue

            v_i = stack(att_i, lab_i.roles, layers, cfg)
            v_c = stack(att_c, lab_c.roles, layers, cfg)
            values.append(np.stack([v_i, v_c]))          # [2, L, H, 5]
            meta.append({"pair": idx, "position": pair.position,
                         "category": pair.attack_category,
                         "tokens_inserted": n_ins,
                         "prompt_tokens": len(lab_i),
                         "chars_attack": len(pair.attack),
                         "chars_control": len(ctl.text)})
            del att_i, att_c
        except Exception as exc:                          # noqa: BLE001
            skipped.append((idx, f"{type(exc).__name__}: {exc}"))
            print(f"  [{idx+1}/{len(pairs)}] ignoree : {exc}",
                  file=sys.stderr, flush=True)
            continue
        el = time.time() - t0
        print(f"  [{idx+1}/{len(pairs)}] {pair.position:<6} {n_ins:>3} tok "
              f"inseres, prompt {len(lab_i):>4} · {el/(idx+1):.1f} s/paire",
              flush=True)

    if not values:
        print("aucune paire exploitable"); return 1
    arr = np.asarray(values, dtype=np.float32)            # [N, 2, L, H, 5]
    np.savez_compressed(os.path.join(out_dir, "values.npz"),
                        values=arr, layers=np.asarray(layers),
                        names=np.asarray(NAMES))
    with open(os.path.join(out_dir, "pairs.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(meta[0].keys()))
        w.writeheader(); w.writerows(meta)
    print(f"\n{arr.shape[0]} paires, {len(skipped)} ignorees -> {out_dir}")
    for i, why in skipped[:10]:
        print(f"  paire {i} : {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
