"""Capture l'attention une fois, analyse-la autant de fois qu'on veut.

Reclame depuis le premier jour et jamais construit : jusqu'ici chaque
hypothese nouvelle coutait une passe complete du modele. Le cache rend les
variantes d'analyse gratuites, ce qui est la condition pour balayer les
choix de construction du graphe (H10) sans refaire tourner le modele neuf
fois.

Format, un fichier par paire :

    att_inj, att_ctl   [couches, tetes, T, T] en float16
    roles_inj, roles_ctl   roles des tokens, un par position
    meta               position, categorie, tokens inseres, longueur

L'attention causale est triangulaire inferieure, donc la moitie superieure
est exactement nulle et ``savez_compressed`` la reduit a presque rien : on
garde la matrice entiere pour la simplicite de lecture sans en payer le
stockage. En float16 l'erreur relative est de l'ordre de 1e-3, tres en
dessous des effets mesures (quelques pour cent), et surtout identique dans
les deux conditions d'une paire -- le contraste apparie n'en souffre pas.
"""

import argparse
import csv
import os
import sys
import time

import numpy as np

from .bipia import ensure_data, load_pairs, _load_emails
from .config import RunConfig
from .controls import EmailTextControl, control_segments
from .prompts import build_labelled_prompt_from_segments
from .runner import AttentionRunner


def cache_path(out_dir: str, idx: int) -> str:
    return os.path.join(out_dir, f"pair_{idx:04d}.npz")


def load_pair(path: str):
    d = np.load(path, allow_pickle=True)
    return (d["att_inj"], d["att_ctl"],
            d["roles_inj"], d["roles_ctl"], d["meta"].item())


def iter_cache(out_dir: str):
    for f in sorted(os.listdir(out_dir)):
        if f.startswith("pair_") and f.endswith(".npz"):
            yield load_pair(os.path.join(out_dir, f))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen2.5-0.5b")
    p.add_argument("--split", default="train")
    p.add_argument("--n-pairs", type=int, default=40)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attack-source", default="bipia")
    p.add_argument("--attack-max-chars", type=int, default=None)
    p.add_argument("--out", default="cache/attn")
    args = p.parse_args(argv)

    cfg = RunConfig(model=args.model, max_tokens=args.max_tokens)
    runner = AttentionRunner(cfg)
    layers = list(range(runner.n_layers))
    pairs = load_pairs(split=args.split, n_pairs=args.n_pairs, seed=args.seed,
                       attack_source=args.attack_source,
                       attack_max_chars=args.attack_max_chars)
    emails = _load_emails(ensure_data("data/bipia")[f"email_{args.split}"])
    pool = EmailTextControl(emails, runner.tokenizer, seed=args.seed)

    out_dir = os.path.join(args.out, os.path.basename(cfg.model_id),
                           args.split + ("" if args.attack_source == "bipia"
                                         else "_" + args.attack_source))
    os.makedirs(out_dir, exist_ok=True)
    print(f"{cfg.model_id} : {runner.n_layers} couches x "
          f"{runner.model.config.num_attention_heads} tetes -> {out_dir}",
          flush=True)

    kept, skipped, t0, total_mb = 0, [], time.time(), 0.0
    meta_rows = []
    for idx, pair in enumerate(pairs):
        try:
            lab_i, att_i = runner.analyse_segments(pair.injected_segments)
            n_ins = lab_i.count("injection")

            def in_prompt(text, _p=pair):
                lab = build_labelled_prompt_from_segments(
                    runner.tokenizer, control_segments(_p, text),
                    system_prompt=cfg.system_prompt)
                return lab.count("injection"), len(lab)

            ctl = pool.choose(pair, idx, in_prompt=in_prompt,
                              n_target=(n_ins, len(lab_i)), match_chars=True)
            segs = control_segments(pair, ctl.text)
            lab_c, att_c = runner.analyse_segments(segs)
            if lab_c.count("injection") != n_ins or len(lab_c) != len(lab_i):
                skipped.append((idx, "longueurs non appariees"))
                continue

            A_i = np.stack([att_i[l] for l in layers]).astype(np.float16)
            A_c = np.stack([att_c[l] for l in layers]).astype(np.float16)
            meta = {"pair": idx, "position": pair.position,
                    "category": pair.attack_category,
                    "tokens_inserted": n_ins, "prompt_tokens": len(lab_i),
                    "chars_attack": len(pair.attack),
                    "chars_control": len(ctl.text)}
            path = cache_path(out_dir, kept)
            np.savez_compressed(path, att_inj=A_i, att_ctl=A_c,
                                roles_inj=lab_i.roles, roles_ctl=lab_c.roles,
                                meta=np.array(meta, dtype=object))
            total_mb += os.path.getsize(path) / 1e6
            meta_rows.append(meta)
            kept += 1
            del att_i, att_c, A_i, A_c
        except Exception as exc:                            # noqa: BLE001
            skipped.append((idx, f"{type(exc).__name__}: {exc}"))
            print(f"  [{idx+1}] ignoree : {exc}", file=sys.stderr, flush=True)
            continue
        el = time.time() - t0
        print(f"  [{idx+1}/{len(pairs)}] {pair.position:<6} {n_ins:>3} tok, "
              f"prompt {len(lab_i):>4}, cache {total_mb:>7.0f} Mo "
              f"({el/(idx+1):.1f} s/paire)", flush=True)

    if meta_rows:
        with open(os.path.join(out_dir, "pairs.csv"), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(meta_rows[0].keys()))
            w.writeheader(); w.writerows(meta_rows)
    print(f"\n{kept} paires en cache, {len(skipped)} ignorees, "
          f"{total_mb:.0f} Mo ({total_mb/max(kept,1):.1f} Mo/paire)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
