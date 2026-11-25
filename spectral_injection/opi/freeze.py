"""Fige la configuration AVANT execution, et l'horodate.

Le fichier produit precede le run : il atteste de ce qui a ete decide avant
d'avoir vu le moindre resultat. ``run.py`` ecrit sa propre provenance a la
fin ; les deux doivent concorder, et toute divergence est un signal.

    python -m spectral_injection.opi.freeze --out results/opi/pilot
"""

import argparse
import hashlib
import json
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd

from . import dataset as ds
from . import provenance as prov
from . import templates as tmpl_mod
from .communities import GAP_SEARCH_MAX, K_VALUES
from .confounders import CATEGORICAL, CONTINUOUS, GAP_COVARIATES, RULES
from .stats import (CONFIRMATORY, CONFIRMATORY_TRAJ, DEGENERATE_POLICY,
                    EXACT_MAX_N, PRIMARY_AGGREGATOR, SECONDARY_CONTRASTS)
from .structural import BANDS, BAND_NAMES, ZERO_TOL


def code_fingerprint() -> Dict[str, str]:
    """SHA-256 de chaque module du paquet, plus une empreinte globale."""
    here = os.path.dirname(os.path.abspath(__file__))
    out, agg = {}, hashlib.sha256()
    for name in sorted(os.listdir(here)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(here, name), "rb") as fh:
            data = fh.read()
        h = hashlib.sha256(data).hexdigest()
        out[name] = h
        agg.update(name.encode())
        agg.update(h.encode())
    out["_combined"] = agg.hexdigest()
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="fige la configuration du pilote")
    p.add_argument("--out", default="results/opi/pilot")
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--parquet",
                   default="data/opi/data/train-00000-of-00001.parquet")
    p.add_argument("--per-stratum", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-tokens", type=int, default=512)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    from transformers import AutoConfig, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Forme du modele lue dans sa configuration, jamais supposee : la
    # campagne multi-modeles passe par des familles de tailles differentes.
    mc = AutoConfig.from_pretrained(args.model)
    mt = getattr(mc, "text_config", mc)
    n_layers = int(mt.num_hidden_layers)
    n_heads = int(mt.num_attention_heads)
    n_kv = int(getattr(mt, "num_key_value_heads", n_heads))
    tmpl, bos_marker = tmpl_mod.resolve(args.model)
    tinfo = tmpl_mod.verify(tok, tmpl, bos_marker)

    df, templates = ds.load(args.parquet)
    samp = ds.sample_rows(df, args.per_stratum, args.seed)
    ids = samp.row_index.tolist()
    id_fp = hashlib.blake2b(",".join(map(str, ids)).encode(),
                            digest_size=16).hexdigest()

    rec = prov.capture(
        model_id=args.model, tokenizer=tok,
        config={
            "campaign": "pilot",
            "n_examples": len(samp), "per_stratum": args.per_stratum,
            "strata": ["task_type", "attack_type"],
            "n_task_types": int(samp.task_type.nunique()),
            "n_attack_types": int(samp.attack_type.nunique()),
            "conditions": list(ds.CONDITIONS),
            "primary_contrast": ["C4", "C3"],
            "secondary_contrasts": [list(x) for x in SECONDARY_CONTRASTS],
            "max_tokens": args.max_tokens,
            "per_head": True,
            "n_layers_expected": n_layers, "n_heads_expected": n_heads,
            "n_kv_heads": n_kv, "graphs_per_prompt": n_layers * n_heads,
            "system_prompt": None,
            "chat_template": f"experimental, {tinfo['name']}",
            "template_marker_ids": tinfo["marker_ids"],
            "template_bos_marker": bos_marker,
            "n_template_tokens": tinfo["n_template_tokens"],
            "sink_removal": "none", "kron": False,
            "self_loops": {"primary": "conserves",
                           "robustness": "variante nsl_"},
            "graph": "W = (A + A^T)/2 ; L_sym = I - D^-1/2 W D^-1/2",
            "h4_bands": [list(b) for b in BANDS],
            "h4_band_names": list(BAND_NAMES),
            "h4_zero_tol": ZERO_TOL,
            "h5_k_values": list(K_VALUES),
            "h5_eigengap_search_max": GAP_SEARCH_MAX,
            "aggregator": PRIMARY_AGGREGATOR,
            "confirmatory_family": list(CONFIRMATORY) + list(CONFIRMATORY_TRAJ),
            "family_size": len(CONFIRMATORY) + len(CONFIRMATORY_TRAJ),
            "multiplicity": "max-statistic commune a toute la famille",
            "exact_perm_max_n": EXACT_MAX_N,
            "degenerate_policy": DEGENERATE_POLICY,
            "confounders": {"gap_covariates": list(GAP_COVARIATES),
                            "level_covariates": list(CONTINUOUS),
                            "categorical": list(CATEGORICAL),
                            "rules": RULES},
        },
        dataset_path=args.parquet, seed=args.seed,
        extra={
            "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "example_row_index": ids,
            "example_uid": samp.uid.tolist(),
            "example_fingerprint_blake2b": id_fp,
            "code_fingerprint": code_fingerprint(),
            "instruction_templates": templates,
            "balance": {
                "task_type": {k: int(v) for k, v in
                              samp.task_type.value_counts().items()},
                "attack_type": {k: int(v) for k, v in
                                samp.attack_type.value_counts().items()},
                "injected_task": {k: int(v) for k, v in
                                  samp.injected_task.value_counts().items()},
            },
        })
    path = prov.write(rec, os.path.join(args.out, "frozen_config.json"))
    samp[["row_index", "uid", "task_type", "attack_type", "injected_task",
          "sample_id"]].to_csv(os.path.join(args.out, "frozen_examples.csv"),
                               index=False)

    print(f"configuration figee : {path}")
    print(f"  exemples            : {len(samp)}")
    print(f"  empreinte des index : {id_fp}")
    print(f"  empreinte du code   : {rec['extra']['code_fingerprint']['_combined'][:32]}...")
    print(f"  modele              : {args.model}")
    print(f"  tokenizer           : {rec['tokenizer']['class']}, "
          f"vocab {rec['tokenizer']['vocab_size']}")
    print(f"  graine              : {args.seed}")
    print(f"  forme               : {n_layers} couches x {n_heads} tetes "
          f"(kv={n_kv}) = {n_layers * n_heads} graphes/prompt")
    print(f"  template            : {tinfo['name']}, "
          f"T={tinfo['n_template_tokens']} tokens")
    print(f"  famille             : {rec['config']['family_size']} metriques")
    print(f"  equilibre taches    : "
          f"{sorted(set(rec['extra']['balance']['task_type'].values()))}")
    print(f"  equilibre attaques  : "
          f"{sorted(set(rec['extra']['balance']['attack_type'].values()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
