"""Chargement, inspection et appariement des deux jeux de donnees.

L'appariement en longueur se fait sur le prompt COMPLET formate, systeme
inclus, apres tokenisation -- jamais sur le texte brut ni sur les morceaux
tokenises separement (le tokenizer fusionne aux frontieres).
"""

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import LENGTH_TOL_FRAC, LENGTH_TOL_MIN


# --------------------------------------------------------------------------
# xTRam1/safe-guard-prompt-injection
# --------------------------------------------------------------------------
def load_xtram1():
    from datasets import load_dataset
    return load_dataset("xTRam1/safe-guard-prompt-injection")


def inspect_xtram1(ds) -> Dict:
    out = {"name": "xTRam1/safe-guard-prompt-injection",
           "splits": {k: len(v) for k, v in ds.items()},
           "columns": list(ds["train"].features.keys()),
           "features": {k: str(v) for k, v in ds["train"].features.items()},
           "labels": {}, "system_prompt_provided": False,
           "note": ("aucun System Prompt fourni ; 'text' est un bloc unique. "
                    "Les exemples label=1 sont des phrases adversariales "
                    "autonomes, non des injections enfouies dans un contenu "
                    "externe : le contraste teste 'ce message est-il "
                    "adversarial', pas 'y a-t-il une injection cachee'.")}
    for sp in ds:
        lab = np.asarray(ds[sp]["label"])
        u, c = np.unique(lab, return_counts=True)
        out["labels"][sp] = {int(a): int(b) for a, b in zip(u, c)}
    return out


# --------------------------------------------------------------------------
# reshabhs/SPML_Chatbot_Prompt_Injection
# --------------------------------------------------------------------------
def load_spml() -> pd.DataFrame:
    from datasets import load_dataset
    return load_dataset("reshabhs/SPML_Chatbot_Prompt_Injection")["train"].to_pandas()


def inspect_spml(df: pd.DataFrame) -> Dict:
    g = df.groupby("System Prompt")["Prompt injection"].agg(["count", "sum"])
    both = g[(g["sum"] > 0) & (g["sum"] < g["count"])]
    ct = pd.crosstab(df["Degree"], df["Prompt injection"])
    return {"name": "reshabhs/SPML_Chatbot_Prompt_Injection",
            "splits": {"train": int(len(df))},
            "columns": list(df.columns),
            "labels": {int(k): int(v) for k, v in
                       df["Prompt injection"].value_counts().items()},
            "degree": {int(k): int(v) for k, v in
                       df["Degree"].value_counts().sort_index().items()},
            "degree_x_injection": {int(d): {int(c): int(ct.loc[d, c])
                                            for c in ct.columns}
                                   for d in ct.index},
            "degree_semantics": ("Degree=0 <=> injection=0 exactement ; "
                                 "Degree in {1,2,3,4,5,7,10} <=> injection=1. "
                                 "Degree est une intensite d'injection."),
            "source": {str(k): int(v) for k, v in
                       df["Source"].fillna("<null>").value_counts().items()},
            "system_prompt_provided": True,
            "n_system_prompts_distinct": int(df["System Prompt"].nunique()),
            "system_prompt_constant": bool(df["System Prompt"].nunique() == 1),
            "rows_per_system_median": float(g["count"].median()),
            "n_systems_with_both_labels": int(len(both)),
            "rows_covered_by_both_label_systems": int(both["count"].sum())}


# --------------------------------------------------------------------------
# Appariement
# --------------------------------------------------------------------------
def _tol(n: int) -> int:
    return max(int(round(LENGTH_TOL_FRAC * n)), LENGTH_TOL_MIN)


def pair_xtram1(ds, n_tokens: Callable[[str], int], n_pairs: int,
                max_tokens: int, seed: int, split: str = "train"
                ) -> Tuple[pd.DataFrame, Dict]:
    """Chaque exemple label=1 est apparie au label=0 **non utilise** de
    longueur la plus proche (prompt complet formate), dans la tolerance.

    Les deux textes sont DIFFERENTS en tout sauf la longueur : c'est un
    controle entre exemples, bien plus faible qu'un controle intra-exemple.
    ``n_tokens`` doit rendre la longueur du prompt formate complet.
    """
    d = ds[split]
    lab = np.asarray(d["label"])
    rng = np.random.default_rng(seed)
    adv_idx = rng.permutation(np.flatnonzero(lab == 1))
    ctl_idx = rng.permutation(np.flatnonzero(lab == 0))

    # longueurs calculees a la demande, memorisees
    cache: Dict[int, int] = {}

    def L(i: int) -> int:
        if i not in cache:
            cache[i] = n_tokens(d[int(i)]["text"])
        return cache[i]

    # vivier de controles : on tokenise un sous-ensemble suffisant
    pool = [int(i) for i in ctl_idx[:max(60 * n_pairs, 600)]]
    pool_len = np.array([L(i) for i in pool])
    keep = pool_len <= max_tokens
    pool = [p for p, k in zip(pool, keep) if k]
    pool_len = pool_len[keep]
    used = np.zeros(len(pool), bool)

    rows, log = [], {"adv_too_long": 0, "adv_unmatched": 0, "ctl_pool": len(pool),
                     "ctl_pool_dropped_too_long": int((~keep).sum())}
    for i in adv_idx:
        if len(rows) >= n_pairs:
            break
        la = L(int(i))
        if la > max_tokens:
            log["adv_too_long"] += 1
            continue
        gap = np.abs(pool_len - la).astype(float)
        gap[used] = np.inf
        j = int(np.argmin(gap))
        if gap[j] > _tol(la):
            log["adv_unmatched"] += 1
            continue
        used[j] = True
        rows.append({"pair_id": len(rows), "adv_row": int(i), "ctl_row": pool[j],
                     "adv_text": d[int(i)]["text"], "ctl_text": d[pool[j]]["text"],
                     "adv_tokens": int(la), "ctl_tokens": int(pool_len[j]),
                     "gap_tokens": int(pool_len[j] - la)})
    log["n_pairs"] = len(rows)
    return pd.DataFrame(rows), log


def pair_spml(df: pd.DataFrame, n_tokens: Callable[[str, str], int],
              n_pairs: int, max_tokens: int, seed: int
              ) -> Tuple[pd.DataFrame, Dict]:
    """Appariement **au sein du meme System Prompt** : pour un systeme ayant
    les deux labels, un User Prompt injecte est apparie au User Prompt non
    injecte de longueur la plus proche. Le systeme est donc identique dans
    la paire par construction, ce que le protocole exige.

    ``n_tokens(system, user)`` rend la longueur du prompt formate complet.
    """
    rng = np.random.default_rng(seed)
    g = df.groupby("System Prompt")["Prompt injection"].agg(["count", "sum"])
    systems = g[(g["sum"] > 0) & (g["sum"] < g["count"])].index.tolist()
    rng.shuffle(systems)

    rows, log = [], {"systems_considered": 0, "systems_unmatched": 0,
                     "pairs_dropped_too_long": 0}
    for sysp in systems:
        if len(rows) >= n_pairs:
            break
        log["systems_considered"] += 1
        sub = df[df["System Prompt"] == sysp]
        adv = sub[sub["Prompt injection"] == 1]
        ctl = sub[sub["Prompt injection"] == 0]
        a = adv.iloc[int(rng.integers(len(adv)))]
        la = n_tokens(sysp, a["User Prompt"])
        if la > max_tokens:
            log["pairs_dropped_too_long"] += 1
            continue
        lc = np.array([n_tokens(sysp, u) for u in ctl["User Prompt"]])
        ok = lc <= max_tokens
        if not ok.any():
            log["pairs_dropped_too_long"] += 1
            continue
        gap = np.abs(lc - la).astype(float)
        gap[~ok] = np.inf
        j = int(np.argmin(gap))
        if gap[j] > _tol(la):
            log["systems_unmatched"] += 1
            continue
        c = ctl.iloc[j]
        rows.append({"pair_id": len(rows), "system_prompt": sysp,
                     "adv_row": int(a.name), "ctl_row": int(c.name),
                     "adv_user": a["User Prompt"], "ctl_user": c["User Prompt"],
                     "degree": int(a["Degree"]), "source": str(a["Source"]),
                     "adv_tokens": int(la), "ctl_tokens": int(lc[j]),
                     "gap_tokens": int(lc[j] - la)})
    log["n_pairs"] = len(rows)
    return pd.DataFrame(rows), log
