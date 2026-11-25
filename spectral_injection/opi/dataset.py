"""Adaptateur Open-Prompt-Injection : decomposition et quatre conditions.

Rien n'est devine. La decomposition

    attack_input = normal_input + P_atk + I_inj + D_inj

est obtenue par combinatoire sur le plan factoriel, puis verifiee ligne a
ligne. Les trois assertions ont ete controlees sur les 33 600 lignes :

    attack_input.startswith(normal_input)      33 600 / 33 600
    gabarit d'instruction localise             33 600 / 33 600
    injected_task != task_type                 0 violation

Le gabarit d'instruction injectee est recupere ainsi : dans une cellule
(task_type, attack_type, injected_task) de 200 lignes, seules les donnees
injectees varient, donc le plus long prefixe commun des spans vaut
``P_atk + I_inj`` ; le plus long suffixe commun de ces prefixes, a travers
les cellules partageant la meme injected_task, vaut ``I_inj``. Le gabarit
est ensuite coupe au premier ``": "``, ce qui corrige les deux taches
structurees (mrpc, rte) dont le prefixe commun avait absorbe le debut des
donnees. Couverture du gabarit raccourci : 100 %.
"""

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NG = 6                       # longueur du n-gramme d'exclusion des doublons
MAX_FILLER_CANDIDATES = 40   # au-dela, la ligne est marquee non appariee
CONDITIONS = ("C1", "C2", "C3", "C4")


# --------------------------------------------------------------------------
# Chargement et decomposition
# --------------------------------------------------------------------------
def _lcp(strings: Sequence[str]) -> str:
    s1, s2 = min(strings), max(strings)
    i = 0
    while i < len(s1) and s1[i] == s2[i]:
        i += 1
    return s1[:i]


def _lcsuffix(strings: Sequence[str]) -> str:
    return _lcp([s[::-1] for s in strings])[::-1]


def recover_templates(df: pd.DataFrame) -> Dict[str, str]:
    """Gabarits d'instruction injectee, un par ``injected_task``."""
    cell = (df.groupby(["task_type", "attack_type", "injected_task"])
              .span.apply(lambda s: _lcp(s.tolist())))
    out = {}
    for task, g in cell.groupby(level="injected_task"):
        full = _lcsuffix(g.tolist())
        i = full.find(": ")
        out[task] = full[:i + 2] if i >= 0 else full
    return out


def load(path: str = "data/opi/data/train-00000-of-00001.parquet",
         templates_path: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Lit le parquet, decompose, verifie. Rend (df, gabarits)."""
    df = pd.read_parquet(path)
    required = {"instruction", "normal_input", "attack_input", "task_type",
                "attack_type", "injected_task", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"colonnes absentes du parquet : {sorted(missing)}")

    # identifiant stable : sample_id n'est PAS unique globalement
    df = df.reset_index(drop=True)
    df["row_index"] = np.arange(len(df))
    df["uid"] = [f"{t}|{a}|{j}|{s}" for t, a, j, s in zip(
        df.task_type, df.attack_type, df.injected_task, df.sample_id)]
    if df.uid.duplicated().any():
        raise ValueError("uid non unique : le quadruplet ne suffit pas")

    pref = np.fromiter((a.startswith(n) for a, n in
                        zip(df.attack_input, df.normal_input)), bool, len(df))
    if not pref.all():
        raise ValueError(
            f"attack_input ne commence pas par normal_input sur "
            f"{int((~pref).sum())} lignes ; la decomposition est invalide")
    df["span"] = [a[len(n):] for a, n in zip(df.attack_input, df.normal_input)]

    tmpl = (json.load(open(templates_path, encoding="utf-8"))
            if templates_path and os.path.exists(templates_path)
            else recover_templates(df))

    pos = np.array([s.find(tmpl[t]) for s, t in zip(df.span, df.injected_task)])
    if (pos < 0).any():
        raise ValueError(f"gabarit introuvable sur {int((pos < 0).sum())} lignes")
    ln = np.array([len(tmpl[t]) for t in df.injected_task])
    df["P_atk"] = [s[:p] for s, p in zip(df.span, pos)]
    df["I_inj"] = [s[p:p + l] for s, p, l in zip(df.span, pos, ln)]
    df["D_inj"] = [s[p + l:] for s, p, l in zip(df.span, pos, ln)]
    if not (df.P_atk + df.I_inj + df.D_inj == df.span).all():
        raise ValueError("recomposition du span inexacte")

    # separateur = espace initiale de P_atk ; le reste de P_atk est une
    # directive (pour ignore/combine) et appartient donc a D = P_atk' + I_inj
    sep = df.P_atk.str.extract(r"^(\s*)", expand=False).fillna("")
    df["sep"] = sep
    df["P_body"] = [p[len(s):] for p, s in zip(df.P_atk, sep)]
    df["D_full"] = df.P_body + df.I_inj          # la directive injectee
    return df, tmpl


# --------------------------------------------------------------------------
# Construction des quatre conditions
# --------------------------------------------------------------------------
def external_C1(row) -> str:
    return row.normal_input


def external_C2(row) -> str:
    """normal + P_atk + D_inj : la directive est retiree, le prefixe reste."""
    return row.normal_input + row.P_atk + row.D_inj


def external_C4(row) -> str:
    return row.attack_input


def external_C3(row, filler: str) -> str:
    """normal + P_atk + remplissage + D_inj.

    Le remplissage occupe exactement la place de ``I_inj``. Le prefixe
    d'attaque ``P_atk`` est **conserve a l'identique de C4** : decision
    methodologique validee.

    Consequence a porter dans toute interpretation : pour ``ignore`` et
    ``combine``, ``P_atk`` contient lui-meme une directive adversariale
    (<< Ignore previous instructions >>, et une fausse reponse pour
    ``combine``). C3 n'est donc PAS un controle purement benin sur ces deux
    familles. Le contraste C4 - C3 y isole la seule instruction de tache
    injectee, le reste de l'appareil d'attaque etant present des deux cotes.
    L'analyse doit etre stratifiee par ``attack_type``.
    """
    return row.normal_input + row.P_atk + filler + row.D_inj


# --------------------------------------------------------------------------
# Remplissage benin apparie au token, dans le prompt assemble
# --------------------------------------------------------------------------
_WORDS = re.compile(r"\S+")


def _grams(text: str, n: int = NG) -> set:
    w = re.sub(r"\d", "#", text.lower()).split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 1))}


class BenignFiller:
    """Vivier de prose benigne, mine dans le dataset lui-meme.

    Jamais de texte ecrit pour l'experience : les candidats sont les
    ``normal_input`` d'autres lignes. L'hote et tout candidat partageant une
    suite de ``NG`` mots avec lui sont exclus.

    L'appariement compte les tokens **dans le prompt assemble**, jamais
    isolement : le tokenizer fusionne aux frontieres et un comptage a part
    se trompe regulierement d'un token.
    """

    def __init__(self, df: pd.DataFrame, seed: int = 0):
        self.pool = df.normal_input.drop_duplicates().tolist()
        self.seed = seed

    def fit(self, row, n_tokens, target: int) -> Tuple[Optional[str], Dict]:
        """Remplissage donnant EXACTEMENT ``target`` tokens de plus que le
        prompt sans l'instruction injectee.

        Le remplissage s'insere entre ``P_atk`` et ``D_inj``, c'est-a-dire
        exactement a la place de ``I_inj``. Le prompt de reference est donc
        ``normal + P_atk + D_inj``, qui est precisement C2.

        Rend (texte ou None, diagnostic).
        """
        prefix = row.normal_input + row.P_atk
        suffix = row.D_inj
        base = n_tokens(row.instruction, prefix + suffix)
        host = _grams(row.normal_input)
        rng = np.random.default_rng(self.seed + int(row.row_index))
        order = rng.permutation(len(self.pool))
        tried = calls = 0
        for j in order[:MAX_FILLER_CANDIDATES]:
            cand = self.pool[j]
            if cand == row.normal_input or (_grams(cand) & host):
                continue
            tried += 1
            ends = [m.end() for m in _WORDS.finditer(cand)]
            if not ends:
                continue
            lo, hi, best = 0, len(ends) - 1, -1
            while lo <= hi:
                mid = (lo + hi) // 2
                d = n_tokens(row.instruction,
                             prefix + cand[:ends[mid]] + " " + suffix) - base
                calls += 1
                if d <= target:
                    best, lo = mid, mid + 1
                else:
                    hi = mid - 1
            if best < 0:
                continue
            txt = cand[:ends[best]] + " "
            calls += 1
            if n_tokens(row.instruction, prefix + txt + suffix) - base == target:
                return txt, {"candidates_tried": tried, "tokenizer_calls": calls,
                             "matched": True}
        return None, {"candidates_tried": tried, "tokenizer_calls": calls,
                      "matched": False}


# --------------------------------------------------------------------------
# Echantillonnage
# --------------------------------------------------------------------------
def _stable_seed(*parts) -> int:
    """Graine deterministe entre processus.

    ``hash()`` de Python est sale aleatoirement a chaque demarrage
    (PYTHONHASHSEED), donc ``hash(("opi", key, seed))`` donne un tirage
    DIFFERENT d'une execution a l'autre pour la meme graine nominale. Le
    defaut a ete constate : deux processus rendaient deux echantillons
    distincts. BLAKE2b est stable par construction.
    """
    import hashlib

    h = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"),
                        digest_size=8)
    return int.from_bytes(h.digest(), "big") % (2 ** 32)


def sample_rows(df: pd.DataFrame, per_stratum: int = 4, seed: int = 0,
                strata: Sequence[str] = ("task_type", "attack_type")
                ) -> pd.DataFrame:
    """Tirage stratifie et reproductible.

    Les strates sont le produit ``task_type x attack_type``, soit 7 x 4 = 28
    cellules ; ``per_stratum=4`` donne les 112 exemples du pilote.

    ``injected_task`` tourne a l'interieur de chaque strate ET d'une strate
    a l'autre. Prendre simplement ``tasks[k % len(tasks)]`` ne marche pas :
    avec 6 taches injectees disponibles et 4 tirages, les deux dernieres de
    l'ordre alphabetique ne sortaient JAMAIS -- defaut constate, ``sentiment``
    et ``spam`` etaient absents des 112 exemples. Le decalage de depart tourne
    donc avec l'indice de strate, ce qui etale les 6 taches sur les 28
    strates.
    """
    out = []
    for s_i, (key, g) in enumerate(df.groupby(list(strata), sort=True)):
        rng = np.random.default_rng(_stable_seed("opi", key, seed))
        tasks = sorted(g.injected_task.unique())
        picks = []
        for k in range(per_stratum):
            t = tasks[(s_i * per_stratum + k) % len(tasks)]
            sub = g[g.injected_task == t]
            picks.append(sub.iloc[rng.integers(len(sub))])
        out.extend(picks)
    return pd.DataFrame(out).sort_values("row_index").reset_index(drop=True)
