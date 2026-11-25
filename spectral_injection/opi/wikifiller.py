"""Remplissage benin tire de wikitext, apparie EXACTEMENT au token.

POURQUOI UNE SECONDE SOURCE. ``dataset.BenignFiller`` mine le remplissage
dans OPI lui-meme (les ``normal_input`` d'autres lignes). C'est le controle
le plus proche en distribution, mais un audit des 112 remplissages du
pilote a montre deux proprietes genantes :

    110 / 112  se terminent en milieu de phrase  (consequence directe de la
               contrainte de token exact : la coupe tombe ou il faut)
     39 / 112  commencent par un marqueur structure -- << Sentence1: ' >> --
               parce que les taches MRPC et RTE d'OPI formatent ainsi leur
               entree. Ce n'est pas de la prose neutre.

Un remplissage tire de wikitext est de la prose encyclopedique continue :
il supprime le marqueur structure, mais introduit un ECART DE DOMAINE par
rapport au texte hote. Aucune des deux sources n'est strictement meilleure.
Elles sont donc toutes deux conservees et traitees comme DEUX BRAS : si la
signature se reproduit sous les deux, elle ne vient pas du remplissage.

CE QUI RESTE IDENTIQUE a ``BenignFiller``, et qui est l'essentiel :

* la cible est le nombre de tokens de ``I_inj`` MESURE DANS LE PROMPT
  ASSEMBLE, jamais isolement -- le tokenizer fusionne aux frontieres ;
* la coupe se fait par recherche binaire sur les frontieres de MOTS ;
* un candidat partageant une suite de ``NG`` mots avec le texte hote est
  ecarte ;
* l'appariement est exact ou la ligne est declaree non appariee ; jamais
  d'approximation silencieuse.

Le vivier est mis en cache sur disque pour que les executions soient
reproductibles et ne dependent pas du reseau.
"""

import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import NG, MAX_FILLER_CANDIDATES, _grams, _WORDS

#: Emplacement du vivier mis en cache.
DEFAULT_CACHE = os.path.join("data", "wikitext", "pool.json")

#: Parametres d'extraction du vivier. Declares, pas ajustes apres coup.
POOL_ROWS = 60000          # lignes brutes parcourues dans le split train
POOL_MIN_CHARS = 400       # un candidat doit pouvoir couvrir la cible
POOL_MIN_WORDS = 60
POOL_SIZE = 4000           # taille du vivier retenu


def _is_prose(text: str) -> bool:
    """Ecarte titres, lignes vides, listes et fragments de tableaux."""
    t = text.strip()
    if len(t) < POOL_MIN_CHARS or len(t.split()) < POOL_MIN_WORDS:
        return False
    if t.startswith("=") or t.startswith("*") or t.startswith("|"):
        return False
    if "@-@" in t[:120] or t.count("=") > 4:
        return False
    return True


def build_pool(cache: str = DEFAULT_CACHE, seed: int = 0) -> List[str]:
    """Telecharge, filtre et met en cache le vivier wikitext.

    Le depot historique ``wikitext`` n'a pas de namespace et casse avec les
    versions recentes de ``huggingface_hub`` ; on passe par le miroir
    canonique ``Salesforce/wikitext``.
    """
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as fh:
            return json.load(fh)["pool"]

    from datasets import load_dataset
    d = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                     split=f"train[:{POOL_ROWS}]")
    keep = [r["text"].strip() for r in d if _is_prose(r["text"])]
    rng = np.random.default_rng(seed)
    if len(keep) > POOL_SIZE:
        keep = [keep[i] for i in rng.permutation(len(keep))[:POOL_SIZE]]
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    with open(cache, "w", encoding="utf-8") as fh:
        json.dump({"source": "Salesforce/wikitext wikitext-103-raw-v1",
                   "split": f"train[:{POOL_ROWS}]", "seed": seed,
                   "filters": {"min_chars": POOL_MIN_CHARS,
                               "min_words": POOL_MIN_WORDS},
                   "n_pool": len(keep), "pool": keep}, fh, ensure_ascii=False)
    return keep


class WikitextFiller:
    """Interchangeable avec ``dataset.BenignFiller``, meme signature.

    La seule difference est le vivier. L'algorithme d'appariement est
    reproduit a l'identique, y compris l'exclusion par n-grammes et le
    plafond de candidats essayes.
    """

    source = "wikitext-103-raw-v1"

    def __init__(self, seed: int = 0, cache: str = DEFAULT_CACHE):
        self.pool = build_pool(cache, seed=0)
        self.seed = seed

    def fit(self, row, n_tokens, target: int
            ) -> Tuple[Optional[str], Dict]:
        """Remplissage donnant EXACTEMENT ``target`` tokens de plus que C2.

        Rend (texte ou None, diagnostic). Voir ``BenignFiller.fit`` : le
        prompt de reference est ``normal + P_atk + D_inj``, c'est-a-dire
        C2, et le remplissage s'insere a la place exacte de ``I_inj``.
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
            if _grams(cand) & host:
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
                return txt, {"candidates_tried": tried,
                             "tokenizer_calls": calls, "matched": True,
                             "source": self.source}
        return None, {"candidates_tried": tried, "tokenizer_calls": calls,
                      "matched": False, "source": self.source}
