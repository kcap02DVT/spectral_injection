"""Segmentation en SIX regions, raffinement de la segmentation U / E / T.

``capture.segment`` rend trois regions : U (instruction legitime), E (bloc
externe), T (tokens de template). E est un agregat :

    E = normal_input + P_atk + I_inj + D_inj

Ce module le decompose, ce qui est possible SANS rien deviner parce que la
decomposition d'OPI est exacte et verifiee ligne a ligne (33 600 / 33 600,
voir ``dataset``). Les six regions sont, dans l'ordre d'apparition :

    task       l'instruction legitime            (= U de la segmentation a 3)
    normal     le contenu externe benin          (normal_input)
    preamble   le prefixe d'attaque              (P_atk)
    slot       l'emplacement variable            (I_inj en C4, filler en C3)
    tail       les donnees injectees             (D_inj)
    template   les tokens de format              (= T de la segmentation a 3)

POURQUOI << slot >>. C'est la seule region dont le CONTENU differe entre
C3 et C4 ; sa longueur en tokens est identique par construction. C'est donc
le seul endroit ou une difference observee peut etre attribuee a
l'instruction injectee plutot qu'a un changement de taille de graphe.

ATTRIBUTION. Un token est attribue a la region qui contient le MILIEU de
son span de caracteres. Le milieu, et non le debut, parce qu'un token peut
chevaucher une frontiere : le tokenizer fusionne volontiers le dernier
caractere d'un segment avec le premier du suivant. Un token frontalier est
alors attribue au segment dont il couvre la plus grande part, ce qui est le
choix le moins arbitraire disponible sans retokeniser chaque segment a part
-- ce qui serait faux, puisque la tokenisation depend du contexte.

Les frontieres sont calculees en CARACTERES sur le prompt formate, a partir
de ``seg.user_start`` et des longueurs des composants. Elles ne sont jamais
devinees : ``audit`` verifie que la somme des tokens des six regions egale
le nombre total de tokens, que ``task`` et ``slot`` sont contigus, et que
les comptes de ``task``, ``normal``, ``preamble``, ``tail`` et
``template`` sont IDENTIQUES entre C3 et C4 -- toute difference y
signalerait une derive de frontiere.

Ce module est ADDITIF : il ne touche pas ``capture.py``, dont l'empreinte
SHA-256 est enregistree dans le gel du pilote.
"""

from typing import Dict, List, Sequence, Tuple

import numpy as np

#: Ordre de declaration. Il fixe l'ordre des 21 blocs et des 36 flux.
REGIONS6: Tuple[str, ...] = ("task", "normal", "preamble", "slot",
                             "tail", "template")

#: Les cinq regions dont la taille en tokens DOIT coincider entre C3 et C4.
#: ``slot`` en fait partie : l'appariement exact l'impose.
INVARIANT_REGIONS: Tuple[str, ...] = REGIONS6


def components(row, slot_text: str) -> List[Tuple[str, str]]:
    """Les quatre composants du bloc externe, dans l'ordre, avec leur texte.

    ``slot_text`` vaut ``row.I_inj`` pour C4 et le remplissage benin pour
    C3. L'ordre reproduit exactement ``dataset.external_C3`` /
    ``external_C4``.
    """
    return [("normal", row.normal_input), ("preamble", row.P_atk),
            ("slot", slot_text), ("tail", row.D_inj)]


def label_tokens(seg, instruction: str,
                 comps: Sequence[Tuple[str, str]]) -> np.ndarray:
    """``Segmented`` + composants -> etiquettes ``[T]`` parmi ``REGIONS6``.

    Le prompt formate est ``prefix + instruction + "\\n\\n" + external +
    suffix`` apres ``rstrip`` du tour utilisateur (voir
    ``capture.build_user_turn``). Les frontieres en caracteres suivent donc

        task      [user_start, user_start + len(instruction))
        external  [task_end + 2, task_end + 2 + len(external.rstrip()))

    ou le ``+ 2`` est le ``"\\n\\n"`` separateur. Le ``rstrip`` final du
    tour utilisateur peut raccourcir le dernier composant, d'ou le
    ``min(fin_composant, fin_externe)``.
    """
    out = np.full(len(seg), "template", dtype="<U10")
    u0 = seg.user_start
    u1 = u0 + len(instruction)
    e0 = u1 + 2                                     # le "\n\n" separateur

    spans, p = [], e0
    for name, txt in comps:
        spans.append((name, p, p + len(txt)))
        p += len(txt)
    e1 = e0 + len("".join(t for _, t in comps).rstrip())

    for k, (lo, hi) in enumerate(seg.offsets):
        mid = (lo + hi) / 2 if hi > lo else float(lo)
        if u0 <= mid < u1:
            out[k] = "task"
        elif e0 <= mid < e1:
            for name, a, b in spans:
                if a <= mid < min(b, e1):
                    out[k] = name
                    break
    return out


def counts(labels: Sequence[str]) -> Dict[str, int]:
    lab = np.asarray([str(x) for x in labels])
    return {r: int((lab == r).sum()) for r in REGIONS6}


def audit(labels_c3: Sequence[str], labels_c4: Sequence[str]) -> Dict:
    """Controles de coherence d'une PAIRE C3 / C4.

    Rend un dictionnaire de diagnostics ; ``ok`` est vrai seulement si tous
    les controles passent. L'appelant doit lever sur ``ok`` faux : une
    frontiere qui glisse deplacerait toutes les energies sans erreur
    visible.
    """
    c3, c4 = counts(labels_c3), counts(labels_c4)
    n3, n4 = len(labels_c3), len(labels_c4)
    out = {
        "counts_C3": c3, "counts_C4": c4,
        "n_tokens_C3": n3, "n_tokens_C4": n4,
        "total_equal": n3 == n4,
        "partition_C3": sum(c3.values()) == n3,
        "partition_C4": sum(c4.values()) == n4,
        "no_empty_C3": all(v > 0 for k, v in c3.items() if k != "preamble"),
        "no_empty_C4": all(v > 0 for k, v in c4.items() if k != "preamble"),
        "region_sizes_equal": {r: c3[r] == c4[r] for r in INVARIANT_REGIONS},
    }
    for cond, lab in (("C3", labels_c3), ("C4", labels_c4)):
        a = np.asarray([str(x) for x in lab])
        for r in ("task", "slot", "normal", "tail"):
            idx = np.flatnonzero(a == r)
            out[f"{r}_contiguous_{cond}"] = bool(
                idx.size > 0 and idx.size == idx.max() - idx.min() + 1)
    out["ok"] = bool(
        out["total_equal"] and out["partition_C3"] and out["partition_C4"]
        and out["no_empty_C3"] and out["no_empty_C4"]
        and all(out["region_sizes_equal"].values())
        and all(v for k, v in out.items() if k.startswith(
            ("task_contiguous", "slot_contiguous", "normal_contiguous",
             "tail_contiguous"))))
    return out


def to_three(labels: Sequence[str]) -> np.ndarray:
    """Replie les six regions sur les trois d'origine : U / E / T.

    Permet de verifier que la segmentation a six regions est un RAFFINEMENT
    de celle a trois, et non une segmentation concurrente.
    """
    m = {"task": "U", "normal": "E", "preamble": "E", "slot": "E",
         "tail": "E", "template": "T"}
    return np.array([m[str(x)] for x in labels], dtype="<U1")
