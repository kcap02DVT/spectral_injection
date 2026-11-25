"""Registre de templates experimentaux, un par famille de modeles.

Ce module est **additif**. Il n'existe que pour permettre la campagne
multi-modeles sans toucher aux modules de calcul, dont les empreintes
SHA-256 sont enregistrees dans le gel du pilote Llama-3.2-1B.

``capture.verify_template`` code en dur le marqueur ``<|begin_of_text|>``
dans son controle d'unicite du BOS, ce qui est specifique a Llama-3. Plutot
que de modifier ``capture.py``, ce module en fournit une version generalisee
qui prend le marqueur de BOS en parametre et le rend facultatif -- Qwen n'a
pas de BOS du tout. ``capture.py`` reste donc bit-identique au pilote.

Le meme principe gouverne les trois templates : **aucun message systeme**,
format declare explicitement, verifie contre le vocabulaire reel du
tokenizer. Les templates officiels des trois familles fabriquent un bloc
systeme non demande :

* Llama-3.2 insere ``Cutting Knowledge Date`` et ``Today Date``, dont la
  valeur **change chaque jour** ;
* Qwen2.5 insere ``You are Qwen, created by Alibaba Cloud. You are a
  helpful assistant.`` ;
* Gemma n'a **pas de role systeme du tout** -- la contrainte y est native,
  et non imposee.

Consequence a porter dans la comparaison : la taille de la region ``T``
depend de la famille (8 tokens pour Qwen, 9 pour Gemma, 10 pour Llama-3,
plus le token frontalier). Ce n'est pas une derive du protocole mais une
propriete du modele. Ce qui doit rester constant, et qui est verifie par
l'audit de segmentation, c'est la taille de ``T`` **a l'interieur** d'un
modele donne.
"""

from typing import Dict, Optional, Tuple

from .capture import LLAMA3_USER_ONLY, ExperimentalTemplate

#: Qwen2.5 / Qwen3 -- format ChatML, message utilisateur seul.
#: Qwen n'a pas de token de debut de sequence : ``bos_marker`` vaut None.
QWEN_CHATML_USER_ONLY = ExperimentalTemplate(
    name="qwen-chatml-user-only-nosystem",
    prefix="<|im_start|>user\n",
    suffix="<|im_end|>\n<|im_start|>assistant\n",
    markers=("<|im_start|>", "<|im_end|>"),
)

#: Gemma-2 / Gemma-3 -- format a tours, message utilisateur seul.
#: Le role assistant s'appelle ``model`` dans cette famille.
GEMMA_USER_ONLY = ExperimentalTemplate(
    name="gemma-user-only-nosystem",
    prefix="<bos><start_of_turn>user\n",
    suffix="<end_of_turn>\n<start_of_turn>model\n",
    markers=("<bos>", "<start_of_turn>", "<end_of_turn>"),
)

#: Mistral / Ministral -- format ``[INST]``, message utilisateur seul.
#: Le template officiel de Ministral-3 injecte un bloc ``[SYSTEM_PROMPT]``
#: de plus de 2 000 caracteres qui contient ``{today}`` et ``{yesterday}``,
#: non resolus : sa valeur changerait chaque jour. Les trois marqueurs
#: ``<s>``, ``[INST]``, ``[/INST]`` sont des tokens uniques (ids 1, 3, 4) et
#: le rendu declare coute exactement 3 tokens de template.
MISTRAL_INST_USER_ONLY = ExperimentalTemplate(
    name="mistral-inst-user-only-nosystem",
    prefix="<s>[INST]",
    suffix="[/INST]",
    markers=("<s>", "[INST]", "[/INST]"),
)

#: Phi-3 / Phi-4 -- cas remarquable : le template OFFICIEL est deja
#: utilisateur seul, ``<|user|>...<|end|><|assistant|>``, sans aucun bloc
#: systeme ni date. Le template declare ci-dessous est donc IDENTIQUE au
#: template officiel, et non une version depouillee de celui-ci. Les trois
#: marqueurs sont des tokens uniques (ids 200021, 200020, 200019) et le
#: format coute exactement 3 tokens.
PHI_USER_ONLY = ExperimentalTemplate(
    name="phi-user-only-official",
    prefix="<|user|>",
    suffix="<|end|><|assistant|>",
    markers=("<|user|>", "<|end|>", "<|assistant|>"),
)

#: nom court -> (template, marqueur de BOS ou None)
REGISTRY: Dict[str, Tuple[ExperimentalTemplate, Optional[str]]] = {
    "llama3": (LLAMA3_USER_ONLY, "<|begin_of_text|>"),
    "qwen": (QWEN_CHATML_USER_ONLY, None),
    "gemma": (GEMMA_USER_ONLY, "<bos>"),
    "mistral": (MISTRAL_INST_USER_ONLY, "<s>"),
    "phi": (PHI_USER_ONLY, None),
}

#: fragment d'identifiant de modele -> nom court du template
#: ``ministral`` doit figurer SEPAREMENT : ce n'est pas une sous-chaine de
#: ``mistral`` (m-i-n-i-s-t-r-a-l contre m-i-s-t-r-a-l).
_PATTERNS = (
    ("llama-3", "llama3"),
    ("llama3", "llama3"),
    ("tinyllama", None),          # format Zephyr, non declare ici
    ("qwen", "qwen"),
    ("gemma", "gemma"),
    ("ministral", "mistral"),
    ("mistral", "mistral"),
    ("phi-3", "phi"),
    ("phi-4", "phi"),
    ("phi3", "phi"),
    ("phi4", "phi"),
)


def resolve(model_id: str) -> Tuple[ExperimentalTemplate, Optional[str]]:
    """Template experimental pour un identifiant de modele.

    Leve plutot que de deviner : un template errone deplacerait la
    segmentation sans produire d'erreur visible.
    """
    low = model_id.lower()
    for frag, key in _PATTERNS:
        if frag in low:
            if key is None:
                raise RuntimeError(
                    f"famille reconnue mais template non declare pour "
                    f"{model_id!r} ; l'ajouter explicitement au registre")
            return REGISTRY[key]
    raise RuntimeError(
        f"aucun template declare pour {model_id!r} ; les familles connues "
        f"sont {sorted(REGISTRY)}. Ne pas se rabattre sur "
        f"apply_chat_template : il fabriquerait un bloc systeme.")


def verify(tokenizer, template: ExperimentalTemplate,
           bos_marker: Optional[str] = None) -> Dict:
    """Controle un template contre le vocabulaire reel du tokenizer.

    Version generalisee de ``capture.verify_template`` : le marqueur de BOS
    est un parametre, et son absence (Qwen) desactive simplement le controle
    d'unicite correspondant. Tous les autres controles sont identiques.

    Leve si un marqueur n'existe pas, s'il ne tokenise pas en un token
    unique, si un role ``system`` subsiste, si le BOS declare n'apparait pas
    exactement une fois, ou si le texte utilisateur ne survit pas a un
    aller-retour de tokenisation.
    """
    ids = {}
    for m in template.markers:
        tid = tokenizer.convert_tokens_to_ids(m)
        unk = getattr(tokenizer, "unk_token_id", None)
        if tid is None or (unk is not None and tid == unk):
            raise RuntimeError(f"marqueur absent du vocabulaire : {m!r}")
        got = tokenizer(m, add_special_tokens=False)["input_ids"]
        if len(got) != 1 or got[0] != tid:
            raise RuntimeError(
                f"marqueur {m!r} ne tokenise pas en un token unique : {got}")
        ids[m] = int(tid)

    probe = "ZZPROBEZZ"
    rendered = template.render(probe)
    if "system" in rendered:
        raise RuntimeError("le template contient encore un role 'system'")
    enc = tokenizer(rendered, add_special_tokens=False)["input_ids"]
    if bos_marker is not None:
        if bos_marker not in ids:
            raise RuntimeError(
                f"le marqueur de BOS {bos_marker!r} doit figurer dans "
                f"template.markers")
        if enc.count(ids[bos_marker]) != 1:
            raise RuntimeError(
                f"le BOS {bos_marker!r} doit apparaitre exactement une fois, "
                f"vu {enc.count(ids[bos_marker])} fois")
    if tokenizer.decode(enc).count(probe) != 1:
        raise RuntimeError("le texte utilisateur ne survit pas a l'aller-retour")
    return {"name": template.name, "marker_ids": ids,
            "bos_marker": bos_marker,
            "n_template_tokens": len(enc) - len(
                tokenizer(probe, add_special_tokens=False)["input_ids"])}
