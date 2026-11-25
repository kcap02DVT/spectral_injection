"""Prompt sans message systeme, segmentation U/E/T, capture de l'attention.

Ce module ne passe PAS par ``prompts.format_chat`` : celui-ci insere
toujours un message systeme, ce que le protocole interdit pour cette
campagne. Le prompt est un unique tour utilisateur.

La segmentation reprend le principe de ``prompts._assign_roles`` -- un token
appartient a la region qui contient le milieu de son span de caracteres --
generalise a trois regions au lieu de deux :

    U  les tokens de l'instruction legitime
    E  les tokens du contenu externe
    T  tout le reste : tokens speciaux du chat template, separateurs,
       en-tete de generation, et le BOS

Les frontieres viennent de la construction du prompt et des offsets du
tokenizer rapide, jamais d'une recherche de chaine a posteriori sur le texte
tokenise, et jamais des labels d'attaque.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REGION_U = "U"
REGION_E = "E"
REGION_T = "T"
REGIONS = (REGION_U, REGION_E, REGION_T)


@dataclass
class Segmented:
    """Un prompt tokenise, avec ses regions."""
    text: str                       # le prompt formate complet
    input_ids: List[int]
    tokens: List[str]
    regions: np.ndarray             # [T] valeurs dans REGIONS
    offsets: List[Tuple[int, int]]
    user_text: str
    user_start: int                 # offset du tour utilisateur dans `text`
    meta: Dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.input_ids)

    def mask(self, region: str) -> np.ndarray:
        return self.regions == region

    def count(self, region: str) -> int:
        return int((self.regions == region).sum())

    def summary(self) -> str:
        return (f"T={len(self)} | U={self.count('U')} E={self.count('E')} "
                f"T_tech={self.count('T')}")


def build_user_turn(instruction: str, external: str) -> str:
    """``(U + "\\n\\n" + E).rstrip()``.

    Le ``rstrip`` n'est pas cosmetique : le chat template de Llama-3 applique
    ``| trim`` au contenu du message, et 42,9 % des lignes d'OPI finissent
    par une espace ou un retour a la ligne. Sans lui, le texte utilisateur
    n'apparait pas verbatim dans le prompt formate et la segmentation par
    offsets devient fausse.
    """
    return f"{instruction}\n\n{external}".rstrip()


@dataclass(frozen=True)
class ExperimentalTemplate:
    """Format conversationnel explicite et deterministe.

    Le template officiel de Llama-3.2 **fabrique un bloc systeme** meme
    lorsqu'on ne lui passe qu'un message utilisateur :

        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        Cutting Knowledge Date: December 2023
        Today Date: 09 Nov 2026
        <|eot_id|><|start_header_id|>user<|end_header_id|>...

    Vingt-cinq tokens que personne n'a demandes, et dont la date **change
    chaque jour** : deux runs a deux dates differentes ne sont pas
    comparables. Le protocole interdit d'ajouter un message systeme, et un
    message systeme vide n'est pas une solution -- il laisserait les
    marqueurs de role.

    On declare donc le format minimal, sans bloc systeme. Il est fige dans le
    code, verifie contre le vocabulaire du tokenizer, et enregistre dans la
    provenance.
    """

    name: str
    prefix: str
    suffix: str
    markers: Tuple[str, ...]

    def render(self, user_text: str) -> str:
        return self.prefix + user_text + self.suffix


#: Format conversationnel de Llama-3.x, message utilisateur seul.
LLAMA3_USER_ONLY = ExperimentalTemplate(
    name="llama3-user-only-nosystem",
    prefix="<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n",
    suffix="<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    markers=("<|begin_of_text|>", "<|start_header_id|>", "<|end_header_id|>",
             "<|eot_id|>"),
)


def verify_template(tokenizer, template: ExperimentalTemplate) -> Dict:
    """Controle le template contre le vocabulaire reel du tokenizer.

    Leve si un marqueur n'existe pas, s'il ne tokenise pas en un token
    unique, si un role ``system`` subsiste, ou si le texte utilisateur ne
    survit pas a un aller-retour.
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
    if enc.count(ids["<|begin_of_text|>"]) != 1:
        raise RuntimeError("le BOS doit apparaitre exactement une fois")
    if tokenizer.decode(enc).count(probe) != 1:
        raise RuntimeError("le texte utilisateur ne survit pas a l'aller-retour")
    return {"name": template.name, "marker_ids": ids,
            "n_template_tokens": len(enc) - len(
                tokenizer(probe, add_special_tokens=False)["input_ids"])}


def format_user_only(tokenizer, user_text: str,
                     template: ExperimentalTemplate = LLAMA3_USER_ONLY) -> str:
    """Rend le prompt selon le template experimental declare."""
    return template.render(user_text)


def segment(tokenizer, instruction: str, external: str,
            template: ExperimentalTemplate = LLAMA3_USER_ONLY) -> Segmented:
    """Construit le prompt et attribue une region a chaque token."""
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("tokenizer rapide requis (offset mapping)")

    user = build_user_turn(instruction, external)
    text = format_user_only(tokenizer, user, template)

    start = text.find(user)
    if start < 0:
        raise RuntimeError(
            "le tour utilisateur n'apparait pas verbatim dans le prompt "
            "formate : le chat template reecrit le contenu. Verifier le "
            "rstrip et l'absence de normalisation.")
    if text.find(user, start + 1) >= 0:
        raise RuntimeError("le tour utilisateur apparait plusieurs fois ; "
                           "l'ancrage par offset serait ambigu")

    enc = tokenizer(text, add_special_tokens=False,
                    return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    toks = tokenizer.convert_ids_to_tokens(ids)

    # spans en coordonnees du prompt formate
    u_lo, u_hi = start, start + len(instruction)
    e_lo, e_hi = start + len(user) - len(external.rstrip()), start + len(user)
    # verification : le decoupage U | separateur | E doit recouvrir `user`
    if not text[u_lo:u_hi] == instruction:
        raise RuntimeError("ancrage de U incorrect")
    if not text[e_lo:e_hi] == external.rstrip():
        raise RuntimeError("ancrage de E incorrect")

    regions = np.full(len(ids), REGION_T, dtype="<U1")
    for i, (lo, hi) in enumerate(offs):
        mid = (lo + hi) / 2.0 if hi > lo else float(lo)
        if u_lo <= mid < u_hi:
            regions[i] = REGION_U
        elif e_lo <= mid < e_hi:
            regions[i] = REGION_E

    if regions.tolist().count(REGION_E) == 0:
        raise RuntimeError("aucun token attribue a E")
    if regions.tolist().count(REGION_U) == 0:
        raise RuntimeError("aucun token attribue a U")

    # U doit preceder E : le protocole en depend (F_{U->E} = 0 par causalite)
    iu = np.flatnonzero(regions == REGION_U)
    ie = np.flatnonzero(regions == REGION_E)
    if iu.max() >= ie.min():
        raise RuntimeError(
            f"U et E se chevauchent : max(U)={iu.max()} >= min(E)={ie.min()}")

    return Segmented(text=text, input_ids=ids, tokens=toks, regions=regions,
                     offsets=offs, user_text=user, user_start=start,
                     meta={"n_U": int((regions == REGION_U).sum()),
                           "n_E": int((regions == REGION_E).sum()),
                           "n_T": int((regions == REGION_T).sum()),
                           "u_span": (u_lo, u_hi), "e_span": (e_lo, e_hi)})


def verify_segmentation(seg: Segmented) -> Dict:
    """Audit de la segmentation, au-dela du simple comptage de tokens.

    Le nombre de tokens par region ne prouve rien : il faut verifier que les
    offsets couvrent le texte sans trou ni recouvrement, que les tokens
    attribues a U et a E reconstituent bien le texte de ces regions, et
    savoir exactement combien de caracteres de chaque token frontiere
    tombent de part et d'autre.

    Rend un dictionnaire de controles ; aucun n'est tolere silencieusement
    par l'appelant -- ``run`` les imprime et signale toute anomalie.
    """
    offs = seg.offsets
    u_lo, u_hi = seg.meta["u_span"]
    e_lo, e_hi = seg.meta["e_span"]
    out: Dict = {}

    # 1. les offsets sont croissants et sans recouvrement
    gaps = overlaps = 0
    prev = 0
    for lo, hi in offs:
        if hi > lo:
            if lo > prev:
                gaps += 1
            elif lo < prev:
                overlaps += 1
            prev = max(prev, hi)
    out["offset_gaps"] = gaps
    out["offset_overlaps"] = overlaps
    out["offset_covers_end"] = bool(prev == len(seg.text))

    # 2. les tokens de U (resp. E) reconstituent-ils le texte de la region ?
    def joined(region):
        idx = np.flatnonzero(seg.regions == region)
        if idx.size == 0:
            return ""
        lo = offs[idx[0]][0]
        hi = max(offs[i][1] for i in idx)
        return seg.text[lo:hi]

    u_txt, e_txt = joined(REGION_U), joined(REGION_E)
    out["U_recovered"] = u_txt
    out["E_recovered"] = e_txt
    out["U_matches_span"] = bool(u_txt == seg.text[u_lo:u_hi])
    out["E_matches_span"] = bool(e_txt == seg.text[e_lo:e_hi])
    # un ecart attendu : le dernier token de U peut fusionner avec le
    # separateur et basculer en T, auquel cas U perd sa ponctuation finale
    out["U_missing_tail"] = seg.text[u_lo:u_hi][len(u_txt):] if \
        seg.text[u_lo:u_hi].startswith(u_txt) else None

    # 3. tokens frontieres : combien de caracteres de part et d'autre
    bnd = []
    for i, (lo, hi) in enumerate(offs):
        if hi <= lo:
            continue
        for name, (a, b) in (("U", (u_lo, u_hi)), ("E", (e_lo, e_hi))):
            inside = max(0, min(hi, b) - max(lo, a))
            if 0 < inside < (hi - lo):
                bnd.append({"index": i, "token": seg.tokens[i],
                            "region_assigned": str(seg.regions[i]),
                            "overlaps": name, "chars_in": inside,
                            "chars_total": hi - lo,
                            "share_in": round(inside / (hi - lo), 3)})
    out["boundary_tokens"] = bnd
    out["n_boundary_tokens"] = len(bnd)

    # 4. ordre des regions et contiguite de chaque bloc
    for r in (REGION_U, REGION_E):
        idx = np.flatnonzero(seg.regions == r)
        out[f"{r}_contiguous"] = bool(
            idx.size > 0 and idx.size == idx[-1] - idx[0] + 1)
    iu = np.flatnonzero(seg.regions == REGION_U)
    ie = np.flatnonzero(seg.regions == REGION_E)
    out["U_before_E"] = bool(iu.size and ie.size and iu.max() < ie.min())
    out["ok"] = bool(out["offset_overlaps"] == 0
                     and out["offset_covers_end"]
                     and out["U_before_E"]
                     and out["U_contiguous"] and out["E_contiguous"])
    return out


# --------------------------------------------------------------------------
# Capture
# --------------------------------------------------------------------------
class Capturer:
    """Charge le modele une fois, rend l'attention brute par couche."""

    def __init__(self, model_id: str, device: str = "auto",
                 dtype: str = "float32", max_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = ("cuda" if (device == "auto" and torch.cuda.is_available())
                       else ("cpu" if device == "auto" else device))
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        td = torch.float16 if (self.device == "cuda" and dtype == "float16") \
            else torch.float32
        # eager obligatoire : sdpa et flash ne renvoient pas les poids
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=td, attn_implementation="eager").to(self.device)
        self.model.eval()
        cfg = self.model.config
        self.n_layers = int(cfg.num_hidden_layers)
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads", self.n_heads))

    def forward(self, seg: Segmented, check: bool = True,
                with_hidden: bool = False):
        """Une passe avant, sans generation. Rend [couches][H, T, T].

        ``with_hidden=True`` rend en plus les etats caches ALIGNES sur
        l'attention : la sortie ``hs[l]`` est ``hidden_states[l + 1]``,
        c'est-a-dire la sortie du bloc dont ``att[l]`` est l'attention. Le
        decalage est fait ici, une seule fois, pour qu'aucun appelant n'ait
        a le refaire -- ``hidden_states[0]`` est l'embedding d'entree et
        n'est l'attention d'aucune couche.

        Ces etats caches sortent du cadre attention-only : toute metrique
        qui les utilise est HYBRIDE et doit etre rapportee comme telle.
        """
        import torch

        if len(seg) > self.max_tokens:
            raise ValueError(f"{len(seg)} tokens > max_tokens={self.max_tokens} ; "
                             f"l'attention est en O(T^2) par tete")
        ids = torch.tensor([seg.input_ids], device=self.device)
        with torch.no_grad():
            out = self.model(ids, output_attentions=True,
                             output_hidden_states=with_hidden)
        if not out.attentions:
            raise RuntimeError("aucune attention : attn_implementation='eager' ?")
        att = [a[0].detach().cpu().float().numpy() for a in out.attentions]
        hs = None
        if with_hidden:
            if not getattr(out, "hidden_states", None):
                raise RuntimeError("aucun etat cache malgre "
                                   "output_hidden_states=True")
            if len(out.hidden_states) != len(att) + 1:
                raise RuntimeError(
                    f"{len(out.hidden_states)} etats caches pour {len(att)} "
                    f"couches d'attention ; attendu {len(att) + 1} "
                    f"(embedding d'entree + une sortie par bloc)")
            hs = [h[0].detach().cpu().float().numpy()
                  for h in out.hidden_states[1:]]
        del out

        if check:
            self.verify(att, len(seg))
            if hs is not None:
                self.verify_hidden(hs, len(seg))
        return (att, hs) if with_hidden else att

    def verify_hidden(self, hs: Sequence[np.ndarray], n_tokens: int) -> None:
        """Dimensions, finitude, et croissance de la norme du flux residuel.

        La norme n'est pas controlee contre un seuil -- elle croit
        legitimement avec la profondeur. Elle est seulement exigee non
        nulle : un etat cache nul rendrait l'indice de lissage nul par le
        garde-fou de la bibliotheque, ce qui se lirait a tort comme un
        graphe parfaitement lisse.
        """
        if len(hs) != self.n_layers:
            raise RuntimeError(f"{len(hs)} etats caches au lieu de {self.n_layers}")
        for l, X in enumerate(hs):
            if X.ndim != 2 or X.shape[0] != n_tokens:
                raise RuntimeError(
                    f"couche {l} : forme {X.shape}, attendu ({n_tokens}, d)")
            if not np.isfinite(X).all():
                raise RuntimeError(f"couche {l} : etats caches non finis")
            if float(np.sum(X * X)) <= 1e-8:
                raise RuntimeError(f"couche {l} : etats caches de norme nulle")

    def verify(self, att: Sequence[np.ndarray], n_tokens: int,
               tol: float = 1e-4) -> None:
        """Masque causal, sommes de lignes, finitude, dimensions."""
        if len(att) != self.n_layers:
            raise RuntimeError(f"{len(att)} couches au lieu de {self.n_layers}")
        for l, A in enumerate(att):
            if A.shape != (self.n_heads, n_tokens, n_tokens):
                raise RuntimeError(
                    f"couche {l} : forme {A.shape}, attendu "
                    f"{(self.n_heads, n_tokens, n_tokens)}")
            if not np.isfinite(A).all():
                raise RuntimeError(f"couche {l} : valeurs non finies")
            up = np.triu(A, 1)
            if np.abs(up).max() > tol:
                raise RuntimeError(
                    f"couche {l} : masque causal viole, max superieur "
                    f"diagonale = {np.abs(up).max():.2e}")
            s = A.sum(-1)
            if np.abs(s - 1.0).max() > tol:
                raise RuntimeError(
                    f"couche {l} : lignes ne sommant pas a 1 "
                    f"(ecart max {np.abs(s - 1.0).max():.2e})")
