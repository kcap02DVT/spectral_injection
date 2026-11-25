"""Rendu par le template OFFICIEL du modele, regions par offsets, capture.

Contrairement a la campagne OPI, le template officiel est utilise ici, par
decision du protocole. Son comportement est donc **documente et non
suppose** : ``render`` rend la chaine exacte, et ``regions`` attribue chaque
token a S (systeme), U (utilisateur) ou T (appareil de template) par le
milieu de son intervalle de caracteres, sur le graphe complet.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

REGION_S, REGION_U, REGION_T = "S", "U", "T"


def render(tok, system: Optional[str], user: str) -> str:
    """``system=None`` -> aucun message systeme (le template peut en
    injecter un par defaut) ; ``""`` -> bloc systeme vide ; sinon explicite."""
    msgs = ([] if system is None else [{"role": "system", "content": system}])
    msgs.append({"role": "user", "content": user})
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def default_system_injected(tok) -> Tuple[bool, str]:
    """Le template injecte-t-il un systeme quand on n'en passe aucun ?"""
    probe = "ZZPROBEZZ"
    none = render(tok, None, probe)
    empty = render(tok, "", probe)
    if none == empty:
        return False, ""
    # extraire le contenu systeme injecte
    a = none.find("system\n") + len("system\n")
    b = none.find("<|im_end|>", a)
    return True, none[a:b] if (a > 6 and b > a) else "<non extrait>"


@dataclass
class Prompt:
    text: str
    input_ids: List[int]
    tokens: List[str]
    regions: np.ndarray
    offsets: List[Tuple[int, int]]
    meta: Dict = field(default_factory=dict)

    def __len__(self):
        return len(self.input_ids)

    def mask(self, r: str) -> np.ndarray:
        return self.regions == r


def _anchor(text: str, role: str, content: str) -> Tuple[int, int]:
    """Intervalle de caracteres du contenu d'un role, ancre par ses
    marqueurs ChatML pour garantir l'unicite."""
    pat = f"<|im_start|>{role}\n{content}<|im_end|>"
    i = text.find(pat)
    if i < 0:
        raise RuntimeError(f"contenu du role {role!r} introuvable verbatim ; "
                           f"le template a reecrit le contenu")
    if text.find(pat, i + 1) >= 0:
        raise RuntimeError(f"contenu du role {role!r} ambigu (plusieurs occurrences)")
    lo = i + len(f"<|im_start|>{role}\n")
    return lo, lo + len(content)


def segment(tok, system: Optional[str], user: str) -> Prompt:
    """Prompt formate + region de chaque token. Le systeme effectivement
    present (y compris celui injecte par le template) est detecte dans le
    texte rendu, jamais suppose."""
    text = render(tok, system, user)
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offs = [tuple(o) for o in enc["offset_mapping"]]
    regions = np.full(len(ids), REGION_T, dtype="<U1")

    u_lo, u_hi = _anchor(text, "user", user)
    # systeme effectif : ce que le texte rendu contient entre les marqueurs
    s_span = None
    si = text.find("<|im_start|>system\n")
    if si >= 0:
        a = si + len("<|im_start|>system\n")
        b = text.find("<|im_end|>", a)
        if b > a:
            s_span = (a, b)
    for i, (lo, hi) in enumerate(offs):
        mid = (lo + hi) / 2.0 if hi > lo else float(lo)
        if u_lo <= mid < u_hi:
            regions[i] = REGION_U
        elif s_span and s_span[0] <= mid < s_span[1]:
            regions[i] = REGION_S
    if (regions == REGION_U).sum() == 0:
        raise RuntimeError("aucun token attribue a U")
    iu = np.flatnonzero(regions == REGION_U)
    is_ = np.flatnonzero(regions == REGION_S)
    if is_.size and is_.max() >= iu.min():
        raise RuntimeError("S doit entierement preceder U")
    return Prompt(text=text, input_ids=ids,
                  tokens=tok.convert_ids_to_tokens(ids), regions=regions,
                  offsets=offs,
                  meta={"n_S": int((regions == "S").sum()),
                        "n_U": int((regions == "U").sum()),
                        "n_T": int((regions == "T").sum()),
                        "system_effective": text[s_span[0]:s_span[1]] if s_span else ""})


class Capturer:
    """Charge le modele une fois (eager, float32), rend l'attention brute."""

    def __init__(self, model_id: str, max_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32,
            attn_implementation="eager").to(self.device).eval()
        c = self.model.config
        self.n_layers = int(c.num_hidden_layers)
        self.n_heads = int(c.num_attention_heads)
        self.n_kv_heads = int(getattr(c, "num_key_value_heads", self.n_heads))

    def forward(self, p: Prompt) -> List[np.ndarray]:
        """Une passe, sans generation -> [couches][H, T, T] float32."""
        import torch
        if len(p) > self.max_tokens:
            raise ValueError(f"{len(p)} tokens > max_tokens={self.max_tokens}")
        ids = torch.tensor([p.input_ids], device=self.device)
        with torch.no_grad():
            out = self.model(ids, output_attentions=True)
        att = [a[0].detach().cpu().float().numpy() for a in out.attentions]
        del out
        self._verify(att, len(p))
        return att

    def _verify(self, att: Sequence[np.ndarray], n: int, tol: float = 1e-4):
        if len(att) != self.n_layers:
            raise RuntimeError(f"{len(att)} couches au lieu de {self.n_layers}")
        for l, A in enumerate(att):
            if A.shape != (self.n_heads, n, n):
                raise RuntimeError(f"couche {l} : forme {A.shape}")
            if not np.isfinite(A).all():
                raise RuntimeError(f"couche {l} : valeurs non finies")
            if np.abs(np.triu(A, 1)).max() > tol:
                raise RuntimeError(f"couche {l} : masque causal viole")
            if np.abs(A.sum(-1) - 1.0).max() > 1e-3:
                raise RuntimeError(f"couche {l} : lignes non stochastiques")

    def generate(self, p: Prompt, max_new_tokens: int, do_sample: bool) -> str:
        import torch
        ids = torch.tensor([p.input_ids], device=self.device)
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new_tokens,
                                      do_sample=do_sample,
                                      pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)


def save_attention(path: str, att: Sequence[np.ndarray]) -> int:
    """npz compresse, une cle par couche. Rend la taille en octets."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **{f"layer_{l:02d}": A for l, A in enumerate(att)})
    return os.path.getsize(path)


def load_attention(path: str) -> List[np.ndarray]:
    d = np.load(path)
    return [d[k] for k in sorted(d.files)]
