"""Prompt building and exact per-token role labelling.

The composite prompt is built here, so the boundary between the benign part
and the injected part is known by construction. Roles are then recovered from
the fast tokenizer's offset mapping rather than by aligning two tokenisations
against each other -- that alignment is what silently desynchronised labels
from attention in the earlier script.

**Binary mode.** The model receives the full prompt -- chat template, system
prompt, task instructions, the question -- because an injected instruction
only means anything when there is a task for it to hijack. But there are only
two roles:

    benign     everything legitimate: chat template, system prompt, task
               scaffolding, the question, and the host text       -> blue
    injection  tokens of the injected instruction                 -> red

``benign`` is the default fill, so any token the injected span does not claim
is blue. That keeps the figure strictly binary with no third bucket.

Two consequences worth carrying into the analysis, both of them consequences
of folding the scaffolding into ``benign`` rather than dropping it:

* the attention sink (BOS, ~22% of the attention mass) is now a *blue* node,
  so ``density_benign`` is inflated by a hub that is not host content;
* ``density_benign`` mixes scaffolding with email text, which are very
  different kinds of token.

``separation`` and the cut statistics should be read with that in mind.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import DEFAULT_SYSTEM_PROMPT

ROLE_BENIGN = "benign"
ROLE_INJECTION = "injection"

#: Retained so ``graphs.keep_mask(drop_template=...)`` and older result files
#: still resolve, but no token is ever labelled with it: the builders below
#: emit ``benign`` and ``injection`` exclusively.
ROLE_TEMPLATE = "template"

VALID_ROLES = (ROLE_BENIGN, ROLE_INJECTION)


@dataclass
class LabelledPrompt:
    """A prompt together with one role per token."""

    formatted: str
    tokens: List[str]
    roles: np.ndarray          # dtype '<U9', one entry per token
    input_ids: List[int]
    benign_text: str
    injection_text: Optional[str]

    def __len__(self) -> int:
        return len(self.tokens)

    def count(self, role: str) -> int:
        return int((self.roles == role).sum())

    def span(self, role: str) -> Optional[Tuple[int, int]]:
        """Half-open (start, end) index range covered by a role, or None.

        Contiguous by construction for a single injected segment; the hull
        otherwise.
        """
        idx = np.flatnonzero(self.roles == role)
        if idx.size == 0:
            return None
        return int(idx[0]), int(idx[-1]) + 1

    def summary(self) -> str:
        return (
            f"{len(self)} tokens "
            f"(benign={self.count(ROLE_BENIGN)}, "
            f"injection={self.count(ROLE_INJECTION)})"
        )


def format_chat(tokenizer, user_text: str, system_prompt: str) -> str:
    """Apply the model's chat template, with a plain fallback."""
    if getattr(tokenizer, "chat_template", None):
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]
        try:
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Some templates reject a system role; retry folded into the user turn.
            chat = [{"role": "user", "content": f"{system_prompt}\n\n{user_text}"}]
            return tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )
    return f"System: {system_prompt}\n\nUser: {user_text}\n\nAssistant:"


def _assign_roles(offsets: Sequence[Tuple[int, int]],
                  spans: Sequence[Tuple[int, int, str]],
                  n_tokens: int, user_start: int) -> np.ndarray:
    """One role per token, from character offsets into the formatted prompt.

    Everything starts ``benign`` and only the spans claim tokens away from it.
    So the chat template, the system prompt, BOS and the task scaffolding --
    none of which the segment list covers -- stay blue by default, which is
    exactly the binary colouring wanted here.

    ``user_start`` shifts the offsets into segment coordinates: the segments
    describe the user text, which sits somewhere inside the formatted string.

    A token is assigned by the midpoint of its character span, so one
    straddling a boundary goes to whichever side holds most of it. Zero-width
    spans (special tokens) fall back to their start offset.
    """
    roles = np.full(n_tokens, ROLE_BENIGN, dtype="<U9")
    for i, (lo, hi) in enumerate(offsets):
        mid = ((lo + hi) / 2.0 if hi > lo else float(lo)) - user_start
        for seg_lo, seg_hi, role in spans:
            if seg_lo <= mid < seg_hi:
                roles[i] = role
                break
    return roles


def build_labelled_prompt_from_segments(
    tokenizer,
    segments: List[Tuple[str, str]],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> LabelledPrompt:
    """Label a prompt assembled from ``(text, role)`` segments.

    The general form. Segments are concatenated in order and each token
    inherits the role of the segment it falls in, so the injected instruction
    can sit anywhere -- start, middle or end of the host content -- which is
    what BIPIA-style insertion requires.

    Roles must be ``benign`` or ``injection``: scaffolding has no role of its
    own any more, it is simply left at the ``benign`` default. The chat
    template and ``system_prompt`` are applied around the assembled segments,
    so the model sees a real task for the injection to hijack.
    """
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError(
            "A fast tokenizer is required for offset mapping. "
            "Load it with AutoTokenizer.from_pretrained(..., use_fast=True)."
        )
    bad = {r for _, r in segments} - set(VALID_ROLES)
    if bad:
        raise ValueError(
            f"unknown role(s): {sorted(bad)}; only {list(VALID_ROLES)} are "
            f"allowed. Task scaffolding is labelled '{ROLE_BENIGN}' like the "
            f"rest of the legitimate prompt, so the graph stays binary."
        )

    # Character spans of each segment inside the assembled user text.
    user_text, spans, cursor = "", [], 0
    for text, role in segments:
        user_text += text
        spans.append((cursor, cursor + len(text), role))
        cursor += len(text)
    if not user_text:
        raise ValueError("empty prompt: the segments produced no text")

    formatted = format_chat(tokenizer, user_text, system_prompt)
    # apply_chat_template often materialises BOS as text; tokenising that
    # string again would prepend a second BOS and shift every index.
    add_special = True
    bos = getattr(tokenizer, "bos_token", None)
    if bos and formatted.startswith(bos):
        add_special = False

    enc = tokenizer(formatted, return_offsets_mapping=True,
                    add_special_tokens=add_special)
    input_ids = list(enc["input_ids"])
    if not input_ids:
        raise RuntimeError("the tokenizer returned no token for this prompt")
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # rfind, not find: the host text can also appear inside the system prompt.
    user_start = formatted.rfind(user_text)
    if user_start < 0:
        raise RuntimeError(
            "User text not found verbatim in the formatted prompt; the chat "
            "template appears to rewrite content. Inspect format_chat()."
        )

    roles = _assign_roles(enc["offset_mapping"], spans, len(input_ids),
                          user_start)

    has_injection = any(r == ROLE_INJECTION for _, r in segments)
    if has_injection and not (roles == ROLE_INJECTION).any():
        raise RuntimeError(
            "No token was labelled 'injection'. Check that the injected "
            "segment is non-empty."
        )

    benign_text = "".join(t for t, r in segments if r == ROLE_BENIGN)
    injection_text = "".join(t for t, r in segments if r == ROLE_INJECTION)

    return LabelledPrompt(
        formatted=formatted,
        tokens=tokens,
        roles=roles,
        input_ids=input_ids,
        benign_text=benign_text,
        injection_text=injection_text if has_injection else None,
    )


def build_labelled_prompt(
    tokenizer,
    benign_text: str,
    injection_text: Optional[str] = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    separator: str = " ",
) -> LabelledPrompt:
    """Build a prompt and label every token by origin.

    The two-segment special case of
    :func:`build_labelled_prompt_from_segments`. ``injection_text=None`` gives
    the benign control: same host text, no injected instruction, every token
    labelled ``benign``.

    The separator between the two parts is charged to the host: it is a single
    space in practice, and in raw-text mode there is no neutral role left to
    put it in.
    """
    benign_text = benign_text.strip()
    segments: List[Tuple[str, str]] = [(benign_text, ROLE_BENIGN)]
    if injection_text is not None:
        injection_text = injection_text.strip()
        if separator:
            segments.append((separator, ROLE_BENIGN))
        segments.append((injection_text, ROLE_INJECTION))

    return build_labelled_prompt_from_segments(
        tokenizer, segments, system_prompt=system_prompt,
    )


def pretty_token(tok: str) -> str:
    """Strip tokenizer artefacts so labels are readable on a plot."""
    return (
        tok.replace("▁", "")   # SentencePiece
        .replace("Ġ", "")      # GPT-2 byte-BPE space
        .replace("Ċ", "\\n")   # GPT-2 byte-BPE newline
        .strip()
    )
