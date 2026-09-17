"""Length-matched insertion controls (``--compare control``).

An injected prompt differs from its benign twin in two ways at once: *some*
text was inserted, and that text is an instruction to the model. A control
insertion keeps the first and removes the second, so what the attention graph
does under both is length/insertion, and what it does only under injection is
content.

``email_text``: sentences from the body of *another* BIPIA email, with exactly
as many model tokens as the attack, inserted where the attack was. The three
ways such a control can quietly fail are handled explicitly:

cloning       The host email is removed from the draw -- and so is every
              candidate sharing a 6-word run (digits masked) with it, because
              BIPIA holds many near-identical templated emails (Mercury card
              notifications differing only by amount) that exact-text
              exclusion would let through.
format        Everything up to ``CONTENT:`` is cut, so header fields
              (``SUBJECT:``, ``EMAIL_FROM:`` ...) are never drawn. Emails with
              no ``CONTENT:`` marker carry no header at all and are kept whole;
              any sentence still containing a header tag is dropped.
false length  Length is counted with the model's own tokenizer, never in
              characters. A window of consecutive sentences with exactly the
              attack's token count is preferred; otherwise a longer one is cut
              at the last whole word that makes the counts equal.
"""

import random
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .bipia import BipiaPair, _sentence_starts
from .prompts import ROLE_INJECTION

HEADER_TAGS = ("SUBJECT:", "EMAIL_FROM:", "RECEIVED DATE:", "CONTENT:")
NGRAM = 6
MAX_WINDOW = 8          # sentences per candidate window


@dataclass
class ControlText:
    text: str
    donor_index: int        # index of the donor email in the split's file
    n_tokens_attack: int
    n_tokens_control: int
    truncated: bool
    n_candidates: int       # windows left after exclusion
    exact: bool = True      # in-prompt target (tokens, and total length) met
    n_chars_attack: int = 0
    n_chars_control: int = 0

    def label(self) -> str:
        eq = "=" if self.exact else "≠"
        return (f"contrôle email #{self.donor_index} · "
                f"{self.n_tokens_control} tok {eq} attaque "
                f"{self.n_tokens_attack} tok · {self.n_chars_control} car. vs "
                f"{self.n_chars_attack}"
                + (" · tronqué" if self.truncated else ""))


def email_body(context: str) -> str:
    """The natural-language body: everything after the first ``CONTENT:``."""
    if "CONTENT:" in context:
        return context.split("CONTENT:", 1)[1]
    return context


def _norm_words(text: str) -> List[str]:
    return re.findall(r"[a-z]+|0", re.sub(r"\d+", "0", text.lower()))


def _ngrams(text: str) -> set:
    w = _norm_words(text)
    return {tuple(w[i:i + NGRAM]) for i in range(len(w) - NGRAM + 1)}


def _sentences(body: str) -> List[str]:
    starts = sorted({0, *(s for s in _sentence_starts(body)
                          if 0 <= s < len(body))})
    out = []
    for a, b in zip(starts, starts[1:] + [len(body)]):
        s = " ".join(body[a:b].split())          # newlines/indent -> spaces
        if len(s.split()) < 3 or not re.search(r"[A-Za-z]{2}", s):
            continue
        if any(tag in s for tag in HEADER_TAGS):
            continue
        out.append(s)
    return out


class EmailTextControl:
    """Sentence-window pool over one split, token counts cached once."""

    def __init__(self, emails: Sequence[Dict], tokenizer, seed: int = 42):
        self.tokenizer = tokenizer
        self.seed = seed
        self.contexts = [e["context"] for e in emails]
        self.windows: List[Tuple[int, str, int]] = []   # (email idx, text, tokens)
        for idx, e in enumerate(emails):
            sents = _sentences(email_body(e["context"]))
            for i in range(len(sents)):
                for j in range(i + 1, min(i + MAX_WINDOW, len(sents)) + 1):
                    text = " ".join(sents[i:j])
                    self.windows.append((idx, text, self.count(text)))

    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    @staticmethod
    def _truncate(text: str, n: int, count: Callable[[str], int]
                  ) -> Optional[str]:
        """Longest whole-word prefix with exactly ``n`` tokens, if one exists."""
        ends = [m.end() for m in re.finditer(r"\S+", text)]
        lo, hi, best = 0, len(ends) - 1, -1
        while lo <= hi:                     # last prefix with count <= n
            mid = (lo + hi) // 2
            if count(text[:ends[mid]]) <= n:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        if best < 0:
            return None
        cut = text[:ends[best]]
        return cut if count(cut) == n else None

    def choose(self, pair: BipiaPair, pair_index: int,
               in_prompt: Optional[Callable[[str], object]] = None,
               n_target: object = None, match_chars: bool = False,
               budget: int = 120) -> ControlText:
        """Pick the control text for one pair.

        Tokens are counted where they matter: inside the assembled prompt.
        Standalone and in-prompt counts can differ by one -- a sub-word merge
        with the newline or text next to the insertion point -- so with
        ``in_prompt`` and ``n_target`` the match is exact in the prompt the
        model actually reads. Standalone counts only preselect candidates.

        ``in_prompt`` may return an int (inserted tokens) or a tuple whose
        first item is the inserted tokens -- ``(inserted, total prompt
        length)`` -- in which case ``n_target`` is the same tuple and the
        whole tuple must match. Requiring the total length as well removes the
        one-token shift a retokenised junction can cause, which would move
        the question and answer positions between the two prompts.

        ``match_chars``: equal token counts can still hide a character-length
        gap (attacks are more fragmented: shorter words, capitals, symbols).
        Instead of the first exact match, up to ``budget`` candidates are
        examined and, among those meeting the token target, the one whose
        character count is closest to the attack's is kept.
        """
        count = in_prompt or self.count
        target = n_target if n_target is not None else self.count(pair.attack)
        n = target[0] if isinstance(target, tuple) else target

        def inserted(text):
            c = count(text)
            return c[0] if isinstance(c, tuple) else c

        host_grams = _ngrams(email_body(pair.context))
        pool = [(i, t, k) for i, t, k in self.windows
                if self.contexts[i] != pair.context
                and not (_ngrams(t) & host_grams)]
        if not pool:
            raise RuntimeError("aucune phrase de contrôle hors de l'email hôte")
        rng = random.Random(self.seed * 100003 + pair_index)
        n_chars = len(pair.attack)
        want = 10 ** 9 if match_chars else 1
        found: List[Tuple[int, str, bool]] = []
        tried = 0

        near = [w for w in pool if abs(w[2] - n) <= 2]
        rng.shuffle(near)
        near.sort(key=lambda w: abs(w[2] - n))     # stable: closest first
        for i, t, _ in near:
            if len(found) >= want or tried >= budget:
                break
            tried += 1
            if count(t) == target:
                found.append((i, t, False))

        longer = [w for w in pool if w[2] > n]
        rng.shuffle(longer)
        longer.sort(key=lambda w: w[2])            # stable: shortest first
        for i, t, _ in longer:
            if len(found) >= want or tried >= budget:
                break
            tried += 1
            cut = self._truncate(t, n, inserted)
            if cut is not None and count(cut) == target:
                found.append((i, cut, True))

        if found:
            # Whole sentences first: a text cut mid-sentence is its own
            # anomaly, so truncation only wins when no untruncated candidate
            # is within 10 % of the attack's characters. min is stable, so
            # ties keep the seeded shuffle order.
            whole = [f for f in found if not f[2]
                     and abs(len(f[1]) - n_chars) <= 0.10 * max(n_chars, 1)]
            i, t, trunc = min(whole or found,
                              key=lambda f: abs(len(f[1]) - n_chars))
            return ControlText(t, i, n, n, trunc, len(pool), True,
                               n_chars, len(t))

        # No exact match reachable (every word boundary over- or undershoots
        # by a sub-word token): closest available, and the label shows the gap.
        i, t, _ = min(pool, key=lambda w: (abs(w[2] - n), -w[2]))
        return ControlText(t, i, n, inserted(t), False, len(pool), False,
                           n_chars, len(t))


def control_segments(pair: BipiaPair, text: str) -> List[Tuple[str, str]]:
    """The injected prompt with the attack swapped for ``text``.

    Built from ``injected_segments`` rather than re-running the insertion, so
    the split point, newlines and scaffolding are guaranteed identical. The
    inserted span keeps the ``injection`` role only so the existing machinery
    colours and reorders it; the figure labels it as a control.
    """
    out, swapped = [], 0
    for seg, role in pair.injected_segments:
        if role == ROLE_INJECTION:
            out.append((text, role))
            swapped += 1
        else:
            out.append((seg, role))
    if swapped != 1:
        raise RuntimeError(f"{swapped} segments injectés au lieu d'un seul")
    return out
