"""Model loading and attention capture.

Kept deliberately thin: it turns a formatted prompt into a list of
``[heads, seq, seq]`` numpy arrays, one per layer, and nothing else. Every
graph and metric computation downstream works on numpy, so the analysis can be
unit-tested without a GPU or a model.
"""

from typing import List, Tuple

import numpy as np

from .config import RunConfig
from .prompts import (
    LabelledPrompt,
    build_labelled_prompt,
    build_labelled_prompt_from_segments,
)


class AttentionRunner:
    """Loads a causal LM once and extracts attention maps per prompt."""

    def __init__(self, config: RunConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.config = config
        if config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = config.device

        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id, use_fast=True)

        dtype = torch.float16 if (self.device == "cuda" and
                                  config.dtype == "float16") else torch.float32
        # eager is mandatory: the sdpa and flash kernels do not return
        # attention weights, so output_attentions silently yields None.
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()

        self.n_layers = int(self.model.config.num_hidden_layers)

    # ------------------------------------------------------------------
    def middle_layer(self) -> int:
        return self.n_layers // 2

    def analyse_segments(self, segments) -> Tuple[LabelledPrompt, List[np.ndarray]]:
        """Label a prompt assembled from (text, role) segments, then run it.

        The general form, used by the BIPIA pipeline where the injected
        instruction sits inside the host content rather than after it.
        """
        labelled = build_labelled_prompt_from_segments(
            self.tokenizer, segments, system_prompt=self.config.system_prompt,
        )
        return labelled, self._forward(labelled)

    def analyse(self, benign_text: str, injection_text=None
                ) -> Tuple[LabelledPrompt, List[np.ndarray]]:
        """Label a prompt and run one forward pass over it."""
        labelled = build_labelled_prompt(
            self.tokenizer, benign_text, injection_text,
            system_prompt=self.config.system_prompt,
        )
        return labelled, self._forward(labelled)

    def _forward(self, labelled: LabelledPrompt) -> List[np.ndarray]:
        import torch

        if len(labelled) > self.config.max_tokens:
            raise ValueError(
                f"Prompt is {len(labelled)} tokens, over max_tokens="
                f"{self.config.max_tokens}. Attention is O(T^2) per head; "
                f"raise --max-tokens deliberately."
            )

        ids = torch.tensor([labelled.input_ids], device=self.device)
        with torch.no_grad():
            out = self.model(ids, output_attentions=True)

        attentions = [a[0].detach().cpu().float().numpy() for a in out.attentions]
        if not attentions:
            raise RuntimeError(
                "No attention returned. The model was not loaded with "
                "attn_implementation='eager'."
            )
        n_tok = attentions[0].shape[-1]
        if n_tok != len(labelled):
            raise RuntimeError(
                f"Token/attention mismatch: {len(labelled)} labels vs {n_tok} "
                f"attention positions. Do not truncate to paper over this."
            )
        return attentions
