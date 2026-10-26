"""Llama Guard 3 soft scores: P(unsafe) from first-label-token logits."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lee.judges.auth import JudgeAuthError, LLAMA_GUARD_SETUP, require_hf_token
from lee.judges.logit_score import batch_positive_probs, pick_device

LLAMA_GUARD_3_1B = "meta-llama/Llama-Guard-3-1B"
LLAMA_GUARD_3_8B = "meta-llama/Llama-Guard-3-8B"


def _user_only_chat(text: str) -> list[dict]:
    return [{"role": "user", "content": [{"type": "text", "text": text}]}]


def build_llama_guard_prompts(
    texts: Sequence[str],
    tokenizer,
) -> list[str]:
    prompts: list[str] = []
    for t in texts:
        # Force next token to be safe/unsafe (skip leading whitespace).
        try:
            prompt = tokenizer.apply_chat_template(
                _user_only_chat(str(t)),
                tokenize=False,
                add_generation_prompt=False,
            )
        except Exception:
            # Fallback for tokenizers that want plain string content.
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": str(t)}],
                tokenize=False,
                add_generation_prompt=False,
            )
        if not prompt.endswith("\n\n"):
            prompt = prompt + "\n\n"
        prompts.append(prompt)
    return prompts


class LlamaGuard3Judge:
    """Local (MPS/CUDA/CPU) Llama Guard 3 scorer."""

    def __init__(
        self,
        model_id: str = LLAMA_GUARD_3_1B,
        *,
        batch_size: int = 4,
        max_length: int = 2048,
        name: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.name = name or (
            "llama_guard3_1b" if "1B" in model_id else "llama_guard3_8b"
        )
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        require_hf_token()
        self._device = pick_device()
        dtype = torch.float16 if self._device.type != "cpu" else torch.float32
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=dtype,
            ).to(self._device)
        except Exception as exc:
            raise JudgeAuthError(
                f"Failed to load {self.model_id}. "
                "Accept the HF license and ensure HF_TOKEN is set.\n"
                + LLAMA_GUARD_SETUP
                + f"\nUnderlying error: {exc}"
            ) from exc
        self._model.eval()
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

    def score(self, frame: pd.DataFrame) -> np.ndarray:
        self._load()
        assert self._model is not None and self._tokenizer is not None
        texts = frame["user_input"].astype(str).tolist()
        out = np.zeros(len(texts), dtype=float)
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            prompts = build_llama_guard_prompts(chunk, self._tokenizer)
            probs = batch_positive_probs(
                self._model,
                self._tokenizer,
                prompts,
                positive_tokens=["unsafe"],
                negative_tokens=["safe"],
                device=self._device,
                max_length=self.max_length,
            )
            out[i : i + len(chunk)] = probs
        return out
