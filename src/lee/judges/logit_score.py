"""Soft scores from next-token logits (yes/no or safe/unsafe)."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_token_id(tokenizer: PreTrainedTokenizerBase, candidates: Sequence[str]) -> int:
    for tok in candidates:
        tid = tokenizer.convert_tokens_to_ids(tok)
        if tid is not None and tid != getattr(tokenizer, "unk_token_id", None):
            # Also accept encode without specials for plain words.
            return int(tid)
        encoded = tokenizer.encode(tok, add_special_tokens=False)
        if len(encoded) == 1:
            return int(encoded[0])
    raise KeyError(f"None of {candidates} found in tokenizer vocab")


def batch_positive_probs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    *,
    positive_tokens: Sequence[str],
    negative_tokens: Sequence[str],
    device: torch.device | None = None,
    max_length: int = 2048,
) -> np.ndarray:
    """P(positive) = softmax over {pos,neg} mass at the last prompt position.

    Restricting the softmax to the two label tokens avoids diluting the score
    across the full vocabulary and yields a calibrated-ish soft score in (0,1).
    """
    device = device or pick_device()
    pos_id = _resolve_token_id(tokenizer, positive_tokens)
    neg_id = _resolve_token_id(tokenizer, negative_tokens)

    enc = tokenizer(
        list(prompts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.inference_mode():
        out = model(**enc)
        # Last non-pad token per row
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        batch_idx = torch.arange(attn.size(0), device=device)
        logits = out.logits[batch_idx, last_idx]
        pair = torch.stack([logits[:, pos_id], logits[:, neg_id]], dim=-1)
        probs = torch.softmax(pair.float(), dim=-1)[:, 0]
    return probs.detach().cpu().numpy().astype(float)
