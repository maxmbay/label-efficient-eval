"""Modal GPU jobs for gated safety judges (personal Modal account).

Deploy / run:
  uv run modal run src/lee/modal_apps/judges.py::score_llama_guard
"""

from __future__ import annotations

import modal

app = modal.App("lee-m1-judges")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch",
        "transformers>=4.43.0",
        "accelerate",
        "huggingface_hub",
        "numpy",
        "pandas",
        "pyarrow",
        "sentencepiece",
        "protobuf",
    )
)

# Secrets: create with
#   modal secret create huggingface-secret HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 3,
    memory=32768,
    secrets=[hf_secret],
)
def score_texts_llama_guard(
    texts: list[str],
    model_id: str = "meta-llama/Llama-Guard-3-1B",
    batch_size: int = 8,
) -> list[float]:
    """Return P(unsafe) for each user prompt on Modal GPU."""
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def prompts_for(chunk: list[str]) -> list[str]:
        out = []
        for t in chunk:
            try:
                p = tokenizer.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": t}]}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception:
                p = tokenizer.apply_chat_template(
                    [{"role": "user", "content": t}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
            if not p.endswith("\n\n"):
                p = p + "\n\n"
            out.append(p)
        return out

    unsafe_id = tokenizer.convert_tokens_to_ids("unsafe")
    safe_id = tokenizer.convert_tokens_to_ids("safe")
    scores: list[float] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        prompts = prompts_for(chunk)
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(device)
        with torch.inference_mode():
            logits = model(**enc).logits
            last = enc["attention_mask"].sum(dim=1) - 1
            idx = torch.arange(logits.size(0), device=device)
            row = logits[idx, last]
            pair = torch.stack([row[:, unsafe_id], row[:, safe_id]], dim=-1)
            probs = torch.softmax(pair.float(), dim=-1)[:, 0]
        scores.extend(probs.detach().cpu().numpy().astype(float).tolist())
    # silence unused imports if type-checkers complain
    _ = (np, pd)
    return scores


@app.local_entrypoint()
def score_llama_guard(
    texts_json: str = "[]",
    model_id: str = "meta-llama/Llama-Guard-3-1B",
) -> None:
    import json

    texts = json.loads(texts_json)
    scores = score_texts_llama_guard.remote(texts, model_id=model_id)
    print(json.dumps(scores[:5]))
    print(f"n={len(scores)}")
