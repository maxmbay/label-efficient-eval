# Judge auth (personal Modal + Hugging Face)

Gated models need Max's personal accounts (not work). Ungated open-IT
(`Qwen/Qwen2.5-1.5B-Instruct`) runs locally with `make judges-open-it` and does
not need a token.

## One-time setup

```bash
cd /Users/maxwellbay/projects/label-efficient-eval
uv sync --extra modal --extra dev

# Hugging Face
uv run huggingface-cli login
# Accept licenses in the browser:
#   https://huggingface.co/meta-llama/Llama-Guard-3-1B
#   https://huggingface.co/google/shieldgemma-2b   # optional

# Modal (personal)
uv run modal setup
# Create the HF secret Modal jobs read:
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
# Deploy / warm the app once:
uv run modal deploy src/lee/modal_apps/judges.py
```

## Score + cache

```bash
make judges-llama          # Llama Guard 3 1B on Modal -> data/cache/judges/
# optional:
make judges-shieldgemma    # local after license accept
make judges-open-it        # ungated local MPS
```

Then:

```bash
make m1-full && make figures
```

Cached scores are gitignored (`data/cache/`). Soft scores only; no raw prompts
in git.
