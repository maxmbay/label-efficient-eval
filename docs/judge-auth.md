# Judge auth (personal Modal + Hugging Face)

Gated models need Max's personal accounts (not work). Ungated open-IT
(`Qwen/Qwen2.5-1.5B-Instruct`) runs locally with `make judges-open-it` and does
not need a token.

## Status (2026-10-26)

| Piece | State |
| --- | --- |
| Local gitignored `.env` with `HF_TOKEN` | Done (sourced from expression portfolio `.env`; never committed) |
| `uv sync --extra modal` | Done |
| Hugging Face CLI login + `~/.cache/huggingface/token` | Done (`whoami` = personal account `maxwellmbay`) |
| Llama Guard / ShieldGemma license accept | Max should confirm in browser if not already |
| Modal auth (`~/.modal.toml`) | **Blocked** — `modal setup` needs interactive browser login |
| Modal secret `huggingface-secret` | Not created yet (needs Modal auth first) |
| `modal deploy` / `make judges-llama` | Not started |

Celestine: no separate local Celestine env or token found; expression `HF_TOKEN` is the personal resume/portfolio credential in use.

## Finish Modal (Max, once)

```bash
cd /Users/maxwellbay/projects/label-efficient-eval
set -a && source .env && set +a   # loads HF_TOKEN; do not echo
uv run modal setup                # complete browser login
uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN"
# If the secret already exists:
#   uv run modal secret create huggingface-secret HF_TOKEN="$HF_TOKEN" --force
uv run modal deploy src/lee/modal_apps/judges.py
make judges-llama
make m1-full && make figures
```

Or, if Max already has token id/secret from the Modal dashboard:

```bash
uv run modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

## One-time setup (reference)

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
