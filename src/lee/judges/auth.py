"""Auth / license blockers with exact commands for Max."""

from __future__ import annotations


class JudgeAuthError(RuntimeError):
    """Raised when Modal or Hugging Face credentials block a gated judge."""


LLAMA_GUARD_SETUP = """
Llama Guard 3 is gated on Hugging Face and needs a personal Modal account
for the GPU path (prefer personal Modal, not work).

1) Hugging Face (once):
   uv sync --extra modal
   uv run huggingface-cli login
   # Or: export HF_TOKEN=hf_...
   Open and accept the license:
   https://huggingface.co/meta-llama/Llama-Guard-3-1B
   (Optional ShieldGemma: https://huggingface.co/google/shieldgemma-2b)

2) Modal (personal):
   uv run modal setup
   # Or: uv run modal token new
   Confirm: uv run modal profile current

3) Score + cache (from repo root):
   make judges-llama
   # or: uv run python scripts/run_judges.py --judge llama_guard3_1b --backend modal

4) Then re-run the grid:
   make m1-full && make figures
""".strip()


SHIELDGEMMA_SETUP = """
ShieldGemma is gated. After HF login and license accept at
https://huggingface.co/google/shieldgemma-2b, and Modal setup as above:

  uv run python scripts/run_judges.py --judge shieldgemma_2b --backend modal
""".strip()


def require_hf_token() -> str:
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise JudgeAuthError(
            "No Hugging Face token found.\n" + LLAMA_GUARD_SETUP
        )
    return token


def require_modal() -> None:
    try:
        import modal  # noqa: F401
    except ImportError as exc:
        raise JudgeAuthError(
            "modal package missing. Run: uv sync --extra modal\n" + LLAMA_GUARD_SETUP
        ) from exc
    # Probe config file / env; Modal raises at runtime if unauthenticated.
    from pathlib import Path

    home = Path.home()
    has_config = (home / ".modal.toml").exists() or bool(
        __import__("os").environ.get("MODAL_TOKEN_ID")
    )
    if not has_config:
        raise JudgeAuthError(
            "Modal is not authenticated on this machine.\n" + LLAMA_GUARD_SETUP
        )
