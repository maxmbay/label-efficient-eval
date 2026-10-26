"""Judge ladder for Milestone 1."""

from lee.judges.auth import JudgeAuthError, LLAMA_GUARD_SETUP
from lee.judges.llama_guard import LlamaGuard3Judge
from lee.judges.open_it import OpenITJudge
from lee.judges.openai_moderation import OpenAIModerationMaxJudge
from lee.judges.shieldgemma import ShieldGemmaJudge
from lee.judges.stubs import LlamaGuardStub, OpenInstructStub, WeakKeywordJudge

# Always available (no GPU / no HF gate).
ACTIVE_M1_JUDGES = (
    OpenAIModerationMaxJudge,
    WeakKeywordJudge,
)

# Ladder rungs that need Model weights / Modal; scores live in data/cache/judges/.
LADDER_JUDGE_NAMES = (
    "llama_guard3_1b",
    "shieldgemma_2b",
    "open_it_toxicity",
    "open_it_jailbreaking",
)

STUB_M1_JUDGES = (
    LlamaGuardStub,
    OpenInstructStub,
)

__all__ = [
    "ACTIVE_M1_JUDGES",
    "LADDER_JUDGE_NAMES",
    "LLAMA_GUARD_SETUP",
    "JudgeAuthError",
    "LlamaGuard3Judge",
    "LlamaGuardStub",
    "OpenITJudge",
    "OpenAIModerationMaxJudge",
    "OpenInstructStub",
    "ShieldGemmaJudge",
    "STUB_M1_JUDGES",
    "WeakKeywordJudge",
]
