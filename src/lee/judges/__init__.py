"""Judge ladder for Milestone 1."""

from lee.judges.openai_moderation import OpenAIModerationMaxJudge
from lee.judges.stubs import LlamaGuardStub, OpenInstructStub, WeakKeywordJudge

ACTIVE_M1_JUDGES = (
    OpenAIModerationMaxJudge,
    WeakKeywordJudge,
)

STUB_M1_JUDGES = (
    LlamaGuardStub,
    OpenInstructStub,
)

__all__ = [
    "ACTIVE_M1_JUDGES",
    "STUB_M1_JUDGES",
    "LlamaGuardStub",
    "OpenAIModerationMaxJudge",
    "OpenInstructStub",
    "WeakKeywordJudge",
]
