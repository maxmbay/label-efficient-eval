"""Judge unit tests (no Modal / HF)."""

from __future__ import annotations

import pandas as pd
import pytest

from lee.judges import LlamaGuardStub, OpenAIModerationMaxJudge, WeakKeywordJudge


def test_openai_moderation_max() -> None:
    frame = pd.DataFrame(
        {
            "openai_moderation": [
                '[["hate", 0.1], ["violence", 0.4]]',
                '[["sexual", 0.02]]',
            ]
        }
    )
    scores = OpenAIModerationMaxJudge().score(frame)
    assert list(scores) == pytest.approx([0.4, 0.02])


def test_weak_keyword() -> None:
    frame = pd.DataFrame({"user_input": ["hello world", "I hate this", "ok"]})
    scores = WeakKeywordJudge().score(frame)
    assert list(scores) == pytest.approx([0.0, 1.0, 0.0])


def test_llama_guard_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        LlamaGuardStub().score(pd.DataFrame({"user_input": ["x"]}))
