from __future__ import annotations

from typing import Any

from opdlens.fit.fit_jlens import _calibration_texts


class _Tokenizer:
    def apply_chat_template(self, messages: list[dict[str, str]], **_: Any) -> str:
        return "|".join(
            f"{message['role']}:{message['content']}" for message in messages
        )


def test_calibration_texts_can_include_gold_completion():
    rows = [{"question": "2+2?", "answer": "work... 4"}]
    texts = _calibration_texts(
        rows,
        _Tokenizer(),
        prompt_key="question",
        answer_key="answer",
        include_completion=True,
    )
    assert texts == ["user:2+2?|assistant:work... 4"]


def test_calibration_texts_default_to_raw_prompts():
    rows = [{"question": "2+2?", "answer": "4"}]
    texts = _calibration_texts(
        rows,
        _Tokenizer(),
        prompt_key="question",
        answer_key="answer",
        include_completion=False,
    )
    assert texts == ["2+2?"]
