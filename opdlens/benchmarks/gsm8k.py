"""GSM8K — grade-school math, numeric answers. The primary train + eval set."""

from __future__ import annotations

from typing import Any, Literal

from ..types import Example
from .base import _HASH_ANSWER, BaseBenchmark, extract_boxed, last_number


class Gsm8kBenchmark(BaseBenchmark):
    type: Literal["gsm8k"] = "gsm8k"
    hf_id: str | None = "openai/gsm8k"
    hf_name: str | None = "main"

    def _row_to_example(self, row: dict[str, Any]) -> Example:
        answer = str(row[self.answer_key])
        # HF GSM8K gold is the full rationale ending in "#### <number>".
        match = _HASH_ANSWER.search(answer)
        gold = last_number(match.group(1)) if match else last_number(answer)
        return Example(question=str(row[self.question_key]), gold=gold or answer)

    def extract_answer(self, completion: str) -> str:
        # One consistent order (boxed → #### → last number), used for BOTH in-loop
        # selection and reported eval.
        boxed = extract_boxed(completion)
        if boxed is not None and (num := last_number(boxed)) is not None:
            return num
        match = _HASH_ANSWER.search(completion)
        if match is not None and (num := last_number(match.group(1))) is not None:
            return num
        return last_number(completion) or ""

    def grade(self, completion: str, gold: str) -> bool:
        pred = self.extract_answer(completion)
        try:
            return abs(float(pred) - float(gold)) < 1e-3
        except ValueError:
            return False
