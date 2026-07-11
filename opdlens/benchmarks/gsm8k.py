"""GSM8K — grade-school math, numeric answers. The primary train + eval set."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import PrivateAttr

from ..types import Example
from .base import _HASH_ANSWER, BaseBenchmark, extract_boxed, last_number


class Gsm8kBenchmark(BaseBenchmark):
    type: Literal["gsm8k"] = "gsm8k"
    hf_id: str | None = "openai/gsm8k"
    hf_name: str | None = "main"

    _fewshot_cache: str | None = PrivateAttr(default=None)

    def _fewshot_prefix(self) -> str:
        # EvalScope's deterministic n-shot block: the first ``few_shot_num`` train rows,
        # each rendered "<q>\n\nReasoning:\n<rationale>\n\nANSWER: \boxed{<target>}" (the
        # rationale keeps GSM8K's <<...>> calculator spans), joined by blank lines. This
        # matches the ms-swift/EvalScope GSM8K eval that reports the 0.7597 baseline.
        if self._fewshot_cache is None:
            rows = list(self._load_rows("train"))[: self.few_shot_num]
            demos: list[str] = []
            for row in rows:
                question = str(row[self.question_key])
                reasoning, _, target = str(row[self.answer_key]).partition("####")
                demos.append(
                    f"{question}\n\nReasoning:\n{reasoning.strip()}\n\n"
                    f"ANSWER: \\boxed{{{target.strip()}}}"
                )
            body = "\n\n".join(demos)
            self._fewshot_cache = f"Here are some examples of how to solve similar problems:\n\n{body}\n\n"
        return self._fewshot_cache

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
