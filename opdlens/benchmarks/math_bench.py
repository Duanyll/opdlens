"""MATH-family benchmarks graded by symbolic equivalence (``math_verify``).

Math-500 and AIME are eval-only competition sets; MATH-train (Hendrycks) is a
harder-than-GSM8K training set. They share one grader (``math_verify``); they
differ only in the row source / schema, expressed as field defaults.
"""

from __future__ import annotations

from typing import Literal

from .base import BaseBenchmark, extract_boxed


class MathVerifyBenchmark(BaseBenchmark):
    """Grade by symbolic equivalence of the extracted answer to the gold LaTeX."""

    def extract_answer(self, completion: str) -> str:
        return extract_boxed(completion) or completion.strip()

    def grade(self, completion: str, gold: str) -> bool:
        from math_verify import parse, verify

        try:
            return bool(verify(parse(gold), parse(completion)))
        except Exception:
            return False


class Math500Benchmark(MathVerifyBenchmark):
    type: Literal["math500"] = "math500"
    hf_id: str | None = "HuggingFaceH4/MATH-500"
    question_key: str = "problem"
    answer_key: str = "answer"


class AimeBenchmark(MathVerifyBenchmark):
    type: Literal["aime"] = "aime"
    question_key: str = "problem"
    answer_key: str = "answer"
    year: int = 2024


class MathTrainBenchmark(MathVerifyBenchmark):
    type: Literal["math_train"] = "math_train"
    split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
