"""MATH-family benchmarks graded by symbolic equivalence (``math_verify``).

The DAPO train set, canonical MATH test, MATH-500, and AIME share one prompt and
grader. Dataset members own only their row source and schema adaptation.
"""

from __future__ import annotations

from typing import Any, Literal

from ..data import read_hf
from ..types import Example
from .base import BaseBenchmark, extract_boxed

MATH_SUBJECTS = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)
MathSubject = Literal[
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MathVerifyBenchmark(BaseBenchmark):
    """Grade by symbolic equivalence of the extracted answer to the gold LaTeX."""

    def extract_answer(self, completion: str) -> str:
        return extract_boxed(completion) or completion.strip()

    def grade(self, completion: str, gold: str) -> bool:
        from math_verify import parse, verify

        try:
            # Dataset golds are bare expressions, while math_verify's default
            # extractor expects an answer anchor for many LaTeX forms. Give the
            # reference the same unambiguous boxed envelope requested from models.
            reference = self.extract_answer(gold)
            return bool(verify(parse(f"\\boxed{{{reference}}}"), parse(completion)))
        except Exception:
            return False


class Math500Benchmark(MathVerifyBenchmark):
    type: Literal["math500"] = "math500"
    hf_id: str | None = "HuggingFaceH4/MATH-500"
    question_key: str = "problem"
    answer_key: str = "answer"


class DapoMathBenchmark(MathVerifyBenchmark):
    """Deduplicated DAPO-Math-17k training prompts (17,398 rows)."""

    type: Literal["dapo_math"] = "dapo_math"
    hf_id: str | None = "open-r1/DAPO-Math-17k-Processed"
    hf_name: str | None = "all"
    hf_revision: str = "31dd309567e3da778038cc87d868b6097a3ccf68"
    split: str = "train"
    question_key: str = "prompt"
    answer_key: str = "solution"

    def _load_rows(self, split: str | None) -> Any:
        if self.path is not None:
            return super()._load_rows(split)
        if self.hf_id is None:
            raise ValueError("dapo_math needs a path or hf_id")
        return read_hf(
            self.hf_id,
            split or self.split,
            name=self.hf_name,
            revision=self.hf_revision,
        )


class MathBenchmark(MathVerifyBenchmark):
    """Canonical Hendrycks MATH test, aggregated across all seven subjects."""

    type: Literal["math"] = "math"
    hf_id: str | None = "EleutherAI/hendrycks_math"
    hf_revision: str = "21a5633873b6a120296cce3e2df9d5550074f4a3"
    split: str = "test"
    question_key: str = "problem"
    answer_key: str = "solution"
    subjects: tuple[MathSubject, ...] = MATH_SUBJECTS

    def _load_rows(self, split: str | None) -> Any:
        if self.path is not None:
            return super()._load_rows(split)
        if self.hf_id is None:
            raise ValueError("math needs a path or hf_id")
        selected_split = split or self.split
        if self.hf_name is not None:
            return read_hf(
                self.hf_id,
                selected_split,
                name=self.hf_name,
                revision=self.hf_revision,
            )
        rows: list[dict[str, Any]] = []
        for subject in self.subjects:
            rows.extend(
                read_hf(
                    self.hf_id,
                    selected_split,
                    name=subject,
                    revision=self.hf_revision,
                )
            )
        return rows

    def _row_to_example(self, row: dict[str, Any]) -> Example:
        solution = str(row[self.answer_key])
        # A small number of canonical rows use \fbox rather than \boxed.
        gold = extract_boxed(solution) or extract_boxed(
            solution.replace("\\fbox{", "\\boxed{")
        )
        return Example(
            question=str(row[self.question_key]), gold=gold or solution.strip()
        )


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
