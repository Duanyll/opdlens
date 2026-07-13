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


# AIME gold answers are bare integers, but the source schema differs by year: 2024
# boxes the answer inside `solution`, 2025 exposes a bare `answer` field. Both are
# normalized by ``extract_answer`` in grading.
_AIME_SOURCE: dict[int, tuple[str, str]] = {
    2024: ("math-ai/aime24", "solution"),
    2025: ("math-ai/aime25", "answer"),
}


class AimeBenchmark(MathVerifyBenchmark):
    """AIME competition set (30 problems). Tiny and high-variance, so eval with
    avg@k (the OPRD protocol uses k=16 at temperature 0.7). ``year`` picks the
    source dataset; the boxed-vs-bare gold is handled by ``extract_answer``."""

    type: Literal["aime"] = "aime"
    year: Literal[2024, 2025] = 2024
    question_key: str = "problem"
    split: str = "test"

    def _load_rows(self, split: str | None) -> Any:
        if self.path is not None:
            return super()._load_rows(split)
        hf_id, _ = _AIME_SOURCE[self.year]
        return read_hf(hf_id, split or self.split)

    def _row_to_example(self, row: dict[str, Any]) -> Example:
        _, answer_key = _AIME_SOURCE[self.year]
        return Example(question=str(row[self.question_key]), gold=str(row[answer_key]))


class AimoBenchmark(MathVerifyBenchmark):
    """AI-MO validation AMC = AMC 2022 + 2023 (83 problems); the OPRD 'AIMO' set.
    Gold ``answer`` is a numeric string (e.g. ``142.0``), graded by math_verify."""

    type: Literal["aimo"] = "aimo"
    hf_id: str | None = "AI-MO/aimo-validation-amc"
    split: str = "train"
    question_key: str = "problem"
    answer_key: str = "answer"

    def _load_rows(self, split: str | None) -> Any:
        # The eval loop always requests the "test" split, but this validation set
        # ships only a "train" split — pin to our own ``split`` and ignore the hint.
        if self.path is not None:
            return super()._load_rows(split)
        if self.hf_id is None:
            raise ValueError("aimo needs a path or hf_id")
        return read_hf(self.hf_id, self.split)


class MathTrainBenchmark(MathVerifyBenchmark):
    type: Literal["math_train"] = "math_train"
    split: str = "train"
    question_key: str = "question"
    answer_key: str = "answer"
