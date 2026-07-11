"""Benchmark base — the behavior-rich analog of a reward, not a thin dataset.

A benchmark owns prompt construction (``build_prompt``) AND grading (``grade`` /
``extract_answer``), plus the thin row access it delegates to ``opdlens.data``.
The SAME ``build_prompt``/``grade`` serve training-rollout prompts, in-loop eval,
and reported eval — so there is exactly one grader and one prompt per dataset (the
fix for the first draft's select-vs-report grader divergence).
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from ..data import read_hf, read_jsonl
from ..types import Example

MATH_SYSTEM = (
    "Solve the following problem step by step. Put your final answer within \\boxed{}."
)

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_HASH_ANSWER = re.compile(r"####\s*(.+)")


def extract_boxed(text: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}``, or None."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    out: list[str] = []
    while i < len(text) and depth > 0:
        char = text[i]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if depth > 0:
            out.append(char)
        i += 1
    return "".join(out) if depth == 0 else None


def last_number(text: str) -> str | None:
    """Return the last number-like token (commas stripped), or None."""
    matches = _NUMBER.findall(text)
    return matches[-1].replace(",", "") if matches else None


class BaseBenchmark(BaseModel):
    """One interface, per-dataset implementation. Members override ``grade`` and
    (if the row schema differs) ``_row_to_example``."""

    model_config = ConfigDict(extra="forbid")

    type: str
    path: str | None = None
    hf_id: str | None = None
    hf_name: str | None = None
    split: str = "test"
    question_key: str = "question"
    answer_key: str = "answer"
    max_samples: int | None = None
    system_prompt: str | None = MATH_SYSTEM

    # Prompt shape. ``answer_instruction`` is appended to the question (e.g. the
    # EvalScope "...put your final answer within \boxed{}." line); ``few_shot_num``
    # prepends that many in-context demos via ``_fewshot_prefix`` (dataset-specific).
    # A train benchmark and an eval benchmark are SEPARATE instances, so on-policy
    # rollout may run zero-shot while eval runs few-shot — one builder, one grader,
    # and in-loop eval is byte-identical to reported eval (no select-vs-report skew).
    answer_instruction: str | None = None
    few_shot_num: int = 0

    # Eval-time sampling (SEPARATE from training-rollout sampling on the trainer).
    eval_temperature: float = 0.6
    eval_top_p: float = 1.0
    eval_max_tokens: int = 1024
    avg_k: int = 8
    """Samples per question for the avg@k eval metric (temp-0.6, noise-robust)."""

    # --------------------------------- Prompt --------------------------------- #

    def build_prompt(self, example: Example) -> list[dict[str, str]]:
        """Chat messages for one example. Shared by rollout and eval."""
        messages: list[dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        content = ""
        if self.few_shot_num > 0:
            content += self._fewshot_prefix()
        content += example.question
        if self.answer_instruction:
            content += "\n" + self.answer_instruction
        messages.append({"role": "user", "content": content})
        return messages

    def _fewshot_prefix(self) -> str:
        """In-context demo block prepended when ``few_shot_num > 0`` (dataset-specific)."""
        raise NotImplementedError(f"{self.type} does not implement few-shot demos")

    # ------------------------------- Row loading ------------------------------ #

    def _load_rows(self, split: str | None) -> Any:
        split = split or self.split
        if self.path is not None:
            return read_jsonl(self.path)
        if self.hf_id is not None:
            return read_hf(self.hf_id, split, name=self.hf_name)
        raise ValueError(f"benchmark {self.type!r} needs a path or hf_id")

    def _row_to_example(self, row: dict[str, Any]) -> Example:
        return Example(
            question=str(row[self.question_key]), gold=str(row[self.answer_key])
        )

    def iter_examples(self, split: str | None = None) -> list[Example]:
        rows = self._load_rows(split)
        examples = [self._row_to_example(row) for row in rows]
        if self.max_samples:
            examples = examples[: self.max_samples]
        return examples

    # -------------------------------- Grading --------------------------------- #

    def extract_answer(self, completion: str) -> str:
        """Default: the last ``\\boxed{}`` content, else the last number."""
        return extract_boxed(completion) or last_number(completion) or ""

    def grade(self, completion: str, gold: str) -> bool:
        raise NotImplementedError
