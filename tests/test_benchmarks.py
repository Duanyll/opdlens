"""One grader + one prompt per benchmark (the anti-divergence contract)."""

from __future__ import annotations

from opdlens.benchmarks import parse_benchmark
from opdlens.types import Example


def test_gsm8k_grade_numeric():
    g = parse_benchmark({"type": "gsm8k"})
    assert g.grade("reasoning ...\n#### 42", "42")
    assert g.grade("the answer is \\boxed{42}", "42")
    assert not g.grade("the answer is 41", "42")


def test_gsm8k_extract_prefers_boxed_then_hash():
    g = parse_benchmark({"type": "gsm8k"})
    assert g.extract_answer("work \\boxed{18} more") == "18"
    assert g.extract_answer("only a total of 7 apples") == "7"


def test_math500_grade_symbolic():
    m = parse_benchmark({"type": "math500"})
    assert m.grade("so the answer is \\boxed{\\frac{1}{2}}", "\\frac{1}{2}")
    assert not m.grade("\\boxed{\\frac{1}{3}}", "\\frac{1}{2}")


def test_build_prompt_shared_shape():
    g = parse_benchmark({"type": "gsm8k"})
    messages = g.build_prompt(Example(question="What is 2+2?", gold="4"))
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user" and "2+2" in messages[-1]["content"]
