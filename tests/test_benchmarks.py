"""One grader + one prompt per benchmark (the anti-divergence contract)."""

from __future__ import annotations

import json

from opdlens.benchmarks import MathBenchmark, parse_benchmark
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


def test_dapo_math_uses_deduplicated_rows_and_shared_math_prompt(tmp_path):
    path = tmp_path / "dapo.jsonl"
    row = {"prompt": "Find $x$.", "solution": "\\frac{1}{2}"}
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    dapo = parse_benchmark({"type": "dapo_math", "path": str(path)})
    example = dapo.iter_examples()[0]
    messages = dapo.build_prompt(example)

    assert example == Example(question="Find $x$.", gold="\\frac{1}{2}")
    assert dapo.hf_id == "open-r1/DAPO-Math-17k-Processed"
    assert dapo.hf_name == "all"
    assert messages[0]["role"] == "system"
    assert messages[-1] == {"role": "user", "content": "Find $x$."}


def test_canonical_math_extracts_gold_and_aggregates_subjects(tmp_path, monkeypatch):
    path = tmp_path / "math.jsonl"
    row = {
        "problem": "Simplify.",
        "solution": "Work gives \\fbox{\\frac{2}{3}}.",
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    math = parse_benchmark({"type": "math", "path": str(path)})
    assert math.iter_examples()[0].gold == "\\frac{2}{3}"

    calls: list[tuple[str, str, str | None, str | None]] = []

    def fake_read_hf(
        hf_id: str,
        split: str,
        *,
        name: str | None = None,
        revision: str | None = None,
    ) -> list[dict]:
        calls.append((hf_id, split, name, revision))
        return []

    monkeypatch.setattr("opdlens.benchmarks.math_bench.read_hf", fake_read_hf)
    canonical = MathBenchmark()
    assert canonical._load_rows("test") == []
    assert [name for _, _, name, _ in calls] == list(canonical.subjects)
    assert all(revision == canonical.hf_revision for *_, revision in calls)


def test_build_prompt_shared_shape():
    g = parse_benchmark({"type": "gsm8k"})
    messages = g.build_prompt(Example(question="What is 2+2?", gold="4"))
    assert messages[0]["role"] == "system"
    assert messages[-1]["role"] == "user" and "2+2" in messages[-1]["content"]
