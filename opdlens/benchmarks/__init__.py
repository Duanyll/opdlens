"""Benchmark discriminated union (plain pre-registry style)."""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import Discriminator, Tag, TypeAdapter

from .base import BaseBenchmark
from .gsm8k import Gsm8kBenchmark
from .math_bench import (
    AimeBenchmark,
    Math500Benchmark,
    MathTrainBenchmark,
    MathVerifyBenchmark,
)

Benchmark = Annotated[
    Annotated[Gsm8kBenchmark, Tag("gsm8k")]
    | Annotated[Math500Benchmark, Tag("math500")]
    | Annotated[AimeBenchmark, Tag("aime")]
    | Annotated[MathTrainBenchmark, Tag("math_train")],
    Discriminator("type"),
]

_benchmark_ta = TypeAdapter(Benchmark)


def parse_benchmark(conf: dict[str, Any]) -> BaseBenchmark:
    """Parse a benchmark config dict into the appropriate benchmark instance."""
    return _benchmark_ta.validate_python(conf)


__all__ = [
    "AimeBenchmark",
    "BaseBenchmark",
    "Benchmark",
    "Gsm8kBenchmark",
    "Math500Benchmark",
    "MathTrainBenchmark",
    "MathVerifyBenchmark",
    "parse_benchmark",
]
