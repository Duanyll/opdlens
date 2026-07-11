"""EvalMixin — periodic eval across benchmarks, sharing the rollout generator.

For each benchmark: build prompts, ``generate`` avg@k samples (same engine as
rollout), grade with the benchmark's own ``grade``, report accuracy + SE. Grading
uses the exact same prompt + grader as everything else — no select-vs-report skew.
No best-step selection; eval runs at fixed steps.
"""

from __future__ import annotations

from typing import Any

import torch.distributed as dist

from ...benchmarks import Benchmark
from ...types import EvalReport
from ...utils.logging import get_logger
from .generation import GenerationMixin
from .logging import LoggingMixin

logger = get_logger(__name__)


def _stderr(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return (var / n) ** 0.5


class EvalMixin(GenerationMixin, LoggingMixin):
    eval_benchmarks: list[Benchmark] = []
    eval_max_samples: int | None = 200

    def evaluate(self, step: int) -> list[EvalReport]:
        self.sync_weights()  # vLLM must reflect the current student
        reports: list[EvalReport] = []
        for benchmark in self.eval_benchmarks:
            report = self._evaluate_one(benchmark, step)
            if report is not None:
                reports.append(report)
        return reports

    def _evaluate_one(self, benchmark: Benchmark, step: int) -> EvalReport | None:
        examples = benchmark.iter_examples("test")
        if self.eval_max_samples:
            examples = examples[: self.eval_max_samples]
        local = examples[self.rank :: self.world_size]  # shard across ranks

        prompt_ids = [self.encode_prompt(benchmark, ex) for ex in local]
        samples = self.generate(
            prompt_ids,
            temperature=benchmark.eval_temperature,
            top_p=benchmark.eval_top_p,
            max_tokens=benchmark.eval_max_tokens,
            n=benchmark.avg_k,
        )
        local_soft: list[float] = []
        for example, completions in zip(local, samples, strict=True):
            texts = [self.decode(c) for c in completions]
            correct = sum(benchmark.grade(t, example.gold) for t in texts)
            local_soft.append(correct / max(len(texts), 1))

        soft = self._gather_soft(local_soft)
        if not self.is_main_process:
            return None
        n = len(soft)
        accuracy = sum(soft) / n if n else 0.0
        se = _stderr(soft)
        self.log_main(
            {f"eval/{benchmark.type}/acc": accuracy, f"eval/{benchmark.type}/se": se},
            step,
        )
        logger.info(
            "eval[%s] step %d: acc=%.4f se=%.4f (n=%d)",
            benchmark.type,
            step,
            accuracy,
            se,
            n,
        )
        return EvalReport(
            benchmark=benchmark.type, n=n, accuracy=accuracy, soft=soft, se=se
        )

    def _gather_soft(self, local: list[float]) -> list[float]:
        if self.world_size <= 1:
            return local
        buffer: list[Any] = [None] * self.world_size
        dist.all_gather_object(buffer, local)
        pooled: list[float] = []
        for chunk in buffer:
            pooled.extend(chunk)
        return pooled
