"""RolloutMixin — on-policy sampling from the train benchmark's prompts.

Each rank draws its own prompts (data parallelism), samples completions through the
shared ``generate``, and assembles ``RolloutBatch``es (prompt + completion, with the
output-position ``loss_mask``).
"""

from __future__ import annotations

import random
from typing import Any

import torch
from pydantic import PrivateAttr

from ...benchmarks import Benchmark
from ...types import Example, RolloutBatch
from ...utils.logging import get_logger
from .generation import GenerationMixin

logger = get_logger(__name__)


class RolloutMixin(GenerationMixin):
    train_benchmark: Benchmark
    num_prompts_per_step: int = 8
    rollouts_per_prompt: int = 1

    _train_examples: list[Example] = PrivateAttr(default_factory=list)
    _deck: list[int] = PrivateAttr(default_factory=list)
    _cursor: int = PrivateAttr(default=0)
    _rng: Any = PrivateAttr(default=None)

    def load_rollout_data(self) -> None:
        self._train_examples = self.train_benchmark.iter_examples("train")
        self._rng = random.Random(self.seed + self.rank)
        self._deck = []
        self._cursor = 0
        logger.info("Loaded %d train examples for rollout.", len(self._train_examples))

    def _draw(self, n: int) -> list[Example]:
        drawn: list[Example] = []
        for _ in range(n):
            if self._cursor >= len(self._deck):
                self._deck = list(range(len(self._train_examples)))
                self._rng.shuffle(self._deck)
                self._cursor = 0
            drawn.append(self._train_examples[self._deck[self._cursor]])
            self._cursor += 1
        return drawn

    def rollout(self) -> list[RolloutBatch]:
        examples = self._draw(self.num_prompts_per_step)
        prompt_ids = [self.encode_prompt(self.train_benchmark, ex) for ex in examples]
        completions = self.generate(
            prompt_ids,
            temperature=self.rollout_temperature,
            top_p=self.rollout_top_p,
            max_tokens=self.rollout_max_tokens,
            n=self.rollouts_per_prompt,
        )
        batches: list[RolloutBatch] = []
        for ids, samples in zip(prompt_ids, completions, strict=True):
            for completion in samples:
                if not completion:
                    continue
                full = ids + completion
                length = len(full)
                input_ids = torch.tensor(full, device=self.device)
                loss_mask = torch.zeros(length, dtype=torch.bool, device=self.device)
                # Output positions t whose predicted next token (t+1) is a
                # completion token: prompt_len-1 .. T-2.
                loss_mask[len(ids) - 1 : length - 1] = True
                batches.append(
                    RolloutBatch(
                        input_ids=input_ids, prompt_len=len(ids), loss_mask=loss_mask
                    )
                )
        return batches
