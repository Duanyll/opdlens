"""Optimizer / scheduler mixin."""

from __future__ import annotations

from typing import Any

import torch
from pydantic import PrivateAttr

from ...utils.logging import get_logger
from ...utils.types import (
    OptimizerConfig,
    SchedulerConfig,
    parse_optimizer,
    parse_scheduler,
)
from ..base import BaseTrainer

logger = get_logger(__name__)


class OptimMixin(BaseTrainer):
    optimizer_config: OptimizerConfig = {"class_name": "AdamW", "lr": 2e-6}
    scheduler_config: SchedulerConfig = {"class_name": "ConstantLR", "factor": 1.0}
    clip_grad_norm: float = 1.0

    _optimizer: Any = PrivateAttr(default=None)
    _scheduler: Any = PrivateAttr(default=None)
    _trainable_params: list[torch.nn.Parameter] = PrivateAttr(default_factory=list)

    def make_optimizer_and_scheduler(self, parameters: Any) -> None:
        params = [p for p in parameters if p.requires_grad]
        if not params:
            raise RuntimeError("No trainable parameters found.")
        self._trainable_params = params
        n = sum(p.numel() for p in params)
        self._optimizer = parse_optimizer(self.optimizer_config, params)
        self._scheduler = parse_scheduler(self.scheduler_config, self._optimizer)
        logger.info("Optimizer ready with %.2fM trainable parameters.", n / 1e6)

    def optimizer_step(self) -> float:
        """Reduce grads across ranks, clip, step, and zero. Returns grad norm."""
        self.reduce_gradients(self._trainable_params)
        grad_norm = 0.0
        if self.clip_grad_norm > 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    self._trainable_params, self.clip_grad_norm
                )
            )
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        return grad_norm

    @property
    def lr(self) -> float:
        return float(self._scheduler.get_last_lr()[0])
