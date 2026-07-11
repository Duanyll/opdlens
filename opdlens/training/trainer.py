"""OpdTrainer — the one trainer. Four arms, one spine.

Composed from the cohesive mixins; the ONLY per-arm variation is ``self.arm``. The
``train_step`` below is byte-identical across arms: rollout → teacher & student
forward → shared ``opd_base_loss`` (the invariant) → ``arm.aux_loss`` (the seam) →
step → weight-sync. When ``aux_weight == 0`` (arm A, or any arm under
``test_spine``) the aux path is skipped entirely — no hidden capture, no RNG use —
so every arm reduces to identical base OPD.
"""

from __future__ import annotations

from typing import Any

import torch
from pydantic import ConfigDict, PrivateAttr
from rich.progress import Progress

from ..arms import Arm
from ..losses import opd_base_loss
from ..types import CaptureSpec
from ..utils.logging import console, get_logger
from .base import distributed_main
from .mixins import (
    CheckpointingMixin,
    EvalMixin,
    OptimMixin,
    RolloutMixin,
    TeacherMixin,
)

logger = get_logger(__name__)


class OpdTrainer(RolloutMixin, EvalMixin, TeacherMixin, OptimMixin, CheckpointingMixin):
    model_config = ConfigDict(extra="forbid")

    arm: Arm
    train_steps: int = 200
    base_temperature: float = 1.0
    base_beta: float = 0.0
    """Base-loss divergence: 0=forward KL (default), 0.5=JSD (ms-swift GKD), 1=reverse."""
    eval_steps: int = 50
    eval_at_start: bool = True

    _current_step: int = PrivateAttr(default=0)

    # ------------------------------- Train step ------------------------------- #

    def train_step(self) -> None:
        active_aux = self.arm.aux_weight != 0.0
        spec = (
            self.arm.capture_spec(self.student.num_layers, self.teacher.num_layers)
            if active_aux
            else CaptureSpec()
        )
        batches = self.rollout()
        if not batches:
            logger.warning("Empty rollout at step %d; skipping.", self._current_step)
            return

        self.student.model.train()
        # Accumulate the logging scalars on-device and read them back ONCE after the
        # step. Per-batch ``float(...)`` would force a GPU->CPU sync every micro-batch,
        # serializing the 96-way grad-accumulation loop and starving the GPU.
        base_sum = torch.zeros((), device=self.device)
        aux_sum = torch.zeros((), device=self.device)
        for batch in batches:
            student_out = self.student.forward_capture(
                batch.input_ids, layers=spec.student_layers, need_logits=True
            )
            teacher_out = self.teacher_forward(batch.input_ids, spec)
            assert student_out.logits is not None and teacher_out.logits is not None
            base = opd_base_loss(
                student_out.logits,
                teacher_out.logits,
                batch.loss_mask,
                temperature=self.base_temperature,
                beta=self.base_beta,
            )
            if active_aux:
                aux, _ = self.arm.aux_loss(
                    student_out,
                    teacher_out,
                    batch.loss_mask,
                    spec,
                    unembed_s=self.student.unembed,
                    unembed_t=self.teacher.unembed,
                )
                loss = base + self.arm.aux_weight * aux
                aux_sum = aux_sum + aux.detach()
            else:
                loss = base
            (loss / len(batches)).backward()
            base_sum = base_sum + base.detach()

        grad_norm = self.optimizer_step()
        self._current_step += 1
        self.sync_weights()

        n = len(batches)
        base_mean = base_sum / n
        aux_mean = aux_sum / n
        self.log_metrics(
            {
                "train/base": base_mean,
                "train/aux": aux_mean,
                "train/loss": base_mean + self.arm.aux_weight * aux_mean,
                "train/lr": self.lr,
                "train/grad_norm": grad_norm,
            },
            step=self._current_step,
        )

    # --------------------------------- Run loop ------------------------------- #

    @distributed_main
    def run(self) -> None:
        self.set_seed()
        self.resolve_run_context()
        self.init_tracker()
        self.load_student()
        self.load_teacher()
        self.make_optimizer_and_scheduler(self.student.model.parameters())
        self.init_generation()
        self.load_rollout_data()
        self.maybe_auto_resume(self.resume_from_dir)

        if self.eval_at_start:
            self.evaluate(self._current_step)

        progress = Progress(*self.get_progress_columns(), console=console)
        task = progress.add_task(
            "OPD training", total=self.train_steps, completed=self._current_step
        )
        with self.status_bar("OPD training"), progress:
            while self._current_step < self.train_steps:
                self.train_step()
                progress.update(task, completed=self._current_step)
                self.save_maybe(self._current_step)
                if self.eval_steps > 0 and self._current_step % self.eval_steps == 0:
                    self.evaluate(self._current_step)

        self.save_checkpoint(self._current_step)
        self.finish_tracker()

    # ------------------------------ Checkpointing ----------------------------- #

    def state_dict(self) -> dict[str, Any]:
        return {
            "student": self.student.model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "step": self._current_step,
            "rng": self.get_rng_state_bytes(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.student.model.load_state_dict(state["student"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        self._current_step = int(state["step"])
        self.load_rng_state_bytes(state.get("rng"))
