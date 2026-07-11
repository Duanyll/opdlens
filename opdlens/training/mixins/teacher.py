"""TeacherMixin — the frozen teacher and its capturing forward."""

from __future__ import annotations

import torch

from ...models import LanguageModel
from ...types import CaptureSpec, Readout
from ..base import BaseTrainer


class TeacherMixin(BaseTrainer):
    teacher: LanguageModel

    def load_teacher(self) -> None:
        self.teacher.load(self.device, trainable=False)

    def teacher_forward(self, input_ids: torch.Tensor, spec: CaptureSpec) -> Readout:
        """One frozen forward, capturing final logits + the requested teacher layers."""
        with torch.no_grad():
            return self.teacher.forward_capture(
                input_ids, layers=spec.teacher_layers, need_logits=True
            )
