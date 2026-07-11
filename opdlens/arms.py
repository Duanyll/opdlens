"""The four experiment arms — the single seam where the arms differ.

Every arm shares the same rollout / teacher-forward / base OPD loss / eval spine
(``opdlens.training``). The ONLY per-arm code is ``aux_loss`` below; the four
bodies are stacked here on purpose so the difference is visible at a glance:

- ``LogitsArm``     (A): no aux — plain OPD.
- ``LogitLensArm``  (B): teacher **logit-lens** readout ``unembed_t(h)`` → vocab KL.
- ``JLensArm``      (C): teacher **Jacobian-lens** readout ``unembed_t(h @ Jᵀ)`` → vocab KL.
- ``HiddenMseArm``  (D): raw-hidden MSE through a frozen bridge (no unembed).

B and C differ by exactly one line (the teacher readout); the shared masked-KL
machinery lives in ``losses.lens_kl``. Arms load their artifacts (``lens.pt`` /
``bridge.pt``) directly — the trainer never touches the ``jlens`` package.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, PrivateAttr, Tag, TypeAdapter

from .losses import KLDir, hidden_mse, lens_kl, map_student_layer
from .types import CaptureSpec, Readout


class BaseArm(BaseModel):
    """Shared arm config + the ``capture_spec`` layer-pairing (identical for all
    arms). Subclasses implement only ``aux_loss``."""

    model_config = ConfigDict(extra="forbid")

    type: str
    aux_weight: float = 0.0
    temperature: float = 1.0
    aux_max_tokens: int = 512
    kl: KLDir = "forward"
    teacher_layers: tuple[int, ...] = ()
    """Teacher block indices to supervise (empty for arm A)."""

    def capture_spec(self, n_student: int, n_teacher: int) -> CaptureSpec:
        """Which hidden layers each forward must retain. Student layers are paired
        to ``teacher_layers`` by depth fraction (``losses.map_student_layer``)."""
        student_layers = tuple(
            map_student_layer(layer, n_student, n_teacher)
            for layer in self.teacher_layers
        )
        return CaptureSpec(
            student_layers=student_layers, teacher_layers=self.teacher_layers
        )

    def aux_loss(
        self,
        student: Readout,
        teacher: Readout,
        loss_mask: torch.Tensor,
        spec: CaptureSpec,
        *,
        unembed_s: Any,
        unembed_t: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Return ``(aux_scalar, diagnostics)`` for one sequence. Never called when
        ``aux_weight == 0`` (the trainer gates on it)."""
        raise NotImplementedError

    def _mean_over_pairs(self, terms: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(terms).mean()


class LogitsArm(BaseArm):
    """Arm A — plain OPD, no auxiliary supervision."""

    type: Literal["logits"] = "logits"

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        return loss_mask.new_zeros((), dtype=torch.float32), {}


class LogitLensArm(BaseArm):
    """Arm B — teacher logit-lens readout ``unembed_t(h)`` vs student logit-lens."""

    type: Literal["logit_lens"] = "logit_lens"

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        terms = [
            lens_kl(
                student.hidden[ls],
                teacher.hidden[lt],
                unembed_s,
                unembed_t,  # ← B: teacher readout is the plain logit-lens
                loss_mask,
                temperature=self.temperature,
                aux_max_tokens=self.aux_max_tokens,
                kl=self.kl,
            )
            for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True)
        ]
        return self._mean_over_pairs(terms), {}


class JLensArm(BaseArm):
    """Arm C — teacher Jacobian-lens readout ``unembed_t(h @ Jᵀ)``."""

    type: Literal["jspace"] = "jspace"
    jacobian_path: str
    """Path to a ``lens.pt`` written by ``opdlens.jlens.JacobianLens.save``."""

    _jacobians: dict[int, torch.Tensor] | None = PrivateAttr(default=None)

    def _load_jacobians(self) -> dict[int, torch.Tensor]:
        jacobians = self._jacobians
        if jacobians is None:
            ckpt = torch.load(self.jacobian_path, map_location="cpu", weights_only=True)
            if "J" not in ckpt:
                raise ValueError(f"{self.jacobian_path} is not a JacobianLens file")
            jacobians = ckpt["J"]
            self._jacobians = jacobians
        return jacobians

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        jacobians = self._load_jacobians()
        terms: list[torch.Tensor] = []
        for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True):
            h = teacher.hidden[lt]
            jac = jacobians[lt].to(device=h.device, dtype=h.dtype)
            terms.append(
                lens_kl(
                    student.hidden[ls],
                    h,
                    unembed_s,
                    lambda x, jac=jac: unembed_t(
                        x @ jac.T
                    ),  # ← C: transport then unembed
                    loss_mask,
                    temperature=self.temperature,
                    aux_max_tokens=self.aux_max_tokens,
                    kl=self.kl,
                )
            )
        return self._mean_over_pairs(terms), {}


class HiddenMseArm(BaseArm):
    """Arm D — OPRD-style raw-hidden MSE through a frozen affine bridge."""

    type: Literal["hidden_mse"] = "hidden_mse"
    bridge_path: str
    """Path to a ``bridge.pt`` = ``{"W": [d_t, d_s], "b_x": [d_t], "b_y": [d_s]}``."""

    _bridge: dict[str, torch.Tensor] | None = PrivateAttr(default=None)

    def _load_bridge(self, ref: torch.Tensor) -> dict[str, torch.Tensor]:
        if self._bridge is None:
            raw = torch.load(self.bridge_path, map_location="cpu", weights_only=True)
            self._bridge = {
                k: raw[k].to(device=ref.device, dtype=ref.dtype)
                for k in ("W", "b_x", "b_y")
            }
        return self._bridge

    def _apply_bridge(self, h: torch.Tensor) -> torch.Tensor:
        b = self._load_bridge(h)
        return (h - b["b_x"]) @ b["W"] + b["b_y"]

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        terms = [
            hidden_mse(
                student.hidden[ls], self._apply_bridge(teacher.hidden[lt]), loss_mask
            )
            for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True)
        ]
        return self._mean_over_pairs(terms), {}


Arm = Annotated[
    Annotated[LogitsArm, Tag("logits")]
    | Annotated[LogitLensArm, Tag("logit_lens")]
    | Annotated[JLensArm, Tag("jspace")]
    | Annotated[HiddenMseArm, Tag("hidden_mse")],
    Discriminator("type"),
]

_arm_ta = TypeAdapter(Arm)


def parse_arm(conf: dict[str, Any]) -> BaseArm:
    """Parse an arm config dict into the appropriate arm instance."""
    return _arm_ta.validate_python(conf)


__all__ = [
    "Arm",
    "BaseArm",
    "HiddenMseArm",
    "JLensArm",
    "LogitLensArm",
    "LogitsArm",
    "parse_arm",
]


if __name__ == "__main__":
    from rich import print

    torch.manual_seed(0)
    n_student, n_teacher, d, v, seq = 4, 6, 16, 32, 10
    unembed_s = torch.nn.Linear(d, v)
    unembed_t = torch.nn.Linear(d, v)

    arm = parse_arm({"type": "logit_lens", "aux_weight": 0.1, "teacher_layers": [2, 4]})
    assert isinstance(arm, LogitLensArm)
    spec = arm.capture_spec(n_student, n_teacher)
    print(f"spec: {spec}")

    student = Readout(
        logits=None, hidden={ls: torch.randn(seq, d) for ls in spec.student_layers}
    )
    teacher = Readout(
        logits=None, hidden={lt: torch.randn(seq, d) for lt in spec.teacher_layers}
    )
    loss_mask = torch.zeros(seq, dtype=torch.bool)
    loss_mask[4:] = True

    aux, diag = arm.aux_loss(
        student, teacher, loss_mask, spec, unembed_s=unembed_s, unembed_t=unembed_t
    )
    print(f"logit_lens aux: {aux.item():.4f}")
    assert torch.isfinite(aux)

    for conf in (
        {"type": "logits"},
        {"type": "jspace", "jacobian_path": "x.pt"},
        {"type": "hidden_mse", "bridge_path": "b.pt"},
    ):
        print(f"parsed {conf['type']}: {type(parse_arm(conf)).__name__}")

    print("[green]arms smoke test passed[/green]")
