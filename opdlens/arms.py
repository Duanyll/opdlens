"""The five experiment arms — the single seam where the arms differ.

Every arm shares the same rollout / teacher-forward / base OPD loss / eval spine
(``opdlens.training``). The ONLY per-arm code is ``aux_loss`` below; the five
bodies are stacked here on purpose so the difference is visible at a glance:

- ``LogitsArm``     (A): no aux — plain OPD.
- ``LogitLensArm``  (B): teacher **logit-lens** readout ``unembed_t(h)`` → vocab KL.
- ``JLensArm``      (C): teacher **Jacobian-lens** readout ``unembed_t(h @ Jᵀ)`` → vocab KL.
- ``HiddenMseArm``  (D): raw-hidden MSE through a frozen bridge (no unembed).
- ``SymmetricJLensArm`` (E): offline-fit Jacobian-lens readouts on both sides.

B, C, and E share the masked-KL machinery in ``losses.lens_kl`` and differ only
in their readouts. Arms load their artifacts (``lens.pt`` / ``bridge.pt``)
directly — the trainer never touches the ``jlens`` package.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Discriminator, PrivateAttr, Tag, TypeAdapter

from .losses import (
    KLDir,
    hidden_mse,
    lens_kl,
    map_student_layer,
    sample_aux_token_index,
)
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
        token_index = sample_aux_token_index(loss_mask, self.aux_max_tokens)
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
                token_index=token_index,
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
        missing = [lt for lt in spec.teacher_layers if lt not in jacobians]
        if missing:
            raise ValueError(
                f"{self.jacobian_path} has no Jacobian for teacher_layers {missing}; "
                f"fitted source layers are {sorted(jacobians)}. Re-fit the lens with "
                f"--source-layers matching this arm's teacher_layers."
            )
        token_index = sample_aux_token_index(loss_mask, self.aux_max_tokens)
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
                    token_index=token_index,
                )
            )
        return self._mean_over_pairs(terms), {}


class HiddenMseArm(BaseArm):
    """Arm D — OPRD-style raw-hidden MSE through a frozen affine bridge."""

    type: Literal["hidden_mse"] = "hidden_mse"
    bridge_path: str
    """Path to a ``bridge.pt`` = ``{"W": [d_t, d_s], "b_x": [d_t], "b_y": [d_s]}``."""

    _bridges: dict[int, dict[str, torch.Tensor]] | None = PrivateAttr(default=None)
    _bridge_student_layers: dict[int, int] = PrivateAttr(default_factory=dict)

    def _load_bridges(self, ref: torch.Tensor) -> dict[int, dict[str, torch.Tensor]]:
        if self._bridges is not None:
            return self._bridges

        raw = torch.load(self.bridge_path, map_location="cpu", weights_only=True)
        if all(key in raw for key in ("W", "b_x", "b_y")):
            # -1 is the shared bridge written by opdlens.fit.fit_bridge.
            sources = {-1: raw}
        elif isinstance(raw.get("pairs"), dict):
            # Also accept the higher-quality per-layer bridge artifact produced by
            # the original jlens experiments.
            sources = {int(layer): params for layer, params in raw["pairs"].items()}
        else:
            raise ValueError(
                f"{self.bridge_path} is not a bridge file: expected W/b_x/b_y "
                "or a per-layer pairs mapping"
            )

        bridges: dict[int, dict[str, torch.Tensor]] = {}
        for layer, params in sources.items():
            missing = [key for key in ("W", "b_x", "b_y") if key not in params]
            if missing:
                raise ValueError(
                    f"{self.bridge_path} bridge for teacher layer {layer} is "
                    f"missing {missing}"
                )
            bridges[layer] = {
                key: params[key].to(device=ref.device, dtype=torch.float32)
                for key in ("W", "b_x", "b_y")
            }
            if layer >= 0 and "l_s" in params:
                self._bridge_student_layers[layer] = int(params["l_s"])
        self._bridges = bridges
        return bridges

    def _apply_bridge(
        self, h: torch.Tensor, teacher_layer: int, student_layer: int
    ) -> torch.Tensor:
        bridges = self._load_bridges(h)
        b = bridges.get(teacher_layer, bridges.get(-1))
        if b is None:
            available = sorted(layer for layer in bridges if layer >= 0)
            raise ValueError(
                f"{self.bridge_path} has no bridge for teacher layer "
                f"{teacher_layer}; available layers are {available}"
            )
        fitted_student_layer = self._bridge_student_layers.get(teacher_layer)
        if fitted_student_layer is not None and fitted_student_layer != student_layer:
            raise ValueError(
                f"{self.bridge_path} maps teacher layer {teacher_layer} to student "
                f"layer {fitted_student_layer}, but this arm maps it to {student_layer}"
            )
        return (h.float() - b["b_x"]) @ b["W"] + b["b_y"]

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        token_index = sample_aux_token_index(loss_mask, self.aux_max_tokens)
        terms = [
            hidden_mse(
                student.hidden[ls],
                self._apply_bridge(teacher.hidden[lt], lt, ls),
                loss_mask,
                aux_max_tokens=self.aux_max_tokens,
                token_index=token_index,
            )
            for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True)
        ]
        return self._mean_over_pairs(terms), {}


class SymmetricJLensArm(BaseArm):
    """Arm E — offline-fit Jacobian-lens readouts for teacher and student."""

    type: Literal["symmetric_jlens"] = "symmetric_jlens"
    student_jacobian_path: str
    """Lens fit on the frozen student initialization at the mapped layers."""
    teacher_jacobian_path: str
    """Lens fit on the frozen teacher at ``teacher_layers``."""

    _student_jacobians: dict[int, torch.Tensor] | None = PrivateAttr(default=None)
    _teacher_jacobians: dict[int, torch.Tensor] | None = PrivateAttr(default=None)

    def _load_jacobians(self, path: str, *, student: bool) -> dict[int, torch.Tensor]:
        jacobians = self._student_jacobians if student else self._teacher_jacobians
        if jacobians is None:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
            if "J" not in ckpt:
                raise ValueError(f"{path} is not a JacobianLens file")
            jacobians = ckpt["J"]
            if student:
                self._student_jacobians = jacobians
            else:
                self._teacher_jacobians = jacobians
        return jacobians

    @staticmethod
    def _require_layers(
        path: str,
        jacobians: dict[int, torch.Tensor],
        layers: tuple[int, ...],
        layer_role: str,
    ) -> None:
        missing = [layer for layer in layers if layer not in jacobians]
        if missing:
            raise ValueError(
                f"{path} has no Jacobian for {layer_role} {missing}; fitted source "
                f"layers are {sorted(jacobians)}. Re-fit the lens with "
                f"--source-layers matching this arm's mapped layers."
            )

    def aux_loss(self, student, teacher, loss_mask, spec, *, unembed_s, unembed_t):
        student_jacobians = self._load_jacobians(
            self.student_jacobian_path, student=True
        )
        teacher_jacobians = self._load_jacobians(
            self.teacher_jacobian_path, student=False
        )
        self._require_layers(
            self.student_jacobian_path,
            student_jacobians,
            spec.student_layers,
            "mapped student_layers",
        )
        self._require_layers(
            self.teacher_jacobian_path,
            teacher_jacobians,
            spec.teacher_layers,
            "teacher_layers",
        )

        token_index = sample_aux_token_index(loss_mask, self.aux_max_tokens)
        terms: list[torch.Tensor] = []
        for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True):
            student_hidden = student.hidden[ls]
            teacher_hidden = teacher.hidden[lt]
            student_jac = student_jacobians[ls].to(
                device=student_hidden.device, dtype=student_hidden.dtype
            )
            teacher_jac = teacher_jacobians[lt].to(
                device=teacher_hidden.device, dtype=teacher_hidden.dtype
            )
            terms.append(
                lens_kl(
                    student_hidden,
                    teacher_hidden,
                    lambda x, jac=student_jac: unembed_s(x @ jac.T),
                    lambda x, jac=teacher_jac: unembed_t(x @ jac.T),
                    loss_mask,
                    temperature=self.temperature,
                    aux_max_tokens=self.aux_max_tokens,
                    kl=self.kl,
                    token_index=token_index,
                )
            )
        return self._mean_over_pairs(terms), {}


Arm = Annotated[
    Annotated[LogitsArm, Tag("logits")]
    | Annotated[LogitLensArm, Tag("logit_lens")]
    | Annotated[JLensArm, Tag("jspace")]
    | Annotated[HiddenMseArm, Tag("hidden_mse")]
    | Annotated[SymmetricJLensArm, Tag("symmetric_jlens")],
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
    "SymmetricJLensArm",
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
        {
            "type": "symmetric_jlens",
            "student_jacobian_path": "student-lens.pt",
            "teacher_jacobian_path": "teacher-lens.pt",
        },
    ):
        print(f"parsed {conf['type']}: {type(parse_arm(conf)).__name__}")

    print("[green]arms smoke test passed[/green]")
