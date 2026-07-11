"""Each arm's aux_loss produces a finite scalar on toy tensors."""

from __future__ import annotations

import torch

from opdlens.arms import parse_arm
from opdlens.types import CaptureSpec, Readout


def _readouts(spec: CaptureSpec, d: int = 16, seq: int = 10):
    student = Readout(
        logits=None, hidden={ls: torch.randn(seq, d) for ls in spec.student_layers}
    )
    teacher = Readout(
        logits=None, hidden={lt: torch.randn(seq, d) for lt in spec.teacher_layers}
    )
    mask = torch.zeros(seq, dtype=torch.bool)
    mask[4:] = True
    return student, teacher, mask


def test_logits_arm_returns_zero():
    arm = parse_arm({"type": "logits"})
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=None, unembed_t=None)
    assert float(aux) == 0.0


def test_logit_lens_arm_finite():
    d, v = 16, 32
    us, ut = torch.nn.Linear(d, v), torch.nn.Linear(d, v)
    arm = parse_arm({"type": "logit_lens", "aux_weight": 0.1, "teacher_layers": [2, 4]})
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec, d=d)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=us, unembed_t=ut)
    assert torch.isfinite(aux) and aux.ndim == 0


def test_jlens_arm_finite(tmp_path):
    d, v = 16, 32
    us, ut = torch.nn.Linear(d, v), torch.nn.Linear(d, v)
    path = tmp_path / "lens.pt"
    torch.save(
        {
            "J": {2: torch.eye(d), 4: torch.eye(d)},
            "n_prompts": 1,
            "source_layers": [2, 4],
            "d_model": d,
        },
        path,
    )
    arm = parse_arm(
        {
            "type": "jspace",
            "aux_weight": 0.1,
            "teacher_layers": [2, 4],
            "jacobian_path": str(path),
        }
    )
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec, d=d)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=us, unembed_t=ut)
    assert torch.isfinite(aux)


def test_hidden_mse_arm_finite(tmp_path):
    d = 16
    path = tmp_path / "bridge.pt"
    torch.save({"W": torch.eye(d), "b_x": torch.zeros(d), "b_y": torch.zeros(d)}, path)
    arm = parse_arm(
        {
            "type": "hidden_mse",
            "aux_weight": 0.1,
            "teacher_layers": [2, 4],
            "bridge_path": str(path),
        }
    )
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec, d=d)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=None, unembed_t=None)
    assert torch.isfinite(aux)
