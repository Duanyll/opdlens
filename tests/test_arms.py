"""Each arm's aux_loss produces a finite scalar on toy tensors."""

from __future__ import annotations

import pytest
import torch

from opdlens.arms import parse_arm
from opdlens.losses import lens_kl
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


def test_jlens_arm_reports_missing_layer(tmp_path):
    path = tmp_path / "lens.pt"
    torch.save({"J": {2: torch.eye(16)}}, path)
    arm = parse_arm(
        {
            "type": "jspace",
            "aux_weight": 0.1,
            "teacher_layers": [2, 4],
            "jacobian_path": str(path),
        }
    )
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec)
    unembed = torch.nn.Linear(16, 32)
    with pytest.raises(ValueError, match=r"teacher_layers \[4\]"):
        arm.aux_loss(
            student,
            teacher,
            mask,
            spec,
            unembed_s=unembed,
            unembed_t=unembed,
        )


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


def test_hidden_mse_arm_supports_per_layer_bridge(tmp_path):
    d = 16
    path = tmp_path / "bridge.pt"
    torch.save(
        {
            "pairs": {
                2: {
                    "l_s": 1,
                    "W": torch.eye(d),
                    "b_x": torch.zeros(d),
                    "b_y": torch.zeros(d),
                },
                4: {
                    "l_s": 2,
                    "W": torch.eye(d),
                    "b_x": torch.zeros(d),
                    "b_y": torch.zeros(d),
                },
            }
        },
        path,
    )
    arm = parse_arm(
        {
            "type": "hidden_mse",
            "aux_weight": 0.1,
            "teacher_layers": [2, 4],
            "bridge_path": str(path),
        }
    )
    spec = arm.capture_spec(4, 6)
    assert spec.student_layers == (1, 2)
    student, teacher, mask = _readouts(spec, d=d)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=None, unembed_t=None)
    assert torch.isfinite(aux)


def test_symmetric_jlens_arm_applies_both_lenses(tmp_path):
    d_student, d_teacher, vocab = 12, 16, 32
    unembed_s = torch.nn.Linear(d_student, vocab)
    unembed_t = torch.nn.Linear(d_teacher, vocab)
    student_path = tmp_path / "student-lens.pt"
    teacher_path = tmp_path / "teacher-lens.pt"

    arm = parse_arm(
        {
            "type": "symmetric_jlens",
            "aux_weight": 0.1,
            "aux_max_tokens": 0,
            "teacher_layers": [2, 4],
            "student_jacobian_path": str(student_path),
            "teacher_jacobian_path": str(teacher_path),
        }
    )
    spec = arm.capture_spec(4, 6)
    student_jacobians = {
        layer: 2 * torch.eye(d_student) for layer in spec.student_layers
    }
    teacher_jacobians = {
        layer: 3 * torch.eye(d_teacher) for layer in spec.teacher_layers
    }
    torch.save({"J": student_jacobians}, student_path)
    torch.save({"J": teacher_jacobians}, teacher_path)

    student = Readout(
        logits=None,
        hidden={layer: torch.randn(10, d_student) for layer in spec.student_layers},
    )
    teacher = Readout(
        logits=None,
        hidden={layer: torch.randn(10, d_teacher) for layer in spec.teacher_layers},
    )
    mask = torch.arange(10) >= 4
    aux, _ = arm.aux_loss(
        student,
        teacher,
        mask,
        spec,
        unembed_s=unembed_s,
        unembed_t=unembed_t,
    )
    expected_terms = [
        lens_kl(
            student.hidden[ls],
            teacher.hidden[lt],
            lambda x, jac=student_jacobians[ls]: unembed_s(x @ jac.T),
            lambda x, jac=teacher_jacobians[lt]: unembed_t(x @ jac.T),
            mask,
            aux_max_tokens=0,
        )
        for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True)
    ]
    assert torch.allclose(aux, torch.stack(expected_terms).mean())
