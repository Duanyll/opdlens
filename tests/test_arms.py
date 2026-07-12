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


def test_logit_lens_layers_share_one_token_sample(monkeypatch):
    calls = 0

    def reverse_randperm(n, *, device=None):
        nonlocal calls
        calls += 1
        return torch.arange(n - 1, -1, -1, device=device)

    monkeypatch.setattr(torch, "randperm", reverse_randperm)
    arm = parse_arm(
        {
            "type": "logit_lens",
            "aux_weight": 0.1,
            "aux_max_tokens": 3,
            "aux_token_policy": "shared",
            "teacher_layers": [2, 4],
        }
    )
    spec = arm.capture_spec(4, 6)
    rows = torch.arange(8, dtype=torch.float32).unsqueeze(1).expand(-1, 4)
    student = Readout(
        logits=None, hidden={layer: rows.clone() for layer in spec.student_layers}
    )
    teacher = Readout(
        logits=None, hidden={layer: rows.clone() for layer in spec.teacher_layers}
    )
    student_seen: list[torch.Tensor] = []
    teacher_seen: list[torch.Tensor] = []

    def student_readout(x):
        student_seen.append(x[:, 0].clone())
        return x[:, :1].expand(-1, 8)

    def teacher_readout(x):
        teacher_seen.append(x[:, 0].clone())
        return x[:, :1].expand(-1, 8)

    arm.aux_loss(
        student,
        teacher,
        torch.ones(8, dtype=torch.bool),
        spec,
        unembed_s=student_readout,
        unembed_t=teacher_readout,
    )
    expected = torch.tensor([7.0, 6.0, 5.0])
    assert calls == 1
    assert all(torch.equal(seen, expected) for seen in student_seen + teacher_seen)

    compat_arm = parse_arm(
        {
            "type": "logit_lens",
            "aux_weight": 0.1,
            "aux_max_tokens": 3,
            "teacher_layers": [2, 4],
        }
    )
    compat_arm.aux_loss(
        student,
        teacher,
        torch.ones(8, dtype=torch.bool),
        spec,
        unembed_s=student_readout,
        unembed_t=teacher_readout,
    )
    assert compat_arm.aux_token_policy == "compat"
    assert calls == 3


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


def test_hidden_mse_caps_shared_tokens_and_uses_fp32(tmp_path, monkeypatch):
    calls = 0

    def reverse_randperm(n, *, device=None):
        nonlocal calls
        calls += 1
        return torch.arange(n - 1, -1, -1, device=device)

    monkeypatch.setattr(torch, "randperm", reverse_randperm)
    d, seq = 4, 6
    path = tmp_path / "bridge.pt"
    torch.save({"W": torch.eye(d), "b_x": torch.zeros(d), "b_y": torch.zeros(d)}, path)
    arm = parse_arm(
        {
            "type": "hidden_mse",
            "aux_weight": 1.94,
            "aux_max_tokens": 2,
            "aux_token_policy": "shared",
            "mse_dtype": "fp32",
            "teacher_layers": [2, 4],
            "bridge_path": str(path),
        }
    )
    spec = arm.capture_spec(4, 6)
    zeros = torch.zeros(seq, d, dtype=torch.bfloat16)
    rows = (
        torch.arange(seq, dtype=torch.float32)
        .unsqueeze(1)
        .expand(-1, d)
        .to(torch.bfloat16)
    )
    student = Readout(
        logits=None, hidden={layer: zeros.clone() for layer in spec.student_layers}
    )
    teacher = Readout(
        logits=None, hidden={layer: rows.clone() for layer in spec.teacher_layers}
    )
    aux, _ = arm.aux_loss(
        student,
        teacher,
        torch.ones(seq, dtype=torch.bool),
        spec,
        unembed_s=None,
        unembed_t=None,
    )
    assert calls == 1
    assert aux.dtype == torch.float32
    assert torch.allclose(aux, torch.tensor((5.0**2 + 4.0**2) / 2))

    compat_arm = parse_arm(
        {
            "type": "hidden_mse",
            "aux_weight": 1.94,
            "aux_max_tokens": 2,
            "teacher_layers": [2, 4],
            "bridge_path": str(path),
        }
    )
    compat_aux, _ = compat_arm.aux_loss(
        student,
        teacher,
        torch.ones(seq, dtype=torch.bool),
        spec,
        unembed_s=None,
        unembed_t=None,
    )
    assert compat_arm.aux_token_policy == "compat"
    assert compat_arm.mse_dtype == "input"
    assert calls == 1
    assert compat_aux.dtype == torch.bfloat16
    assert float(compat_aux) == pytest.approx(
        sum(i**2 for i in range(seq)) / seq, rel=0.01
    )


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


def test_lens_kl_top_k_truncates_to_teacher_head():
    torch.manual_seed(0)
    d, v, seq = 8, 40, 6
    us, ut = torch.nn.Linear(d, v), torch.nn.Linear(d, v)
    student_h, teacher_h = torch.randn(seq, d), torch.randn(seq, d)
    mask = torch.ones(seq, dtype=torch.bool)

    full = lens_kl(student_h, teacher_h, us, ut, mask, aux_max_tokens=0)
    # k >= V is a no-op: the KL is permutation-invariant over the full vocab.
    for k in (v, v + 100):
        same = lens_kl(student_h, teacher_h, us, ut, mask, aux_max_tokens=0, top_k=k)
        assert torch.allclose(full, same, atol=1e-5)
    # A real truncation restricts the support, stays finite, and shifts the value.
    trunc = lens_kl(student_h, teacher_h, us, ut, mask, aux_max_tokens=0, top_k=5)
    assert torch.isfinite(trunc)
    assert not torch.allclose(full, trunc)


def test_logit_lens_arm_threads_top_k():
    d, v = 16, 32
    us, ut = torch.nn.Linear(d, v), torch.nn.Linear(d, v)
    arm = parse_arm(
        {
            "type": "logit_lens",
            "aux_weight": 0.1,
            "aux_top_k": 8,
            "aux_max_tokens": 0,
            "teacher_layers": [2, 4],
        }
    )
    assert arm.aux_top_k == 8
    spec = arm.capture_spec(4, 6)
    student, teacher, mask = _readouts(spec, d=d)
    aux, _ = arm.aux_loss(student, teacher, mask, spec, unembed_s=us, unembed_t=ut)
    expected = torch.stack(
        [
            lens_kl(
                student.hidden[ls],
                teacher.hidden[lt],
                us,
                ut,
                mask,
                aux_max_tokens=0,
                top_k=8,
            )
            for lt, ls in zip(spec.teacher_layers, spec.student_layers, strict=True)
        ]
    ).mean()
    assert torch.allclose(aux, expected)
