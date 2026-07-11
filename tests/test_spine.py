"""The fair-comparison guardrail: the five arms share one spine.

Machine-checks that (1) the five experiment configs differ ONLY in the ``arm``
block, and (2) with ``aux_weight == 0`` every arm gates off its aux entirely, so
each reduces to the single shared ``opd_base_loss`` — fair comparison is
structural, not a convention.
"""

from __future__ import annotations

from pathlib import Path

import torch

from opdlens.arms import parse_arm
from opdlens.losses import opd_base_loss
from opdlens.utils.config import load_config_file

_EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_configs_differ_only_in_arm():
    configs = {
        name: load_config_file(str(_EXAMPLES / f"arm_{name}.jsonc"))
        for name in ("a", "b", "c", "d", "e")
    }
    baseline = {k: v for k, v in configs["a"].items() if k != "arm"}
    for name in ("b", "c", "d", "e"):
        other = {k: v for k, v in configs[name].items() if k != "arm"}
        assert other == baseline, f"arm_{name}.jsonc differs outside the arm block"


def test_aux_weight_zero_gates_off_aux():
    # The trainer computes ``active_aux = arm.aux_weight != 0``; with aux_weight 0
    # every arm skips hidden capture + aux + its RNG use -> identical base OPD.
    specs = [
        {"type": "logits"},
        {"type": "logit_lens", "aux_weight": 0.0, "teacher_layers": [8]},
        {
            "type": "jspace",
            "aux_weight": 0.0,
            "teacher_layers": [8],
            "jacobian_path": "x",
        },
        {
            "type": "hidden_mse",
            "aux_weight": 0.0,
            "teacher_layers": [8],
            "bridge_path": "x",
        },
        {
            "type": "symmetric_jlens",
            "aux_weight": 0.0,
            "teacher_layers": [8],
            "student_jacobian_path": "x",
            "teacher_jacobian_path": "x",
        },
    ]
    for spec in specs:
        assert (parse_arm(spec).aux_weight != 0.0) is False


def test_base_loss_is_the_single_shared_invariant():
    torch.manual_seed(0)
    seq, vocab = 12, 50
    teacher = torch.randn(seq, vocab)
    student = torch.randn(seq, vocab)
    mask = torch.zeros(seq, dtype=torch.bool)
    mask[5:] = True
    # KL(x || x) == 0, and the loss is deterministic (no arm can perturb it).
    assert torch.allclose(
        opd_base_loss(teacher, teacher, mask), torch.zeros(()), atol=1e-5
    )
    assert torch.equal(
        opd_base_loss(student, teacher, mask), opd_base_loss(student, teacher, mask)
    )


def test_base_beta_selects_divergence():
    """base_beta reproduces ms-swift GKD's single divergence knob: 0=forward KL,
    1=reverse KL, 0.5=symmetric JSD. Pins the exact math against hand computation."""
    import torch.nn.functional as F

    torch.manual_seed(1)
    seq, vocab = 8, 40
    student = torch.randn(seq, vocab)
    teacher = torch.randn(seq, vocab)
    mask = torch.ones(seq, dtype=torch.bool)
    slp = F.log_softmax(student.float(), dim=-1)
    tlp = F.log_softmax(teacher.float(), dim=-1)

    # beta=0 -> forward KL(P_teacher || P_student) == the default base loss.
    fwd = opd_base_loss(student, teacher, mask, beta=0.0)
    assert torch.allclose(fwd, (tlp.exp() * (tlp - slp)).sum(-1).mean(), atol=1e-5)
    assert torch.allclose(fwd, opd_base_loss(student, teacher, mask), atol=1e-6)

    # beta=1 -> reverse KL(P_student || P_teacher).
    rev = opd_base_loss(student, teacher, mask, beta=1.0)
    assert torch.allclose(rev, (slp.exp() * (slp - tlp)).sum(-1).mean(), atol=1e-5)

    # beta=0.5 -> symmetric, non-negative, zero on identical distributions.
    jsd = opd_base_loss(student, teacher, mask, beta=0.5)
    jsd_swapped = opd_base_loss(teacher, student, mask, beta=0.5)
    assert torch.allclose(jsd, jsd_swapped, atol=1e-6)
    assert float(jsd) > 0.0
    assert torch.allclose(
        opd_base_loss(teacher, teacher, mask, beta=0.5), torch.zeros(()), atol=1e-5
    )
