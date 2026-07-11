"""Shared, arm-agnostic loss primitives.

``opd_base_loss`` is arm A and the single base loss every arm shares (the
invariant pinned by ``tests/test_spine.py``). ``lens_kl`` is the shared
vocab-space aux body for arms B/C — they differ only in the teacher readout they
pass in. ``hidden_mse`` is arm D's aux. All operate on one sequence and mask on
the ``[T]`` output-position ``loss_mask`` (see ``types.RolloutBatch``) with no
next-token shift.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F

KLDir = Literal["forward", "reverse"]


def map_student_layer(teacher_layer: int, n_student: int, n_teacher: int) -> int:
    """Pair a teacher block index to a student block index by depth fraction.

    ``l' = round((l+1) * nS / nT) - 1``, clamped to ``[0, nS-2]`` (the ``-2``
    excludes the post-final-norm hidden-states entry).
    """
    paired = round((teacher_layer + 1) * n_student / n_teacher) - 1
    return max(0, min(paired, n_student - 2))


def _slice_common_vocab(
    a: torch.Tensor, b: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    v = min(a.shape[-1], b.shape[-1])
    return a[..., :v], b[..., :v]


def _token_kl(
    ref_logits: torch.Tensor, other_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Per-position ``KL(ref || other)`` over the (temperature-scaled) vocab."""
    ref_logits, other_logits = _slice_common_vocab(ref_logits, other_logits)
    ref_logp = F.log_softmax(ref_logits.float() / temperature, dim=-1)
    other_logp = F.log_softmax(other_logits.float() / temperature, dim=-1)
    return (ref_logp.exp() * (ref_logp - other_logp)).sum(dim=-1)


def _directed_kl(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
    kl: KLDir,
) -> torch.Tensor:
    if kl == "forward":  # KL(teacher || student) — mode-covering (default)
        return _token_kl(teacher_logits, student_logits, temperature)
    return _token_kl(student_logits, teacher_logits, temperature)  # reverse


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def opd_base_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    kl: KLDir = "forward",
) -> torch.Tensor:
    """Plain OPD (arm A): masked-mean per-token KL on final logits.

    ``student_logits``/``teacher_logits`` are ``[T, V]`` (vocab may differ; sliced
    to the common size); ``loss_mask`` is ``[T]``.
    """
    per_pos = _directed_kl(teacher_logits, student_logits, temperature, kl)
    return _masked_mean(per_pos, loss_mask)


def _supervised_index(loss_mask: torch.Tensor, aux_max_tokens: int) -> torch.Tensor:
    idx = loss_mask.nonzero(as_tuple=False).squeeze(-1)
    if aux_max_tokens and idx.numel() > aux_max_tokens:
        sel = torch.randperm(idx.numel(), device=idx.device)[:aux_max_tokens]
        idx = idx[sel]
    return idx


def lens_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    unembed_s: Callable[[torch.Tensor], torch.Tensor],
    teacher_readout: Callable[[torch.Tensor], torch.Tensor],
    loss_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    aux_max_tokens: int = 512,
    kl: KLDir = "forward",
) -> torch.Tensor:
    """Shared vocab-space aux for arms B/C.

    Student side is always the logit-lens ``unembed_s(h_S)``. The ``teacher_readout``
    operator is the arm's single point of difference — logit-lens (B) is
    ``unembed_t``, Jacobian-lens (C) is ``h -> unembed_t(J·h)``. Supervised
    completion positions are subsampled to ``aux_max_tokens`` *before* either
    readout, so the unembed runs on ``[n, d]``, not ``[T, d]``.
    """
    idx = _supervised_index(loss_mask, aux_max_tokens)
    if idx.numel() == 0:
        return student_hidden.new_zeros(())
    student_logits = unembed_s(student_hidden[idx])  # [n, V]
    teacher_logits = teacher_readout(teacher_hidden[idx])  # [n, V]
    return _directed_kl(teacher_logits, student_logits, temperature, kl).mean()


def hidden_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Arm D: raw-hidden MSE (teacher side already mapped through the bridge).

    Both ``[T, d_s]``. Per-position mean over ``d``, masked-mean over positions.
    """
    per_pos = (student_hidden - teacher_hidden).pow(2).mean(dim=-1)
    return _masked_mean(per_pos, loss_mask)
