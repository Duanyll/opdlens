"""Shared, arm-agnostic loss primitives.

``opd_base_loss`` is arm A and the single base loss every arm shares (the
invariant pinned by ``tests/test_spine.py``). ``lens_kl`` is the shared
vocab-space aux body for arms B/C/E — they differ only in the readouts they pass
in. ``hidden_mse`` is arm D's aux. All operate on one sequence and mask on
the ``[T]`` output-position ``loss_mask`` (see ``types.RolloutBatch``) with no
next-token shift.
"""

from __future__ import annotations

import math
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


def _generalized_jsd(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    temperature: float,
    beta: float,
) -> torch.Tensor:
    """Per-position generalized Jensen-Shannon divergence (ms-swift / TRL GKD).

    ``beta`` interpolates the divergence, matching ms-swift's single ``--beta`` knob:
    ``0`` → forward ``KL(P_teacher ‖ P_student)`` (mode-covering distillation),
    ``1`` → reverse ``KL(P_student ‖ P_teacher)``, and ``0<beta<1`` → JSD with mixture
    ``M = beta·P_teacher + (1-beta)·P_student`` and loss
    ``beta·KL(P_T ‖ M) + (1-beta)·KL(P_S ‖ M)`` (``beta=0.5`` is the symmetric JSD).
    Temperature scales both logits (no ``T²`` factor, as in ms-swift). Returns ``[T]``.
    """
    t, s = _slice_common_vocab(teacher_logits, student_logits)
    t_logp = F.log_softmax(t.float() / temperature, dim=-1)
    s_logp = F.log_softmax(s.float() / temperature, dim=-1)
    if beta == 0.0:
        return (t_logp.exp() * (t_logp - s_logp)).sum(dim=-1)  # KL(P_T ‖ P_S)
    if beta == 1.0:
        return (s_logp.exp() * (s_logp - t_logp)).sum(dim=-1)  # KL(P_S ‖ P_T)
    m_logp = torch.logsumexp(
        torch.stack([t_logp + math.log(beta), s_logp + math.log1p(-beta)]), dim=0
    )
    kl_t = (t_logp.exp() * (t_logp - m_logp)).sum(dim=-1)  # KL(P_T ‖ M)
    kl_s = (s_logp.exp() * (s_logp - m_logp)).sum(dim=-1)  # KL(P_S ‖ M)
    return beta * kl_t + (1.0 - beta) * kl_s


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.to(values.dtype)
    return (values * m).sum() / m.sum().clamp_min(1.0)


def opd_base_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    beta: float = 0.0,
) -> torch.Tensor:
    """Plain OPD (arm A): masked-mean per-token divergence on final logits.

    ``student_logits``/``teacher_logits`` are ``[T, V]`` (vocab may differ; sliced
    to the common size); ``loss_mask`` is ``[T]``. ``beta`` selects the divergence
    (``0`` forward KL — the default and the README's "final-logit KL"; ``0.5`` the
    JSD ms-swift GKD reports; ``1`` reverse KL), reduced as a per-token masked mean.

    Reduction stays a fully static ``masked_mean`` (no ``nonzero``/boolean gather) so
    the hot path launches zero data-dependent GPU→CPU syncs; sequence length is bounded
    by ``rollout_max_tokens`` instead, which is what keeps the ``[T, V]`` (V≈151k) fp32
    divergence within memory.
    """
    per_pos = _generalized_jsd(teacher_logits, student_logits, temperature, beta)
    return _masked_mean(per_pos, loss_mask)


def sample_aux_token_index(
    loss_mask: torch.Tensor, aux_max_tokens: int
) -> torch.Tensor:
    """Select one completion-token subset to share across supervised layers."""
    idx = loss_mask.nonzero(as_tuple=False).squeeze(-1)
    if aux_max_tokens and idx.numel() > aux_max_tokens:
        sel = torch.randperm(idx.numel(), device=idx.device)[:aux_max_tokens]
        idx = idx[sel]
    return idx


def lens_kl(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    student_readout: Callable[[torch.Tensor], torch.Tensor],
    teacher_readout: Callable[[torch.Tensor], torch.Tensor],
    loss_mask: torch.Tensor,
    *,
    temperature: float = 1.0,
    aux_max_tokens: int = 512,
    kl: KLDir = "forward",
    token_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shared vocab-space aux for arms B/C/E.

    B uses plain logit-lens readouts on both sides, C transports only the teacher
    hidden state, and E transports both sides through their offline-fit Jacobians.
    Supervised completion positions are subsampled to ``aux_max_tokens`` *before*
    either readout, so the unembed runs on ``[n, d]``, not ``[T, d]``.
    """
    idx = (
        sample_aux_token_index(loss_mask, aux_max_tokens)
        if token_index is None
        else token_index
    )
    if idx.numel() == 0:
        return student_hidden.new_zeros(())
    student_logits = student_readout(student_hidden[idx])  # [n, V]
    teacher_logits = teacher_readout(teacher_hidden[idx])  # [n, V]
    return _directed_kl(teacher_logits, student_logits, temperature, kl).mean()


def hidden_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    aux_max_tokens: int = 512,
    token_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Arm D: raw-hidden MSE (teacher side already mapped through the bridge).

    Both ``[T, d_s]``. Compute in fp32, then mean over ``d`` and over one shared
    completion-token subset.
    """
    idx = (
        sample_aux_token_index(loss_mask, aux_max_tokens)
        if token_index is None
        else token_index
    )
    if idx.numel() == 0:
        return student_hidden.float().sum() * 0.0
    residual = student_hidden[idx].float() - teacher_hidden[idx].float()
    return residual.pow(2).mean(dim=-1).mean()
