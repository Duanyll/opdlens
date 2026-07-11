"""Structured, validated payloads passed between training phases.

Frozen dataclasses with ``__post_init__`` shape checks — the opdlens analog of
flow_control's ``RewardResult``. Everything is single-sequence (micro batch = 1,
grad-accumulated), so tensors are ``[T, ...]`` with no batch axis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass(frozen=True)
class Example:
    """One benchmark row: a question and its gold answer."""

    question: str
    gold: str
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class CaptureSpec:
    """What a forward pass must retain, declared by ``arm.capture_spec()``.

    Final logits are always captured (the base OPD loss needs both sides' final
    logits). The variable part is which *intermediate* hidden layers to keep — the
    memory-discipline analog of a reward's ``_batch_fields``.
    """

    student_layers: tuple[int, ...] = ()
    teacher_layers: tuple[int, ...] = ()

    @property
    def needs_student_hidden(self) -> bool:
        return bool(self.student_layers)

    @property
    def needs_teacher_hidden(self) -> bool:
        return bool(self.teacher_layers)


@dataclass(frozen=True)
class Readout:
    """A model forward's captured outputs for one sequence.

    ``logits`` is ``[T, V]`` (final-layer logits) or None; ``hidden`` maps a layer
    index to its ``[T, d]`` hidden states (only the requested layers are present).
    """

    logits: torch.Tensor | None
    hidden: dict[int, torch.Tensor]

    def __post_init__(self) -> None:
        if self.logits is not None and self.logits.ndim != 2:
            raise ValueError(f"logits must be [T, V], got {tuple(self.logits.shape)}")
        for layer, h in self.hidden.items():
            if h.ndim != 2:
                raise ValueError(
                    f"hidden[{layer}] must be [T, d], got {tuple(h.shape)}"
                )


@dataclass(frozen=True)
class RolloutBatch:
    """One on-policy sample: prompt + student-sampled completion.

    ``input_ids`` is the full ``[T]`` sequence; ``loss_mask`` is ``[T]`` bool, True
    at **output positions** ``t`` whose predicted next token (``t+1``) is a
    completion token — i.e. ``prompt_len-1 <= t <= T-2``. All losses mask on this
    directly (no per-loss next-token shift), so base OPD, lens-KL and hidden-MSE
    supervise exactly the same positions.
    """

    input_ids: torch.Tensor
    prompt_len: int
    loss_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError(
                f"input_ids must be [T], got {tuple(self.input_ids.shape)}"
            )
        if self.loss_mask.shape != self.input_ids.shape:
            raise ValueError(
                "loss_mask must match input_ids shape, got "
                f"{tuple(self.loss_mask.shape)} vs {tuple(self.input_ids.shape)}"
            )
        if not 0 <= self.prompt_len <= self.input_ids.shape[0]:
            raise ValueError(
                f"prompt_len {self.prompt_len} out of range for T={self.input_ids.shape[0]}"
            )


@dataclass(frozen=True)
class EvalReport:
    """Per-benchmark eval result. ``accuracy`` = mean over questions of the soft
    (avg@k) score; ``se`` is its standard error."""

    benchmark: str
    n: int
    accuracy: float
    soft: list[float]
    se: float

    def __post_init__(self) -> None:
        if len(self.soft) != self.n:
            raise ValueError(f"soft length {len(self.soft)} does not match n={self.n}")
