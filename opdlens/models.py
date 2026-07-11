"""The language model wrapper — one config+behavior class for student and teacher.

Both sides of the distillation are ``LanguageModel`` instances of the same HF
CausalLM family (shared tokenizer/vocab, the standard OPD setup). The trainer
wraps the student in DDP and keeps the teacher frozen; grad behavior is the
caller's context (the student's ``unembed`` shares the live ``lm_head``/final
norm, so the logit-lens keeps gradient; the teacher is called under ``no_grad``).
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .types import Readout
from .utils.logging import get_logger

logger = get_logger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


class LanguageModel(BaseModel):
    """An HF CausalLM plus its tokenizer, with layer-capturing forward + unembed."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_impl: str = "sdpa"
    trust_remote_code: bool = True

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _final_norm: Any = PrivateAttr(default=None)
    _lm_head: Any = PrivateAttr(default=None)

    # ------------------------------- Properties ------------------------------- #

    @property
    def torch_dtype(self) -> torch.dtype:
        return _DTYPES[self.dtype]

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def num_layers(self) -> int:
        return int(self._model.config.num_hidden_layers)

    @property
    def hidden_size(self) -> int:
        return int(self._model.config.hidden_size)

    @property
    def vocab_size(self) -> int:
        return int(self._model.config.vocab_size)

    # --------------------------------- Loading -------------------------------- #

    def load(self, device: torch.device, *, trainable: bool = True) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading %s (dtype=%s, attn=%s, trainable=%s)",
            self.model_id,
            self.dtype,
            self.attn_impl,
            trainable,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        model: Any = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=self.torch_dtype,
            attn_implementation=self.attn_impl,
            trust_remote_code=self.trust_remote_code,
        )
        model.to(device)
        if trainable:
            model.train()
        else:
            model.requires_grad_(False)
            model.eval()
        self._model = model
        # Grab the live final-norm + lm_head so ``unembed`` shares parameters with
        # the model (grad-preserving for the student logit-lens).
        self._final_norm = model.model.norm
        self._lm_head = model.get_output_embeddings()

    # --------------------------------- Behavior ------------------------------- #

    def unembed(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project hidden states ``[..., d]`` to vocab logits ``[..., V]`` through
        the model's own final norm + lm_head (the logit-lens readout)."""
        return self._lm_head(self._final_norm(hidden))

    def forward_capture(
        self,
        input_ids: torch.Tensor,
        *,
        layers: tuple[int, ...] = (),
        need_logits: bool = True,
    ) -> Readout:
        """Forward one ``[T]`` sequence, returning final logits (if requested) and
        the requested intermediate hidden layers.

        Block ``l``'s output is HF ``hidden_states[l + 1]`` (index 0 is the
        embedding output). Only the requested layers are retained.
        """
        out = self._model(
            input_ids=input_ids.unsqueeze(0),
            output_hidden_states=bool(layers),
            use_cache=False,
        )
        logits = out.logits[0] if need_logits else None
        hidden: dict[int, torch.Tensor] = {}
        if layers:
            hs = out.hidden_states
            for layer in layers:
                hidden[layer] = hs[layer + 1][0]
        return Readout(logits=logits, hidden=hidden)


if __name__ == "__main__":
    from rich import print

    lm = LanguageModel(model_id="Qwen/Qwen3-0.6B", dtype="bf16")
    print(lm)
    print(f"torch_dtype: {lm.torch_dtype}")
