"""The language model wrapper — one config+behavior class for student and teacher.

Both sides of the distillation are ``LanguageModel`` instances of the same HF
CausalLM family (shared tokenizer/vocab, the standard OPD setup). The trainer
wraps the student in DDP and keeps the teacher frozen; grad behavior is the
caller's context (the student's ``unembed`` shares the live ``lm_head``/final
norm, so the logit-lens keeps gradient; the teacher is called under ``no_grad``).
"""

from __future__ import annotations

from collections.abc import Iterator
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


def _full_tensor(param: torch.Tensor) -> torch.Tensor:
    """All-gather an FSDP2 ``DTensor`` to a plain tensor; pass plain tensors through."""
    gather = getattr(param, "full_tensor", None)
    return gather() if gather is not None else param


class LoraSpec(BaseModel):
    """LoRA adapter config for the student. When set, only the adapter trains; the
    base model is frozen. Defaults mirror ms-swift's GKD LoRA recipe (rank 8, alpha
    32, all-linear). The colocated vLLM engine is synced with MERGED weights, so the
    reproduction stays a drop-in swap of ``full`` fine-tune for ``lora`` regularized."""

    model_config = ConfigDict(extra="forbid")

    r: int = 8
    alpha: int = 32
    dropout: float = 0.05
    target_modules: str | list[str] = "all-linear"


class LanguageModel(BaseModel):
    """An HF CausalLM plus its tokenizer, with layer-capturing forward + unembed."""

    model_config = ConfigDict(extra="forbid")

    model_id: str
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    attn_impl: str = "sdpa"
    trust_remote_code: bool = True
    lora: LoraSpec | None = None
    """When set (student only), wrap the model in a LoRA adapter and train only it."""

    _model: Any = PrivateAttr(default=None)
    _tokenizer: Any = PrivateAttr(default=None)
    _final_norm: Any = PrivateAttr(default=None)
    _lm_head: Any = PrivateAttr(default=None)
    _vllm_name_map: dict[str, str] | None = PrivateAttr(default=None)

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
        if self.lora is not None and trainable:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=self.lora.r,
                    lora_alpha=self.lora.alpha,
                    lora_dropout=self.lora.dropout,
                    target_modules=self.lora.target_modules,
                    task_type="CAUSAL_LM",
                ),
            )
            logger.info("Wrapped %s in LoRA (r=%d).", self.model_id, self.lora.r)
        model.to(device)
        if trainable:
            model.train()
        else:
            model.requires_grad_(False)
            model.eval()
        self._model = model
        # Grab the live final-norm + lm_head so ``unembed`` shares parameters with the
        # model (grad-preserving for the student logit-lens). Under LoRA these live on
        # the frozen base model — unembed is never a LoRA target.
        base = self._base_model()
        self._final_norm = base.model.norm
        self._lm_head = base.get_output_embeddings()

    def _base_model(self) -> Any:
        """The underlying HF model, unwrapping the LoRA adapter if present."""
        if self.lora is not None and hasattr(self._model, "get_base_model"):
            return self._model.get_base_model()
        return self._model

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

    # ----------------------------- vLLM weight sync --------------------------- #

    def iter_vllm_weights(self) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield ``(vllm_name, tensor)`` for streaming the live student weights into
        the colocated vLLM engine.

        Most architectures need no rename, but some HF ``AutoModelForCausalLM`` models
        disagree with their vLLM counterpart on weight names — e.g. Qwen3.5 is natively
        multimodal (``Qwen3_5ForConditionalGeneration``): vLLM keeps the full model, so
        the text decoder lives under ``model.language_model.`` and the tied ``lm_head``
        is dropped, whereas HF's causal-LM view exposes ``model.`` + ``lm_head``. The
        rename map is derived once from the on-disk checkpoint names vLLM loads from.
        Under LoRA, the adapter is merged into the base weights for the duration of
        the sync (then unmerged), so vLLM always receives full merged weights.
        FSDP2 params are ``DTensor`` and get all-gathered via ``full_tensor()``.
        """
        merged = self.lora is not None
        if merged:
            self._model.merge_adapter()
        try:
            state = self._clean_state_dict()
            if self._vllm_name_map is None:
                self._vllm_name_map = self._build_vllm_name_map(list(state))
            name_map = self._vllm_name_map
            for name, param in state.items():
                target = name_map.get(name)
                if target is None:
                    continue  # no checkpoint entry (tied lm_head) — vLLM ties it too
                yield target, _full_tensor(param)
        finally:
            if merged:
                self._model.unmerge_adapter()

    def _clean_state_dict(self) -> dict[str, torch.Tensor]:
        """State dict keyed by clean HF names. Plain models pass through; a LoRA model
        (call after ``merge_adapter``) has peft's ``base_model.model.`` prefix and
        ``.base_layer`` infix stripped and its ``lora_*`` tensors dropped, leaving the
        merged base weights under their original names."""
        raw = self._model.state_dict()
        if self.lora is None:
            return raw
        clean: dict[str, torch.Tensor] = {}
        for name, tensor in raw.items():
            if ".lora_" in name or "lora_embedding" in name:
                continue
            key = name.removeprefix("base_model.model.").replace(".base_layer.", ".")
            clean[key] = tensor
        return clean

    def _build_vllm_name_map(self, hf_names: list[str]) -> dict[str, str]:
        ckpt = self._checkpoint_weight_names()
        if not ckpt or set(hf_names) <= ckpt:
            return {name: name for name in hf_names}  # names already agree
        # The wrapper differs only by a leading prefix; recover it from the deepest
        # HF name (a unique suffix in the checkpoint), then apply it to all params.
        probe = max(
            (n for n in hf_names if n.startswith("model.")), key=len, default=""
        )
        tail = probe[len("model.") :]
        matches = [c for c in ckpt if c.endswith(tail)] if tail else []
        if len(matches) != 1:
            raise RuntimeError(
                f"Cannot map {self.model_id} weights to vLLM: no unique checkpoint "
                f"name matches {probe!r}."
            )
        new_prefix = matches[0][: len(matches[0]) - len(tail)]
        name_map: dict[str, str] = {}
        for name in hf_names:
            if name in ckpt:
                name_map[name] = name
            elif (
                name.startswith("model.")
                and (cand := new_prefix + name[len("model.") :]) in ckpt
            ):
                name_map[name] = cand
            # else: absent from checkpoint (tied lm_head) — skip
        logger.info(
            "vLLM weight map for %s: %d/%d params (prefix 'model.' -> %r)",
            self.model_id,
            len(name_map),
            len(hf_names),
            new_prefix,
        )
        return name_map

    def _checkpoint_weight_names(self) -> set[str]:
        """The weight names in the on-disk checkpoint (what vLLM's ``load_weights``
        consumes). Empty set if it can't be resolved (falls back to an identity map)."""
        import json

        from huggingface_hub import hf_hub_download

        try:
            index = hf_hub_download(
                self.model_id, "model.safetensors.index.json", local_files_only=True
            )
            with open(index) as f:
                return set(json.load(f)["weight_map"])
        except Exception:
            pass
        try:
            from safetensors import safe_open

            path = hf_hub_download(
                self.model_id, "model.safetensors", local_files_only=True
            )
            with safe_open(path, framework="pt") as f:
                return set(f.keys())
        except Exception:
            logger.warning(
                "Could not read checkpoint weight names for %s; using identity vLLM map.",
                self.model_id,
            )
            return set()


if __name__ == "__main__":
    from rich import print

    lm = LanguageModel(model_id="Qwen/Qwen3-0.6B", dtype="bf16")
    print(lm)
    print(f"torch_dtype: {lm.torch_dtype}")
