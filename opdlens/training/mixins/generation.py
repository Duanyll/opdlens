"""GenerationMixin — the student and its sampling engine.

Owns the student model and one ``generate`` used by BOTH rollout and eval (so the
two never diverge). Two backends behind one interface:

- ``vllm``: colocated in-process engine; ``sync_weights`` streams the (DDP: plain,
  FSDP2: ``full_tensor``) student weights into it via the validated
  ``apply_model(load_weights(...))`` path.
- ``hf``: ``model.generate`` in-process (same module, no sync needed) — for quick
  tests and vLLM-less environments.

``generate`` works in token-id space: prompt ids from ``encode_prompt`` (chat
template) go in, completion ids come out, so generation and the training forward
tokenize identically.
"""

from __future__ import annotations

import os
from typing import Any, Literal

import torch

from ...benchmarks import BaseBenchmark
from ...models import LanguageModel
from ...types import Example
from ...utils.logging import get_logger
from ..base import BaseTrainer

logger = get_logger(__name__)

_VLLM_DTYPE = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}


class GenerationMixin(BaseTrainer):
    student: LanguageModel

    rollout_backend: Literal["vllm", "hf"] = "vllm"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_max_model_len: int = 2048

    # Training-rollout sampling — SEPARATE from a benchmark's eval sampling and
    # from the base-KL / aux temperatures.
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_max_tokens: int = 1024

    _llm: Any = None

    # ------------------------------- Lifecycle -------------------------------- #

    def load_student(self) -> None:
        self.student.load(self.device, trainable=True)

    def init_generation(self) -> None:
        if self.rollout_backend != "vllm":
            return
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        from vllm import LLM

        vllm_dtype: Any = _VLLM_DTYPE[self.student.dtype]
        self._llm = LLM(
            model=self.student.model_id,
            dtype=vllm_dtype,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            max_model_len=self.vllm_max_model_len,
            enforce_eager=True,
            trust_remote_code=self.student.trust_remote_code,
        )
        logger.info("Colocated vLLM engine initialized.")

    def sync_weights(self) -> None:
        """Stream current student weights into the colocated vLLM engine.

        DDP params are plain tensors; FSDP2 params are ``DTensor`` and get
        all-gathered per-parameter via ``full_tensor()``. vLLM's ``load_weights``
        fuses qkv / gate_up and reshards to its own TP layout internally.
        """
        if self.rollout_backend != "vllm" or self._llm is None:
            return
        weights = (
            (name, param.full_tensor() if hasattr(param, "full_tensor") else param)
            for name, param in self.student.model.state_dict().items()
        )
        self._llm.apply_model(lambda m: m.load_weights(weights))

    # ------------------------------- Encoding --------------------------------- #

    def encode_prompt(self, benchmark: BaseBenchmark, example: Example) -> list[int]:
        # Render to text then tokenize explicitly (apply_chat_template's tokenize=True
        # return type varies; this always yields a clean list[int]). The template
        # already adds special tokens, so add_special_tokens=False.
        text = self.student.tokenizer.apply_chat_template(
            benchmark.build_prompt(example), add_generation_prompt=True, tokenize=False
        )
        encoded = self.student.tokenizer(text, add_special_tokens=False)
        return list(encoded["input_ids"])

    def decode(self, token_ids: list[int]) -> str:
        return self.student.tokenizer.decode(token_ids, skip_special_tokens=True)

    # ------------------------------ Generation -------------------------------- #

    def generate(
        self,
        prompt_ids: list[list[int]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int = 1,
    ) -> list[list[list[int]]]:
        """Return, per prompt, ``n`` completion token-id lists."""
        if not prompt_ids:
            return []
        if self.rollout_backend == "vllm":
            return self._generate_vllm(prompt_ids, temperature, top_p, max_tokens, n)
        return self._generate_hf(prompt_ids, temperature, top_p, max_tokens, n)

    def _generate_vllm(self, prompt_ids, temperature, top_p, max_tokens, n):
        from vllm import SamplingParams

        params = SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        prompts = [{"prompt_token_ids": ids} for ids in prompt_ids]
        outputs = self._llm.generate(prompts, params)
        return [[list(o.token_ids) for o in out.outputs] for out in outputs]

    def _generate_hf(self, prompt_ids, temperature, top_p, max_tokens, n):
        model = self.student.model
        tokenizer = self.student.tokenizer
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        results: list[list[list[int]]] = []
        for ids in prompt_ids:
            prompt = torch.tensor([ids], device=model.device)
            with torch.no_grad():
                out = model.generate(
                    prompt,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                    top_p=top_p,
                    max_new_tokens=max_tokens,
                    num_return_sequences=n,
                    pad_token_id=pad_id,
                )
            results.append([out[i, len(ids) :].tolist() for i in range(n)])
        return results
