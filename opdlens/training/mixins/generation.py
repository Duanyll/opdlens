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

# torchrun exports these; the colocated vLLM V1 engine otherwise tries to bootstrap
# its own distributed init through torchrun's rendezvous (MASTER_ADDR/RANK/WORLD_SIZE)
# and hangs inside the container. They are cleared only around engine construction.
_TORCHRUN_DIST_ENV = (
    "RANK",
    "WORLD_SIZE",
    "LOCAL_RANK",
    "LOCAL_WORLD_SIZE",
    "GROUP_RANK",
    "ROLE_RANK",
    "ROLE_NAME",
    "ROLE_WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
    "TORCHELASTIC_RUN_ID",
    "TORCHELASTIC_RESTART_COUNT",
    "TORCHELASTIC_MAX_RESTARTS",
    "TORCHELASTIC_USE_AGENT_STORE",
    "TORCHELASTIC_ERROR_FILE",
)


class GenerationMixin(BaseTrainer):
    student: LanguageModel

    rollout_backend: Literal["vllm", "hf"] = "vllm"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_max_model_len: int = 2048
    vllm_enforce_eager: bool = True
    """Disable CUDA-graph capture in the colocated engine. ``True`` is the safe
    default (no graph memory, no capture step). ``False`` captures decode graphs —
    much faster decode for the small student, and compatible with the in-place
    ``apply_model(load_weights)`` weight sync (graphs replay from the persistent
    param buffers) — at the cost of extra capture memory, so it may need a higher
    ``vllm_gpu_memory_utilization``."""

    # Training-rollout sampling — SEPARATE from a benchmark's eval sampling and
    # from the base-KL / aux temperatures.
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_max_tokens: int = 1024

    # Chat-template control, applied identically to rollout AND eval so the two never
    # diverge. ``enable_thinking=False`` disables Qwen3.5's hybrid-thinking block (the
    # ms-swift GKD recipe runs with thinking off); ``None`` leaves the template default
    # (for models with no thinking toggle).
    enable_thinking: bool | None = None

    _llm: Any = None

    # ------------------------------- Lifecycle -------------------------------- #

    def load_student(self) -> None:
        self.student.load(self.device, trainable=True)

    def init_generation(self) -> None:
        if self.rollout_backend != "vllm":
            return
        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        # Colocated vLLM initializes its own (single-process, TP=1) torch.distributed
        # group. Inside the enroot/pyxis container the node's LAN IP is unreachable
        # across the net namespace, so vLLM's default ``get_ip()`` store hangs until
        # a 600 s TCPStore timeout. Pin the store to loopback — each rank's engine is
        # independent and single-node, so 127.0.0.1 is always correct here.
        os.environ.setdefault("VLLM_HOST_IP", "127.0.0.1")
        from vllm import LLM

        vllm_dtype: Any = _VLLM_DTYPE[self.student.dtype]
        # Each rank's colocated engine is an independent single-process TP=1 engine on
        # the already-selected local device; it must form its OWN group, so clear the
        # torchrun rendezvous env during construction. The trainer's process group was
        # created earlier and persists as a live object, so restoring the env afterward
        # keeps the trainer's own collectives working.
        saved_env = {
            k: os.environ.pop(k) for k in _TORCHRUN_DIST_ENV if k in os.environ
        }
        try:
            self._llm = LLM(
                model=self.student.model_id,
                dtype=vllm_dtype,
                gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                max_model_len=self.vllm_max_model_len,
                enforce_eager=self.vllm_enforce_eager,
                trust_remote_code=self.student.trust_remote_code,
            )
        finally:
            os.environ.update(saved_env)
        logger.info("Colocated vLLM engine initialized.")

    def sync_weights(self) -> None:
        """Stream current student weights into the colocated vLLM engine.

        Name mapping (archs whose HF and vLLM weight names disagree, e.g. Qwen3.5) and
        FSDP2 ``full_tensor()`` gathering are handled by ``LanguageModel.iter_vllm_weights``.
        vLLM's ``load_weights`` fuses qkv / gate_up and reshards to its TP layout.
        """
        if self.rollout_backend != "vllm" or self._llm is None:
            return
        weights = self.student.iter_vllm_weights()
        self._llm.apply_model(lambda m: m.load_weights(weights))

    # ------------------------------- Encoding --------------------------------- #

    def encode_prompt(self, benchmark: BaseBenchmark, example: Example) -> list[int]:
        # Render to text then tokenize explicitly (apply_chat_template's tokenize=True
        # return type varies; this always yields a clean list[int]). The template
        # already adds special tokens, so add_special_tokens=False.
        template_kwargs: dict[str, Any] = {}
        if self.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.enable_thinking
        text = self.student.tokenizer.apply_chat_template(
            benchmark.build_prompt(example),
            add_generation_prompt=True,
            tokenize=False,
            **template_kwargs,
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
