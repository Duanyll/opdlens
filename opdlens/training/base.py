"""BaseTrainer — distributed lifecycle shared by every trainer.

Owns rank/device state, seeding, RNG (de)serialization, and manual data-parallel
gradient reduction (plain DDP-equivalent: each rank forwards its own sequences,
grads are all-reduced before the optimizer step). FSDP2 replaces the manual
reduction later; for the MVP this keeps params as plain tensors so the vLLM weight
sync is trivial.
"""

from __future__ import annotations

import functools
import os
import pickle
import random
from collections.abc import Iterable

import numpy as np
import torch
import torch.distributed as dist
from pydantic import BaseModel, ConfigDict, Field

from ..config import LaunchConfig
from ..utils.logging import get_logger

logger = get_logger(__name__)


class BaseTrainer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    launch: LaunchConfig = Field(default_factory=LaunchConfig)
    seed: int = 42

    _world_size: int = 1
    _rank: int = 0
    _local_rank: int = 0

    # -------------------------------- Properties ------------------------------ #

    @property
    def world_size(self) -> int:
        return self._world_size

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def local_rank(self) -> int:
        return self._local_rank

    @property
    def is_main_process(self) -> bool:
        return self._rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self._local_rank == 0

    @property
    def device(self) -> torch.device:
        # Each rank restricts CUDA_VISIBLE_DEVICES to its own GPU (see init_distributed),
        # so the local GPU is always cuda:0.
        if torch.cuda.is_available():
            return torch.device("cuda", 0)
        return torch.device("cpu")

    # ------------------------------- Lifecycle -------------------------------- #

    def init_distributed(self) -> None:
        self._world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self._rank = int(os.environ.get("RANK", "0"))
        self._local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Pin this rank to ONE visible GPU (exposed as cuda:0) BEFORE any CUDA init. The
        # colocated vLLM uniproc engine always binds cuda:0 regardless of env or current
        # device, so per-rank CUDA_VISIBLE_DEVICES is the only way to place N engines on
        # N GPUs; all downstream code then treats cuda:0 as this rank's GPU.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        gpu_ids = visible.split(",") if visible else [str(self._local_rank)]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids[
            min(self._local_rank, len(gpu_ids) - 1)
        ]
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        if self._world_size > 1:
            # torchrun exports MASTER_ADDR as the node FQDN, which is unreachable
            # across the enroot/pyxis container's net namespace and hangs the NCCL
            # rendezvous. opdlens always launches single-node (torchrun --standalone),
            # so loopback is the correct — and reachable — rendezvous address.
            os.environ["MASTER_ADDR"] = "127.0.0.1"
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        logger.info(
            "Distributed init: world_size=%d rank=%d local_rank=%d device=%s",
            self._world_size,
            self._rank,
            self._local_rank,
            self.device,
        )

    def cleanup(self) -> None:
        # Destroy whatever default process group is live at exit. Even at
        # world_size == 1 (e.g. ``opdlens eval`` on a single GPU) the colocated vLLM
        # engine initializes a TP=1 group; leaving it undestroyed emits the NCCL
        # shutdown warning and makes torchrun exit non-zero.
        if dist.is_available() and dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                logger.exception("destroy_process_group() failed during cleanup")

    def run(self) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not implement run().")

    # --------------------------------- Seeding -------------------------------- #

    def set_seed(self) -> None:
        seed = self.seed + self._rank
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        logger.info("Seed set to %d (rank %d)", seed, self._rank)

    def get_rng_state_bytes(self) -> bytes:
        state: dict[str, object] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state(self.device)
        return pickle.dumps(state)

    def load_rng_state_bytes(self, data: bytes | None) -> None:
        if not data:
            return
        state = pickle.loads(data)
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state(state["cuda"], self.device)

    # --------------------------- Data-parallel grads -------------------------- #

    def reduce_gradients(self, parameters: Iterable[torch.nn.Parameter]) -> None:
        """Average gradients across data-parallel ranks (manual DDP)."""
        if self._world_size <= 1:
            return
        for param in parameters:
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
                param.grad /= self._world_size


def distributed_main(func):
    """Wrap ``run()`` with distributed init + guaranteed cleanup."""

    @functools.wraps(func)
    def wrapper(self: BaseTrainer, *args, **kwargs):
        try:
            self.init_distributed()
            return func(self, *args, **kwargs)
        except Exception:
            logger.exception("Uncaught exception in distributed run()")
            raise
        finally:
            self.cleanup()

    return wrapper
