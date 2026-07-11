"""CheckpointingMixin — simple rank-0 snapshotting for DDP-replicated params.

The MVP keeps params replicated (manual DDP), so rank 0 holds the full state and
a plain ``torch.save`` suffices; all ranks reload the same file on resume. (DCP +
sharded save is the FSDP2 follow-up.) The leaf trainer implements ``state_dict`` /
``load_state_dict``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from ...utils.logging import get_logger
from ..base import BaseTrainer

logger = get_logger(__name__)

_STEP_DIR = re.compile(r"^step_(\d+)$")


class CheckpointingMixin(BaseTrainer):
    checkpoint_root: str = "./checkpoints"
    checkpoint_interval: int = 0
    """Save every N steps (0 disables periodic saves)."""
    max_checkpoints: int = 3
    resume_from_dir: str | None = None
    auto_resume: bool = False

    def state_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def load_state_dict(self, state: dict[str, Any]) -> None:
        raise NotImplementedError

    def save_checkpoint(self, step: int) -> None:
        if self.is_main_process:
            path = Path(self.checkpoint_root) / f"step_{step:07d}"
            path.mkdir(parents=True, exist_ok=True)
            torch.save(self.state_dict(), path / "state.pt")
            self._rotate()
            logger.info("Saved checkpoint %s", path)
        if self.world_size > 1:
            dist.barrier()

    def save_maybe(self, step: int, *, force: bool = False) -> None:
        if self.checkpoint_interval > 0 and (
            force or step % self.checkpoint_interval == 0
        ):
            self.save_checkpoint(step)

    def _rotate(self) -> None:
        root = Path(self.checkpoint_root)
        steps = sorted(
            (int(m.group(1)), p)
            for p in root.glob("step_*")
            if (m := _STEP_DIR.match(p.name))
        )
        for _, path in steps[: -self.max_checkpoints] if self.max_checkpoints else []:
            for child in path.glob("*"):
                child.unlink()
            path.rmdir()

    def find_latest_checkpoint(self) -> str | None:
        root = Path(self.checkpoint_root)
        if not root.is_dir():
            return None
        candidates = [
            (int(m.group(1)), p)
            for p in root.glob("step_*")
            if (m := _STEP_DIR.match(p.name)) and (p / "state.pt").exists()
        ]
        return str(max(candidates)[1]) if candidates else None

    def maybe_auto_resume(self, explicit_dir: str | None) -> str | None:
        path = explicit_dir or (
            self.find_latest_checkpoint() if self.auto_resume else None
        )
        if path is None:
            return None
        state = torch.load(
            Path(path) / "state.pt", map_location=self.device, weights_only=False
        )
        self.load_state_dict(state)
        logger.info("Resumed from %s", path)
        return path
