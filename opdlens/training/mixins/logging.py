"""LoggingMixin — trackio + append-only jsonl + rank-mean metric reduction.

Trimmed from flow_control's LoggingMixin (image logging dropped). Load-bearing
collective contract preserved: ``resolve_run_context`` and ``log_metrics`` are
collective and must be called on every rank (never behind an ``is_main_process``
guard); rank 0 alone writes to trackio / the jsonl.
"""

from __future__ import annotations

import contextlib
import json
import secrets
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from pydantic import PrivateAttr
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    ProgressColumn,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from ...utils.logging import console, get_logger
from ..base import BaseTrainer

logger = get_logger(__name__)


class LoggingMixin(BaseTrainer):
    experiment_name: str = "opdlens"
    run_id: str | None = None
    runs_root: str = "./runs"
    trackio_project: str | None = None

    _run_dir: Path | None = PrivateAttr(default=None)
    _metrics_file: Any = PrivateAttr(default=None)
    _tracker_active: bool = PrivateAttr(default=False)

    # ----------------------------- Run context ------------------------------- #

    def resolve_run_context(self) -> None:
        """Choose (rank 0) and broadcast a stable ``run_id``. Collective."""
        if self.run_id is None and self.is_main_process:
            self.run_id = f"{time.strftime('%Y%m%d%H%M%S')}-{secrets.token_hex(4)}"
        if self.world_size > 1:
            holder = [self.run_id]
            dist.broadcast_object_list(holder, src=0)
            self.run_id = holder[0]
        logger.info("Run id: %s", self.run_id)

    def init_tracker(self) -> None:
        if not self.is_main_process:
            return
        import trackio

        self._run_dir = Path(self.runs_root) / self.experiment_name / str(self.run_id)
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = (self._run_dir / "metrics.jsonl").open(
            "a", encoding="utf-8"
        )
        trackio.init(
            project=self.trackio_project or self.experiment_name,
            name=self.run_id,
            resume="allow",
        )
        self._tracker_active = True
        logger.info("Tracker initialized at %s", self._run_dir)

    def finish_tracker(self) -> None:
        if not self.is_main_process:
            return
        if self._metrics_file is not None:
            self._metrics_file.close()
            self._metrics_file = None
        if self._tracker_active:
            import trackio

            trackio.finish()
            self._tracker_active = False

    # ------------------------------- Metrics --------------------------------- #

    def log_metrics(
        self, metrics: Mapping[str, float | torch.Tensor], step: int
    ) -> None:
        """Rank-mean reduce (ranks may omit keys) then log on rank 0. Collective."""
        local = {k: float(v) for k, v in metrics.items()}
        if self.world_size > 1:
            gathered: list[Any] = [None] * self.world_size
            dist.all_gather_object(gathered, local)
        else:
            gathered = [local]
        if not self.is_main_process:
            return

        total: dict[str, float] = {}
        count: dict[str, int] = {}
        for payload in gathered:
            for key, value in payload.items():
                total[key] = total.get(key, 0.0) + value
                count[key] = count.get(key, 0) + 1
        reduced = {key: total[key] / count[key] for key in total}
        self._emit(reduced, step)

    def log_main(self, metrics: Mapping[str, float], step: int) -> None:
        """Log already-reduced metrics on rank 0 only (NOT collective). Use for
        eval, which reduces across ranks itself before logging."""
        if self.is_main_process:
            self._emit({k: float(v) for k, v in metrics.items()}, step)

    def _emit(self, reduced: Mapping[str, float], step: int) -> None:
        if self._tracker_active:
            import trackio

            trackio.log(dict(reduced), step=step)
        if self._metrics_file is not None:
            self._metrics_file.write(json.dumps({"step": step, **reduced}) + "\n")
            self._metrics_file.flush()

    # ------------------------------- Progress -------------------------------- #

    @classmethod
    def get_progress_columns(cls) -> tuple[ProgressColumn, ...]:
        return (
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )

    @contextlib.contextmanager
    def status_bar(self, title: str) -> Iterator[None]:
        if self.is_main_process:
            console.rule(f"[bold]{title}")
        try:
            yield
        finally:
            if self.is_main_process:
                console.rule(f"[bold]{title} — done")
