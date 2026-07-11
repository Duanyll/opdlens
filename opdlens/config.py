"""Entrypoint-layer config (torch-free so the launch parent can parse it fast).

Kept deliberately import-light (pydantic only, no torch): the launch parent reads
``launch`` to size torchrun and exec, without paying torch's import cost.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict


class LaunchConfig(BaseModel):
    """How to launch: process count + extra env. The trainer type is fixed to
    ``opd`` (opdlens has a single trainer; no polymorphic registry)."""

    model_config = ConfigDict(extra="forbid")

    type: str = "opd"
    devices: int | list[int] | Literal["all"] = "all"
    env: dict[str, str] = {}
