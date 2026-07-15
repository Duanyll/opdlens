"""``opdlens eval`` — score a checkpoint or a bare model, no training.

Mirrors ``launch.py``'s parent → torchrun → child pattern (so multi-GPU eval shards
the benchmark across ranks exactly like training eval): the parent sizes torchrun
from ``launch.devices`` and re-execs; each child builds the ``OpdTrainer`` and runs
``evaluate_only`` (load frozen student / optional checkpoint → generator → the shared
``evaluate`` spine). No teacher / optimizer / rollout data is ever built.

Eval the teacher's ceiling by pointing ``student`` at the teacher model in the config
(``rollout_backend: "hf"`` avoids needing a vLLM-complete snapshot for the big model).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence

from ..config import LaunchConfig
from ..utils.config import (
    add_config_patch_arguments,
    format_config_patch_args,
    load_config_file,
)
from .launch import _default_log_dir, _neutralize_tilelang_stub, _num_processes


def run(
    config_path: str,
    checkpoint: str | None,
    updates: Sequence[str],
    removes: Sequence[str],
) -> None:
    """Parent: size torchrun from ``launch.devices``, set env, exec torchrun."""
    config = load_config_file(config_path, updates, removes)
    launch = LaunchConfig(**config.get("launch", {}))
    n_proc = _num_processes(launch.devices)

    env = os.environ.copy()
    env.setdefault("LOG_DIR", _default_log_dir())
    if isinstance(launch.devices, list):
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(d) for d in launch.devices)
    env.update(launch.env)

    cmd = [
        "torchrun",
        "--standalone",
        f"--nproc_per_node={n_proc}",
        "-m",
        "opdlens.scripts.eval",
        config_path,
        *format_config_patch_args(updates, removes),
    ]
    if checkpoint is not None:
        cmd += ["--checkpoint", checkpoint]
    cmd += ["--child"]
    os.execvpe("torchrun", cmd, env)


def _run_child(
    config_path: str,
    checkpoint: str | None,
    updates: Sequence[str],
    removes: Sequence[str],
) -> None:
    _neutralize_tilelang_stub()
    from ..training import OpdTrainer

    config = load_config_file(config_path, updates, removes)
    trainer = OpdTrainer(**config)
    trainer.evaluate_only(checkpoint=checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="opdlens eval (parent/child).")
    parser.add_argument("config")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    add_config_patch_arguments(parser)
    args = parser.parse_args()
    if args.child:
        _run_child(
            args.config, args.checkpoint, args.config_updates, args.config_removes
        )
    else:
        run(args.config, args.checkpoint, args.config_updates, args.config_removes)


if __name__ == "__main__":
    main()
