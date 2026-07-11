"""Torch-free launch parent → torchrun → child builds the OpdTrainer.

The parent parses only the (torch-free) config to size ``torchrun`` and set env,
then ``execvp``s torchrun. Each torchrun worker re-enters this module with
``--child``, imports torch + the trainer, and runs it.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Sequence

from ..config import LaunchConfig
from ..utils.config import (
    add_config_patch_arguments,
    format_config_patch_args,
    load_config_file,
)


def _num_processes(devices: int | list[int] | str) -> int:
    if isinstance(devices, int):
        return max(1, devices)
    if isinstance(devices, list):
        return max(1, len(devices))
    try:
        listing = subprocess.check_output(["nvidia-smi", "-L"], text=True)
        return max(1, len([line for line in listing.splitlines() if line.strip()]))
    except Exception:
        return 1


def _default_log_dir() -> str:
    job = os.getenv("SLURM_JOB_ID")
    return f"logs/slurm-{job}" if job else f"logs/local-{int(time.time())}"


def run(config_path: str, updates: Sequence[str], removes: Sequence[str]) -> None:
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
        "opdlens.scripts.launch",
        config_path,
        *format_config_patch_args(updates, removes),
        "--child",
    ]
    os.execvpe("torchrun", cmd, env)


def _neutralize_tilelang_stub() -> None:
    """Block ``import tilelang`` before any model (hence ``fla``) load.

    ``tilelang`` (a vLLM dependency, imported by ``fla`` during the Qwen3.5 load)
    ships a ``libcudart_stub.so`` lacking ``cudaDeviceReset``. Once loaded, it
    shadows torch's real libcudart: vLLM/flashinfer's ``CudaRTLibrary`` resolve
    ``libcudart`` via ``find_loaded_library`` (first ``/proc/self/maps`` match) and
    crash binding ``cudaDeviceReset`` during the multi-GPU engine init. ``fla`` only
    uses the tilelang backend on Hopper (we run Ampere A800, where it uses Triton),
    so making the import fail is safe — ``fla``'s backend probe catches ``ImportError``
    and falls back to Triton — and keeps the stub out of the process entirely. Must
    run before the trainer (and its lazy transformers/fla imports) is imported."""
    import sys

    sys.modules.setdefault("tilelang", None)  # type: ignore[assignment]


def _run_child(
    config_path: str, updates: Sequence[str], removes: Sequence[str]
) -> None:
    _neutralize_tilelang_stub()
    from ..training import OpdTrainer

    config = load_config_file(config_path, updates, removes)
    trainer = OpdTrainer(**config)
    trainer.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="opdlens launch (parent/child).")
    parser.add_argument("config")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    add_config_patch_arguments(parser)
    args = parser.parse_args()
    if args.child:
        _run_child(args.config, args.config_updates, args.config_removes)
    else:
        run(args.config, args.config_updates, args.config_removes)


if __name__ == "__main__":
    main()
