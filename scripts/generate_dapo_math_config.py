"""Generate one DAPO17K-train / canonical-MATH-test matrix config."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from opdlens.benchmarks.base import MATH_SYSTEM
from opdlens.utils.config import load_config_file

CHECKPOINT_ROOT = "/gdata/users/duanyll/opdlens/ckpt/dapo17k-math"
BRIDGE_PATH = "/gdata/users/duanyll/jlens/qwen3p5_bridge/bridge.pt"
STUDENT_JACOBIAN_PATH = "/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot.pt"
TEACHER_JACOBIAN_PATH = "/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt"
DAPO_REVISION = "31dd309567e3da778038cc87d868b6097a3ccf68"
MATH_REVISION = "21a5633873b6a120296cce3e2df9d5550074f4a3"
MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
TEACHER_LAYERS = [8, 16, 24]
MEMORY_PROFILE = "chunk256vllm020"


def arm_config(arm: str) -> tuple[str, dict[str, Any]]:
    shared_vocab = {
        "aux_weight": 0.01,
        "aux_token_policy": "shared",
        "teacher_layers": TEACHER_LAYERS,
    }
    configs: dict[str, tuple[str, dict[str, Any]]] = {
        "a": ("logits", {"type": "logits"}),
        "b": (
            "logitlens",
            {"type": "logit_lens", **shared_vocab, "temperature": 1.0},
        ),
        "c": (
            "jlens",
            {
                "type": "jspace",
                **shared_vocab,
                "temperature": 1.0,
                "jacobian_path": TEACHER_JACOBIAN_PATH,
            },
        ),
        "d": (
            "hiddenmse",
            {
                "type": "hidden_mse",
                "aux_weight": 1.94,
                "aux_token_policy": "shared",
                "teacher_layers": TEACHER_LAYERS,
                "mse_dtype": "fp32",
                "bridge_path": BRIDGE_PATH,
            },
        ),
        "e": (
            "symjlens",
            {
                "type": "symmetric_jlens",
                **shared_vocab,
                "temperature": 1.0,
                "student_jacobian_path": STUDENT_JACOBIAN_PATH,
                "teacher_jacobian_path": TEACHER_JACOBIAN_PATH,
            },
        ),
    }
    return configs[arm]


def apply_dataset_spine(config: dict[str, Any]) -> None:
    config["trackio_project"] = "opdlens-math"
    config["train_benchmark"] = {
        "type": "dapo_math",
        "hf_id": "open-r1/DAPO-Math-17k-Processed",
        "hf_name": "all",
        "hf_revision": DAPO_REVISION,
        "split": "train",
        "system_prompt": MATH_SYSTEM,
    }
    config["eval_benchmarks"] = [
        {
            "type": "math",
            "hf_id": "EleutherAI/hendrycks_math",
            "hf_revision": MATH_REVISION,
            "subjects": MATH_SUBJECTS,
            "split": "test",
            "system_prompt": MATH_SYSTEM,
            "eval_temperature": 0.0,
            "eval_top_p": 1.0,
            "avg_k": 1,
            "eval_max_tokens": 2048,
        }
    ]
    config["eval_max_samples"] = None
    config["rollout_max_tokens"] = 2048
    config["base_loss_chunk_size"] = 256
    config["vllm_gpu_memory_utilization"] = 0.20
    config["eval_steps"] = 100
    config["launch"]["devices"] = 2
    config["launch"]["env"]["HF_DATASETS_OFFLINE"] = "1"
    config["launch"]["env"]["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices="abcde", required=True)
    parser.add_argument("--finetune", choices=("full", "lora"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_config_file(f"examples/repro_gkd_2b_{args.finetune}.jsonc")
    config = deepcopy(baseline)
    apply_dataset_spine(config)
    tag, config["arm"] = arm_config(args.arm)
    run_name = f"{tag}-{args.finetune}-{MEMORY_PROFILE}"
    config["experiment_name"] = run_name
    config["checkpoint_root"] = f"{CHECKPOINT_ROOT}/{run_name}"
    if args.finetune == "lora":
        config["checkpoint_interval"] = config["eval_steps"]
        config["max_checkpoints"] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = {"$schema": "../schema/opd.schema.json", **config}
    args.out.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
