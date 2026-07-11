"""Generate one GSM8K BCDE round-one config from the frozen logits baseline."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from opdlens.utils.config import load_config_file

ARTIFACT_ROOT = "/gdata/users/duanyll/jlens"
OPDLENS_ROOT = "/gdata/users/duanyll/opdlens"
TEACHER_LAYERS = [8, 16, 24]


def arm_config(
    arm: str,
    aux_weight: float,
    aux_token_policy: str,
    mse_dtype: str,
) -> tuple[str, dict[str, Any]]:
    shared = {
        "aux_weight": aux_weight,
        "aux_token_policy": aux_token_policy,
        "teacher_layers": TEACHER_LAYERS,
    }
    configs: dict[str, tuple[str, dict[str, Any]]] = {
        "b": (
            "logitlens",
            {"type": "logit_lens", **shared, "temperature": 1.0},
        ),
        "c": (
            "jlens",
            {
                "type": "jspace",
                **shared,
                "temperature": 1.0,
                "jacobian_path": f"{ARTIFACT_ROOT}/qwen3p5_9b_v2/lens.pt",
            },
        ),
        "d": (
            "hiddenmse",
            {
                "type": "hidden_mse",
                **shared,
                "mse_dtype": mse_dtype,
                "bridge_path": f"{ARTIFACT_ROOT}/qwen3p5_bridge/bridge.pt",
            },
        ),
        "e": (
            "symjlens",
            {
                "type": "symmetric_jlens",
                **shared,
                "temperature": 1.0,
                "student_jacobian_path": (
                    f"{OPDLENS_ROOT}/artifacts/qwen3p5-2b-jlens-cot.pt"
                ),
                "teacher_jacobian_path": (f"{ARTIFACT_ROOT}/qwen3p5_9b_v2/lens.pt"),
            },
        ),
    }
    return configs[arm]


def apply_legacy_profile(config: dict[str, Any], finetune: str) -> None:
    """Apply the historical jlens numerical recipe on the current eval spine."""
    config.update(
        {
            "base_beta": 0.0,
            "base_temperature": 1.0,
            "rollout_temperature": 1.0,
            "rollout_top_p": 1.0,
            "rollout_max_tokens": 512,
            "global_batch_size": 8,
            "scheduler_config": {"class_name": "ConstantLR", "factor": 1.0},
        }
    )
    config["optimizer_config"] = {
        "class_name": "AdamW",
        "lr": 2e-6 if finetune == "full" else 5e-5,
        "betas": [0.9, 0.999],
        "weight_decay": 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices="bcde", required=True)
    parser.add_argument("--finetune", choices=("full", "lora"), required=True)
    parser.add_argument("--profile", choices=("current", "legacy"), default="current")
    parser.add_argument("--aux-weight", type=float)
    parser.add_argument("--aux-token-policy", choices=("compat", "shared"))
    parser.add_argument("--mse-dtype", choices=("input", "fp32"), default="input")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = Path(f"examples/repro_gkd_2b_{args.finetune}.jsonc")
    baseline = load_config_file(str(baseline_path))
    config = deepcopy(baseline)
    if args.profile == "legacy" and args.arm == "d":
        parser.error("the legacy fallback matrix contains only arms B/C/E")
    aux_weight = args.aux_weight
    if aux_weight is None:
        aux_weight = 0.01 if args.profile == "legacy" else 0.1
    aux_token_policy = args.aux_token_policy
    if aux_token_policy is None:
        aux_token_policy = "shared" if args.profile == "legacy" else "compat"
    tag, config["arm"] = arm_config(
        args.arm, aux_weight, aux_token_policy, args.mse_dtype
    )
    if args.profile == "legacy":
        apply_legacy_profile(config, args.finetune)
        run_name = f"{tag}-{args.finetune}-legacy"
    else:
        run_name = f"{tag}-{args.finetune}"
        if aux_weight != 0.1:
            run_name += f"-aux{aux_weight:g}"
    config["experiment_name"] = run_name
    config["launch"]["devices"] = 2
    config["checkpoint_root"] = f"{OPDLENS_ROOT}/ckpt/{run_name}"
    if args.finetune == "lora":
        # LoRA checkpoints are small enough to preserve every evaluated state.
        config["checkpoint_interval"] = config["eval_steps"]
        config["max_checkpoints"] = 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rendered = {"$schema": "../schema/opd.schema.json", **config}
    args.out.write_text(json.dumps(rendered, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
