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


def arm_config(arm: str, aux_weight: float) -> tuple[str, dict[str, Any]]:
    shared = {"aux_weight": aux_weight, "teacher_layers": TEACHER_LAYERS}
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
                    f"{OPDLENS_ROOT}/artifacts/qwen3p5-2b-jlens.pt"
                ),
                "teacher_jacobian_path": (f"{ARTIFACT_ROOT}/qwen3p5_9b_v2/lens.pt"),
            },
        ),
    }
    return configs[arm]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices="bcde", required=True)
    parser.add_argument("--finetune", choices=("full", "lora"), required=True)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    baseline_path = Path(f"examples/repro_gkd_2b_{args.finetune}.jsonc")
    baseline = load_config_file(str(baseline_path))
    config = deepcopy(baseline)
    tag, config["arm"] = arm_config(args.arm, args.aux_weight)
    run_name = f"{tag}-{args.finetune}"
    if args.aux_weight != 0.1:
        run_name += f"-aux{args.aux_weight:g}"
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
