"""Generate round-3 (DAPO->MATH) search configs.

Clone the matching round-2 DAPO base config (so optimizer / base_beta / rollout /
data all stay identical and comparable), then (a) swap the eval to MATH-500 for a
fast training-loop signal (the final MATH-5000 + AIME eval runs separately on the
winners), and (b) override only the arm knobs that differ. Wave A transfers each
arm's GSM8K champion and uses the EXISTING (GSM8K-calibrated) lenses, so it needs no
fit artifacts and can queue immediately. Wave B re-runs the C/E winners with the new
MATH-calibrated lenses (jobs 4414/4415) once those land.
"""

import json
import pathlib

ROOT = pathlib.Path("/home/duanyll/opdlens")
R2 = ROOT / "experiments" / "dapo17k_math_round2"
OUT = ROOT / "experiments" / "dapo_math_round3"

# Fast training-loop eval: MATH-500 (same decode/grader as the round-2 MATH block,
# just the 500-problem subset). The heavy MATH-5000 + AIME finals run on winners.
MATH500_EVAL = [
    {
        "type": "math500",
        "hf_id": "HuggingFaceH4/MATH-500",
        "split": "test",
        "system_prompt": (
            "Solve the following problem step by step. "
            "Put your final answer within \\boxed{}."
        ),
        "eval_temperature": 0.0,
        "eval_top_p": 1.0,
        "avg_k": 1,
        "eval_max_tokens": 2048,
    }
]


def make(base_name: str, name: str, arm_override: dict) -> None:
    cfg = json.loads((R2 / base_name).read_text())
    cfg["eval_benchmarks"] = MATH500_EVAL
    cfg["arm"].update(arm_override)
    cfg["experiment_name"] = name
    cfg["trackio_project"] = "opdlens-math"
    cfg["checkpoint_root"] = f"/gdata/users/duanyll/opdlens/ckpt/dapo-math-round3/{name}"
    path = OUT / (name.replace("-", "_") + ".jsonc")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  arm={cfg['arm']['type']} eval=math500")


# --- Wave A: GSM8K champions transferred to DAPO, existing lenses, MATH-500 eval ---
# A logits: the MATH-500 anchor (round-2 A was on MATH-5000; B/C/E must be compared
# against A on the same metric).
make("logits_full_round2.jsonc", "a-logits-full-r3", {})
# B logit_lens: GSM8K peak was a single deep layer 24 (>> the [16,24] pair).
make("logitlens_full_round2.jsonc", "b-l24-full-r3", {"teacher_layers": [24]})
# C jspace: GSM8K champion [12,16] + reverse-KL (the clearest lens-arm lever).
make(
    "jlens_full_round2.jsonc",
    "c-l12l16-rev-full-r3",
    {"teacher_layers": [12, 16], "kl": "reverse"},
)
# E symmetric_jlens: GSM8K champion [16,24] + reverse-KL + temperature 2.
make(
    "symjlens_full_round2.jsonc",
    "e-l16l24-rev-t2-full-r3",
    {"teacher_layers": [16, 24], "kl": "reverse", "temperature": 2.0},
)
