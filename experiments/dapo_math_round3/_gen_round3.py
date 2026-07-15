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
    cfg["checkpoint_root"] = (
        f"/gdata/users/duanyll/opdlens/ckpt/dapo-math-round3/{name}"
    )
    path = OUT / (name.replace("-", "_") + ".jsonc")
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print(f"wrote {path.relative_to(ROOT)}  arm={cfg['arm']['type']} eval=math500")


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]
MATH_SYSTEM = (
    "Solve the following problem step by step. Put your final answer within \\boxed{}."
)
# AIME/AIMO decode + metric per OPRD (arXiv:2606.06021): Avg@16, temp 0.7, top_p 0.95,
# zero-shot with no system prompt and a trailing boxed instruction. eval_max_tokens is
# 4096 (not OPRD's 31744 — that is for long-CoT R1-distill models; our student is a
# non-thinking Qwen3.5-2B trained at 2048).
OPRD = {
    "system_prompt": None,
    "answer_instruction": (
        "Please reason step by step, and put your final answer within \\boxed{}."
    ),
    "eval_temperature": 0.7,
    "eval_top_p": 0.95,
    "avg_k": 16,
    "eval_max_tokens": 4096,
}
# Final-checkpoint eval: MATH-5000 stays greedy avg@1 (comparable to the round-2
# MATH numbers, se ~ 0.007) while the three competition sets use the OPRD protocol.
FINALS_EVAL = [
    {
        "type": "math",
        "hf_id": "EleutherAI/hendrycks_math",
        "hf_revision": "21a5633873b6a120296cce3e2df9d5550074f4a3",
        "subjects": MATH_SUBJECTS,
        "split": "test",
        "system_prompt": MATH_SYSTEM,
        "eval_temperature": 0.0,
        "eval_top_p": 1.0,
        "avg_k": 1,
        "eval_max_tokens": 2048,
    },
    {"type": "aime", "year": 2024, **OPRD},
    {"type": "aime", "year": 2025, **OPRD},
    {"type": "aimo", **OPRD},
]


def make_finals() -> None:
    """Eval-only config for the finals: run with `opdlens eval <cfg> --checkpoint
    <ckpt>` (or with no checkpoint to score the bare/teacher model). Student is the
    2B base; --checkpoint supplies the trained weights. A wider vLLM context holds
    the 4096-token AIME generations."""
    cfg = json.loads((R2 / "logits_full_round2.jsonc").read_text())
    cfg["arm"] = {"type": "logits"}
    cfg["eval_benchmarks"] = FINALS_EVAL
    cfg["eval_max_samples"] = None
    cfg["vllm_max_model_len"] = 8192
    cfg["vllm_gpu_memory_utilization"] = 0.6
    cfg["experiment_name"] = "finals-eval"
    cfg["trackio_project"] = "opdlens-math-finals"
    cfg["checkpoint_root"] = "/gdata/users/duanyll/opdlens/ckpt/dapo-math-round3/_eval"
    path = OUT / "finals_eval.jsonc"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    ev = [b["type"] + (f"-{b['year']}" if "year" in b else "") for b in FINALS_EVAL]
    print(f"wrote {path.relative_to(ROOT)}  eval={ev}")


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

# --- Wave B: the C/E champions re-run with the round-3 MATH-calibrated lenses -----
# (jobs 4414/4415) + their forward-KL siblings. Against Wave A's GSM8K-lens champions
# this factors the calibration gain apart from the reverse-KL gain. Submit with
# `--dependency=afterok:<teacher-fit-job>` so they wait for the 9B lens artifact.
TEACHER_MATH = "/gdata/users/duanyll/opdlens/artifacts/qwen3p5-9b-jlens-math.pt"
STUDENT_MATH = "/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-math.pt"
_KL = {"reverse": "rev", "forward": "fwd"}
for _kl in ("reverse", "forward"):
    make(
        "jlens_full_round2.jsonc",
        f"c-l12l16-{_KL[_kl]}-mathlens-r3",
        {"teacher_layers": [12, 16], "kl": _kl, "jacobian_path": TEACHER_MATH},
    )
for _kl, _temp in (("reverse", 2.0), ("forward", 1.0)):
    make(
        "symjlens_full_round2.jsonc",
        f"e-l16l24-{_KL[_kl]}-mathlens-r3",
        {
            "teacher_layers": [16, 24],
            "kl": _kl,
            "temperature": _temp,
            "student_jacobian_path": STUDENT_MATH,
            "teacher_jacobian_path": TEACHER_MATH,
        },
    )

# --- Final-checkpoint eval config (MATH-5000 + AIME24/25 + AIMO, OPRD protocol) ---
make_finals()
