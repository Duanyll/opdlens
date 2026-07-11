# BCDE round 1: GSM8K

This matrix extends the validated `logits-full` / `logits-lora` reproduction with
arms B-E. The repository calls the student `Qwen3.5-2B`; this is the checked-in
9B-to-small-model baseline referred to as 9B->3B in the experiment request. We do
not change the model family mid-matrix.

## Fixed protocol

- Teacher/student: `Qwen/Qwen3.5-9B` -> `Qwen/Qwen3.5-2B`.
- Dataset and grader: GSM8K train, full 1,319-example EvalScope-compatible test.
- Base loss: generalized JSD with beta 0.5 and temperature 0.9.
- Schedule: 300 steps, global batch 96, eval every 50 steps, seed 42.
- Optimizer: the validated baseline values, including 2e-5 for full and 5e-5 for
  LoRA. Each run uses two A800 GPUs with the global batch held fixed.
- Auxiliary protocol: weight 0.1, teacher layers 8/16/24, mapped student layers
  6/12/18. Apart from the arm block and run/checkpoint identity, all training and
  evaluation fields come directly from the matching logits baseline.
- LoRA saves every evaluated trained state (steps 50/100/150/200/250/300) and
  disables checkpoint rotation. Full fine-tuning retains the baseline's 100-step
  checkpoint cadence.

The reference curves are `logits-full`: 0.7491 / 0.8165 / 0.8332 / 0.8256 and
`logits-lora`: 0.7491 / 0.7945 / 0.8158 / 0.8196 at steps 0/100/200/300.

## Offline artifacts

| Artifact | Purpose | Status |
|---|---|---|
| `/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt` | teacher Jacobian, 264 prompts | ready |
| `/gdata/users/duanyll/jlens/qwen3p5_bridge/bridge.pt` | per-layer 9B->2B bridge | ready |
| `/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens.pt` | student Jacobian, 264 prompts | pending fit |

## Run ledger

Each Slurm job receives an immutable Git archive of the recorded commit. This
prevents a queued job from silently picking up later edits in the shared checkout.
The same commit is also stored in the Slurm job comment.

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logitlens-full` | `examples/gsm8k_round1_logitlens_full.jsonc` | pending | pending | pending |
| `logitlens-lora` | `examples/gsm8k_round1_logitlens_lora.jsonc` | pending | pending | pending |
| `jlens-full` | `examples/gsm8k_round1_jlens_full.jsonc` | pending | pending | pending |
| `jlens-lora` | `examples/gsm8k_round1_jlens_lora.jsonc` | pending | pending | pending |
| `hiddenmse-full` | `examples/gsm8k_round1_hiddenmse_full.jsonc` | pending | pending | pending |
| `hiddenmse-lora` | `examples/gsm8k_round1_hiddenmse_lora.jsonc` | pending | pending | pending |
| `symjlens-full` | `examples/gsm8k_round1_symjlens_full.jsonc` | pending | pending | artifact dependency |
| `symjlens-lora` | `examples/gsm8k_round1_symjlens_lora.jsonc` | pending | pending | artifact dependency |

## Monitoring

A run is stable only after model/vLLM initialization, the first successful train
step, finite base/aux/total losses and gradient norm, healthy GPU utilization, and
at least one scheduled eval. Checks are frequent until that point. Stable jobs are
checked every 30 minutes, relaxed to hourly only after sustained healthy behavior.
