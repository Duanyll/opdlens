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
- Auxiliary protocol: teacher layers 8/16/24, mapped student layers 6/12/18.
  After archaeology and scale calibration, vocab-KL arms B/C/E use weight 0.01
  and raw hidden-MSE arm D uses 1.94.
  Apart from the arm block and run/checkpoint identity, all training and evaluation
  fields come directly from the matching logits baseline.
- LoRA saves every evaluated trained state (steps 50/100/150/200/250/300) and
  disables checkpoint rotation. Full fine-tuning retains the baseline's 100-step
  checkpoint cadence.

The reference curves are `logits-full`: 0.7491 / 0.8165 / 0.8332 / 0.8256 and
`logits-lora`: 0.7491 / 0.7945 / 0.8158 / 0.8196 at steps 0/100/200/300.

## Setting profiles

The profile is part of a job's identity. A similarly named arm under a different
profile is an additional experiment, never an in-place replacement.

| Profile | Jobs | Shared train/eval spine | Optimizer and batch | Arm-specific setting |
|---|---|---|---|---|
| `current-vocab-compat` | 4144-4147 (B/C) | JSD beta 0.5, base T=0.9; rollout T=0.9, top-p 1, max 768, thinking off; full GSM8K fixed-step eval | full LR 2e-5 / LoRA LR 5e-5; AdamW wd 0.1; 30-step warmup + cosine; global batch 96 | B/C weight 0.01; 512 cap sampled independently per layer, matching pre-knob opdlens behavior |
| `current-hidden-shared` | 4153/4154 (D) | same current spine | same current optimizer/batch | D weight 1.94; one shared 512-token subset across layers; bridge and MSE in fp32 |
| `current-sym-shared` | 4158/4159 (E) | same current spine | same current optimizer/batch | E weight 0.01; one shared 512-token subset; completion-matched teacher/student Jacobian lenses |
| `legacy-bce-full` | conditional, not submitted | current prompt/grader and fixed-step reporting retained; forward-KL base T=1; rollout T=1, top-p 1, max 512 | historical full LR 2e-6, AdamW wd 0, constant LR, global batch 8 | B/C/E weight 0.01 and shared 512-token subset; this imports old numerical settings without reviving the old 200-question proxy eval |
| `legacy-bce-lora-hybrid` | conditional, not submitted | same as `legacy-bce-full` | no historical LoRA setting exists, so LoRA LR 5e-5 and adapter definition remain from the validated current baseline; global batch 8 | otherwise identical to `legacy-bce-full`; explicitly a hybrid, not claimed as an exact old reproduction |

`aux_token_policy` defaults to `compat`, and D's `mse_dtype` defaults to `input`.
Thus configs written before these knobs retain the behavior they had before commit
`b28d6f1`. Corrected profiles opt in with `aux_token_policy: "shared"`; D also
sets `mse_dtype: "fp32"`. Generated configs follow the same backward-compatible
defaults unless the flags are passed explicitly.

The legacy B/C/E matrix is appended, not substituted, if either (a) a run drops
more than 5 percentage points below its own step-0 score at a scheduled eval, or
(b) its step-100 and step-200 scores are both more than two pooled standard errors
below the matching logits baseline at the same fixed steps. A final step-300 gap
over two pooled standard errors also triggers it. This avoids reacting to one noisy
checkpoint and never performs best-step selection.

## Offline artifacts

| Artifact | Purpose | Status |
|---|---|---|
| `/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt` | teacher Jacobian, 264 prompts | ready |
| `/gdata/users/duanyll/jlens/qwen3p5_bridge/bridge.pt` | per-layer 9B->2B bridge | ready |
| `/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens.pt` | student Jacobian, raw questions | superseded; calibration mismatch |
| `/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot.pt` | student Jacobian, 264 question+gold-CoT chats | queued two-GPU sharded refit, job 4155 |

## Run ledger

Each Slurm job receives an immutable Git archive of the recorded commit. This
prevents a queued job from silently picking up later edits in the shared checkout.
The same commit is also stored in the Slurm job comment.

| Run | Profile | Config | Commit | Slurm job | State |
|---|---|---|---|---|---|
| `logitlens-full-aux0.01` | `current-vocab-compat` | `examples/gsm8k_round1_logitlens_full_aux0p01.jsonc` | `035832d` | 4144 | running |
| `logitlens-lora-aux0.01` | `current-vocab-compat` | `examples/gsm8k_round1_logitlens_lora_aux0p01.jsonc` | `72930be` | 4146 | running |
| `jlens-full-aux0.01` | `current-vocab-compat` | `examples/gsm8k_round1_jlens_full_aux0p01.jsonc` | `02702c6` | 4145 | running |
| `jlens-lora-aux0.01` | `current-vocab-compat` | `examples/gsm8k_round1_jlens_lora_aux0p01.jsonc` | `9755e55` | 4147 | running |
| `hiddenmse-full-aux1.94` | `current-hidden-shared` | `examples/gsm8k_round1_hiddenmse_full_aux1p94.jsonc` | `f3d3ff9` | 4153 | running; replaces underweighted 4140 |
| `hiddenmse-lora-aux1.94` | `current-hidden-shared` | `examples/gsm8k_round1_hiddenmse_lora_aux1p94.jsonc` | `d6fceee` | 4154 | running; replaces underweighted 4141 |
| `symjlens-full-aux0.01` | `current-sym-shared` | `examples/gsm8k_round1_symjlens_full_aux0p01.jsonc` | `564f218` | 4158 | dependency on student-lens job 4155 |
| `symjlens-lora-aux0.01` | `current-sym-shared` | `examples/gsm8k_round1_symjlens_lora_aux0p01.jsonc` | `ada5396` | 4159 | dependency on student-lens job 4155 |

## Monitoring

A run is stable only after model/vLLM initialization, the first successful train
step, finite base/aux/total losses and gradient norm, healthy GPU utilization, and
at least one scheduled eval. Checks are frequent until that point. Stable jobs are
checked every 30 minutes, relaxed to hourly only after sustained healthy behavior.

## Launch incidents

Jobs 4128-4133 exited before Python startup because Slurm resolved `env` to a
non-executable user-local path. No model or dataset state was touched. The launcher
now invokes `/usr/bin/env` explicitly; replacement jobs are recorded in the ledger.

The initial B/C configs used the example-arm weight 0.1. At step 50, B-full,
B-LoRA, C-full, and C-LoRA scored 0.5838, 0.5580, 0.5792, and 0.4541 respectively,
all down from 0.7475, while `hiddenmse-full` reached 0.8287.
The raw vocab-KL auxiliary was about 3.1 for B and 5.9 for C versus a base loss
near 0.037, so weight 0.1 made it dominate optimization. B/C/E replacements use
the previously exercised jlens CoT weight 0.01 and a distinct `-aux0.01` run name;
the failed pilots remain in Trackio as diagnostic evidence. At weight 0.01, the
step-50 B-full/B-LoRA/C-full/C-LoRA scores are 0.7892/0.7870/0.7437/0.7619, so
none shows the earlier collapse. B-full and C-full reach 0.7870 and 0.7619 at
step 100.

The old raw hidden-MSE D arm did not use 0.1: its measured natural weight was
1.94. Current step-1 raw D MSE is 0.02452, matching the old 0.0249, so 0.1 made
the auxiliary only 0.066x of the current base and made D nearly a logits run.
Jobs 4140/4141 were cancelled after steps 181/100 despite healthy
curves, because the comparison was underweighted. Jobs 4153/4154 use 1.94, whose
initial absolute contribution (0.0476) lies between B (0.0310) and C (0.0589).

The archaeological audit also found two token-semantics regressions. B/C/E had
sampled a different capped token subset for every layer, while D ignored the
512-token cap; D also evaluated its bridge in bf16 instead of the old fp32.
Commit `f2623bc` restores one shared token subset per sequence across layers,
applies the cap to D, and computes the D bridge/MSE in fp32. The full test suite,
Ruff, and Pyright pass. B/C jobs are retained: all four are stable, and the
sampling difference changes only which identically distributed tokens are used
when an individual completion exceeds the cap; it does not explain the failed
0.1 pilots. Commit `b28d6f1` then exposes those corrected semantics as opt-in
knobs with backward-compatible defaults, while the current D/E configs select
them explicitly.

The first student lens fit used raw GSM8K questions, whereas the existing v2
teacher lens was calibrated on chat-formatted questions plus gold CoT completions.
Jobs 4148/4149 had not started and were cancelled. The replacement student lens
uses the same completion-aware distribution, split across two GPUs and merged;
E will only launch after that artifact succeeds.

The first two-GPU replacement fit, job 4150, exposed a Slurm-step resource bug:
shard 0 occupied the non-GPU resources of the allocation and shard 1 waited while
GPU 1 sat idle. It and dependent jobs 4151/4152 were cancelled. Commit `e49dc8b`
adds exact per-step CPU/memory requests; replacement fit 4155 gates E jobs
4158/4159. Jobs 4156/4157 were dependency-pending and used no GPU; they were
replaced so the E launch commits contain the new explicit compatibility knobs.

## Archaeological setting audit

The collapse is explained by setting scale, not by a hidden-state capture,
readout, Jacobian-direction, layer-map, completion-mask, or KL-direction bug.
The old B/C runs used forward KL base loss near 0.16, constant full-finetune LR
2e-6, batch 8, and auxiliary weight 0.01. The current validated logits spine uses
JSD beta 0.5 near 0.037, peak full LR 2e-5 (LoRA 5e-5), global batch 96, and the
same raw auxiliary magnitudes. Thus weight 0.1 changed the initial weighted
aux/base ratios from roughly 0.20x/0.37x in the old B/C runs to 8.30x/15.77x.
The corrected 0.01 ratios are 0.83x/1.58x and have passed the first scheduled
evaluations.
