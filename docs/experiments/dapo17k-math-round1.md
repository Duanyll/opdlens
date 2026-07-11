# DAPO17K to MATH round 1

Status: configs prepared and validated; launch is gated on completion of the GSM8K
round-one matrix. No DAPO/MATH training job has been submitted yet.

## Dataset identity

- Train: `open-r1/DAPO-Math-17k-Processed`, config `all`, revision
  `31dd309567e3da778038cc87d868b6097a3ccf68`, 17,398 deduplicated prompts.
  The upstream `BytedTsinghua-SIA/DAPO-Math-17k` parquet contains 1,791,700 rows
  because each unique prompt is repeated 100 times; loading it directly would waste
  memory while producing the same sampling distribution.
- Eval: canonical Hendrycks MATH test from `EleutherAI/hendrycks_math`, revision
  `21a5633873b6a120296cce3e2df9d5550074f4a3`. All seven subject configs are
  concatenated for the full 5,000-example test. This is deliberately not MATH-500.
- Both sides use the same zero-shot math system prompt and `math_verify` symbolic
  grader. Gold extraction succeeds for all 5,000 MATH test solutions, including the
  small number that use `\fbox` rather than `\boxed`.
- Both pinned datasets have been loaded once and verified under
  `HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1`, matching the batch-job environment.

This pairing is not a held-out generalization test: 970 of the 5,000 MATH test
questions (19.4%) occur verbatim in DAPO17K, or 972 after whitespace/case
normalization. The requested full DAPO17K and full MATH test are retained without
filtering, so cross-arm contrasts remain controlled, but absolute post-training
MATH accuracy must be labelled partly in-distribution rather than a clean OOD score.

## Fixed protocol

- Teacher/student: `Qwen/Qwen3.5-9B` -> `Qwen/Qwen3.5-2B`.
- Full and LoRA use the validated repro-GKD optimizer definitions: full LR 2e-5,
  LoRA LR 5e-5, AdamW betas 0.9/0.95, weight decay 0.1, 30-step warmup and cosine.
- Base loss is JSD beta 0.5 at temperature 0.9. Training is 300 steps, global batch
  96, seed 42, and two A800 GPUs per job.
- DAPO rollout is temperature 0.9, top-p 1, max 2,048 tokens. Measured prompt-token
  percentiles are p50=114, p95=226, p99=371, max=1,523, so the 4,096-token engine
  limit admits every prompt plus the configured completion budget.
- Full-MATH eval is greedy pass@1 with max 2,048 tokens at fixed steps
  0/100/200/300. The lower cadence than GSM8K controls the cost of evaluating all
  5,000 rows; there is no best-step selection.
- LoRA preserves every evaluated trained state (100/200/300) and disables rotation.
  Full fine-tuning retains the 100-step checkpoint cadence.
- Jobs launch through `scripts/run_dapo_math.sh`, an immutable Git snapshot runner
  with two GPUs and a 24-hour limit (within the a800 partition's three-day cap).
  It intentionally does not use `headless-tui-run`.

The transferred initial auxiliary settings are B/C/E weight 0.01 and D weight 1.94,
with one shared 512-token subset across layers and fp32 D bridge/MSE. These values are
not assumed to be balanced on DAPO: the first update is a mandatory scale gate. A job
is cancelled and replaced under a new run name if its weighted auxiliary/base ratio
is outside the range established by the stable GSM8K runs or if any loss/gradient is
non-finite.

## Artifact policy

The teacher lens and bridge are model/layer artifacts and are kept identical across
datasets, as in the earlier MATH pivot. Refitting them only for DAPO would change the
arm definition at the same time as the dataset. E instead waits for the matched
student lens from GSM8K job 4155; its teacher and student lenses are fit on the same
264 gold-CoT chats, fixing the cross-side calibration mismatch found in the first E
attempt.

| Artifact | Used by | Status |
|---|---|---|
| `/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt` | C/E teacher lens | ready |
| `/gdata/users/duanyll/jlens/qwen3p5_bridge/bridge.pt` | D per-layer bridge | ready |
| `/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot.pt` | E student lens | job 4155 pending |

## Structural guardrail

Within each finetune mode, the five configs differ only in `arm`, `experiment_name`,
and `checkpoint_root`. `tests/test_spine.py` validates all configs through
`OpdTrainer` and compares the remaining spine byte-for-byte. The dataset revisions,
full-MATH protocol, two-GPU allocation, and LoRA checkpoint policy are also pinned.

## Launch gates

1. All GSM8K current and triggered-legacy jobs finish or have a documented terminal
   replacement; the DAPO jobs must not displace that matrix.
2. Student-lens job 4155 succeeds and the artifact passes layer/shape loading.
3. Arm A establishes step-0 MATH runtime and score without truncation/OOM; the batch
   wall time is adjusted before the remaining matrix is submitted if needed.
4. B/C/D/E pass the first-update auxiliary-scale and finite-gradient checks.
5. Every launch receives its own pre-launch commit, immutable snapshot, Slurm comment,
   and ledger row. GPU util/power is checked after startup.

## Run ledger

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logits-full` | `examples/dapo17k_math_round1_logits_full.jsonc` | pending | pending | GSM8K gate |
| `logits-lora` | `examples/dapo17k_math_round1_logits_lora.jsonc` | pending | pending | GSM8K gate |
| `logitlens-full` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | pending | pending | GSM8K gate |
| `logitlens-lora` | `examples/dapo17k_math_round1_logitlens_lora.jsonc` | pending | pending | GSM8K gate |
| `jlens-full` | `examples/dapo17k_math_round1_jlens_full.jsonc` | pending | pending | GSM8K gate |
| `jlens-lora` | `examples/dapo17k_math_round1_jlens_lora.jsonc` | pending | pending | GSM8K gate |
| `hiddenmse-full` | `examples/dapo17k_math_round1_hiddenmse_full.jsonc` | pending | pending | GSM8K gate |
| `hiddenmse-lora` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | pending | pending | GSM8K gate |
| `symjlens-full` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | pending | pending | GSM8K + artifact gate |
| `symjlens-lora` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | pending | pending | GSM8K + artifact gate |
