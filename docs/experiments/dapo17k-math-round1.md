# DAPO17K to MATH round 1

Status: all ten experiments are submitted behind GSM8K barrier job 4175. They are
present in the queue but cannot consume GPUs until every remaining GSM8K current
and triggered-legacy job exits successfully.

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
- Bare reference expressions must be placed in an unambiguous boxed envelope before
  `math_verify.parse`: the previous direct parse self-graded only 3,957/5,000 golds.
  The corrected grader self-grades 5,000/5,000, including intervals, complex values,
  multi-answer sets, and `\dfrac` forms.
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
arm definition at the same time as the dataset. E uses the matched student lens
completed by GSM8K job 4166; its teacher and student lenses are fit on the same
264 gold-CoT chats, fixing the cross-side calibration mismatch found in the first E
attempt.

| Artifact | Used by | Status |
|---|---|---|
| `/gdata/users/duanyll/jlens/qwen3p5_9b_v2/lens.pt` | C/E teacher lens | ready |
| `/gdata/users/duanyll/jlens/qwen3p5_bridge/bridge.pt` | D per-layer bridge | ready |
| `/gdata/users/duanyll/opdlens/artifacts/qwen3p5-2b-jlens-cot.pt` | E student lens | ready; 264 prompts, finite 2048x2048 layers 6/12/18 |

## Structural guardrail

Within each finetune mode, the five configs differ only in `arm`, `experiment_name`,
and `checkpoint_root`. `tests/test_spine.py` validates all configs through
`OpdTrainer` and compares the remaining spine byte-for-byte. The dataset revisions,
full-MATH protocol, two-GPU allocation, and LoRA checkpoint policy are also pinned.

The 2026-07-12 02:16 HKT pre-release audit passed all eight spine/protocol tests.
For every ledger row, the config blob at the recorded launch commit is byte-identical
to the checked-in config, contains the corrected MATH reference parser in its
ancestry, and matches the commit/run pair stored in the Slurm job comment. None of
the ten checkpoint roots exists before release, so no job can silently resume an
older run under the same experiment name.

## Launch gates

1. All GSM8K current and triggered-legacy jobs finish successfully. Slurm barrier
   4175 encodes `afterok` dependencies on jobs
   4154/4160/4161/4167/4168/4169/4170/4171/4172; every DAPO job depends only on
   that barrier, so none can displace the GSM8K matrix.
2. The matched student lens is complete and has passed layer/shape/finite loading.
3. Arm A establishes step-0 MATH runtime and score without truncation/OOM. If the
   24-hour estimate is inadequate, all still-initializing matrix jobs are cancelled
   and replaced before accepting any trained checkpoint.
4. B/C/D/E pass the first-update auxiliary-scale and finite-gradient checks.
5. Every launch receives its own pre-launch commit, immutable snapshot, Slurm comment,
   and ledger row. GPU util/power is checked after startup.

## Run ledger

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logits-full` | `examples/dapo17k_math_round1_logits_full.jsonc` | `c73a326` | 4176 | dependency on barrier 4175 |
| `logits-lora` | `examples/dapo17k_math_round1_logits_lora.jsonc` | `4cd0046` | 4177 | dependency on barrier 4175 |
| `logitlens-full` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | `1b79fd4` | 4178 | dependency on barrier 4175 |
| `logitlens-lora` | `examples/dapo17k_math_round1_logitlens_lora.jsonc` | `6d9bafc` | 4179 | dependency on barrier 4175 |
| `jlens-full` | `examples/dapo17k_math_round1_jlens_full.jsonc` | `8e3048e` | 4180 | dependency on barrier 4175 |
| `jlens-lora` | `examples/dapo17k_math_round1_jlens_lora.jsonc` | `769c3f7` | 4181 | dependency on barrier 4175 |
| `hiddenmse-full` | `examples/dapo17k_math_round1_hiddenmse_full.jsonc` | `262c17c` | 4182 | dependency on barrier 4175 |
| `hiddenmse-lora` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | `b143418` | 4183 | dependency on barrier 4175 |
| `symjlens-full` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | `a9c5ed8` | 4184 | dependency on barrier 4175 |
| `symjlens-lora` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | `074f0c8` | 4185 | dependency on barrier 4175 |
