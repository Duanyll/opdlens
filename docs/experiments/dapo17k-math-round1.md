# DAPO17K to MATH round 1

Status: GSM8K barrier 4175 completed successfully. The first ten DAPO/MATH jobs
4198-4207 all established the same 0.6472 step-0 MATH baseline, then exposed a
shared base-loss backward-memory failure before step 1. A memory-bounded replacement
profile is being validated and requeued under distinct `-chunk256` run names.

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
- The replacement profile checkpoints the mathematically identical base-loss
  softmax graph in 256-token chunks, lowers only vLLM's operational GPU-memory
  reservation from 0.25 to 0.18, and enables expandable CUDA allocator segments.
  `base_loss_chunk_size` defaults to zero, preserving the old monolithic path for
  every existing config. Dataset, sampling, loss, optimizer, batch, and 2,048-token
  completion cap are unchanged.
- Full-MATH eval is greedy pass@1 with max 2,048 tokens at fixed steps
  0/100/200/300. The lower cadence than GSM8K controls the cost of evaluating all
  5,000 rows; there is no best-step selection.
- LoRA preserves every evaluated trained state (100/200/300) and disables rotation.
  Full fine-tuning retains the 100-step checkpoint cadence.
- Jobs launch through `scripts/run_dapo_math.sh`, an immutable Git snapshot runner
  with two GPUs and a 24-hour limit (within the a800 partition's three-day cap).
  Python safe-path mode prevents the shared checkout from shadowing snapshot code.
  The runner intentionally does not use `headless-tui-run`.

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

The 2026-07-12 02:16 HKT config audit passed all eight spine/protocol tests. A
subsequent exact execution-path probe established that the original `opdlens`
console parent imported the archived package correctly, but the torchrun workers it
spawned imported `opdlens` from the shared checkout. Jobs 4176-4185 were still
dependency-blocked, had no start time, and had created no checkpoint roots, so they
were cancelled without consuming GPU or producing experimental data.

Commit `08687af` hardened the parent with `python -P -m opdlens.scripts.cli`, but a
worker-level probe caught that this flag does not propagate through `exec` to
torchrun. Its dependency-blocked replacements 4186/4188/4189/4191-4197 were also
cancelled before start. Commit `5cd4f5c` additionally exports
`PYTHONSAFEPATH=1`; an actual one-worker torchrun probe then resolves both
`opdlens.__file__` and the CLI inside the archived commit. Regression tests pin the
full runner environment. The complete suite has 27 passing tests; Ruff and Pyright
pass.

At 02:36 HKT, every initial ledger commit contains `5cd4f5c`; its config blob is
byte-identical to the checked-in config and its commit/run pair matches the Slurm
job comment. Slurm's stored batch scripts for the first and last matrix jobs also
contain both safe-path controls, proving the submitted scripts captured the fix.
All ten jobs remained dependency-blocked until barrier 4175 completed at 03:22 HKT.

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
| `logits-full` | `examples/dapo17k_math_round1_logits_full.jsonc` | `4a37183` | 4198 | failed before step 1: base-loss backward OOM after step-0 eval |
| `logits-lora` | `examples/dapo17k_math_round1_logits_lora.jsonc` | `9745def` | 4199 | failed before step 1: same backward OOM |
| `logitlens-full` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | `239a2f2` | 4200 | failed before step 1: same backward OOM |
| `logitlens-lora` | `examples/dapo17k_math_round1_logitlens_lora.jsonc` | `6cc907d` | 4201 | failed before step 1: same backward OOM |
| `jlens-full` | `examples/dapo17k_math_round1_jlens_full.jsonc` | `f4d09fc` | 4202 | failed before step 1: same backward OOM |
| `jlens-lora` | `examples/dapo17k_math_round1_jlens_lora.jsonc` | `32b2396` | 4203 | failed before step 1: same backward OOM |
| `hiddenmse-full` | `examples/dapo17k_math_round1_hiddenmse_full.jsonc` | `3ec201c` | 4204 | failed before step 1: same backward OOM |
| `hiddenmse-lora` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | `051ec10` | 4205 | failed before step 1: same backward OOM |
| `symjlens-full` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | `45d5113` | 4206 | failed before step 1: same backward OOM |
| `symjlens-lora` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | `2a45abf` | 4207 | failed before step 1: same backward OOM |

## Runtime incidents

Jobs 4198-4207 all loaded the intended immutable snapshots, completed the full
5,000-example step-0 MATH evaluation at 0.6472, generated their first on-policy
batch, and then failed in `opd_base_loss` backward. Each rank had about 76-77 GiB
in use and attempted one additional 4.04-GiB allocation. The trainer processes one
sequence at a time, so lowering global batch would not address the peak. The cause
is the monolithic fp32 `[T,V]` JSD graph exposed by longer DAPO completions, not an
arm-specific auxiliary path; even both arm-A jobs failed identically.

The replacement keeps the 2,048-token rollout protocol and exact JSD definition.
It checkpoints the base divergence in 256-token slices so backward recomputes and
releases one slice at a time. The new knob defaults to the original zero/monolithic
behavior, and all ten replacement configs select the same value. A distinct
`-chunk256` suffix prevents Trackio from appending different configs to the failed
runs. A saved-tensor audit measures 0.251x of the original autograd state on the
same JSD while matching both value and gradient; the full 29-test suite, Ruff, and
Pyright pass. No failed job reached step 1 or wrote a checkpoint.
