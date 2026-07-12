# DAPO17K to MATH round 1

Status: GSM8K barrier 4175 completed successfully. Eight current-profile DAPO/MATH
runs are complete and two are running. The completed B/C results met the
predeclared poor-BCE fallback criterion, so legacy BCE jobs 4240-4245 were appended:
all six are now complete. Current jobs 4228-4231 occupy eight A800s on node 1;
every other matrix entry is complete and there is no unsubmitted experiment left
in this stage.

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
  reservation from 0.25 to 0.20, and enables expandable CUDA allocator segments.
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

## Triggered legacy BCE profile

The fallback criterion is the same fixed-step rule used for GSM8K: append, never
substitute, the legacy B/C/E matrix if a current run falls more than five points
below its own initialization or finishes more than two pooled standard errors below
the matching logits baseline. At step 300, B-LoRA is 0.6282 versus A-LoRA 0.6658
(gap 0.0376; trigger threshold 0.0191), C-full is 0.6462 versus A-full 0.6858
(gap 0.0396; threshold 0.0189), and C-LoRA is 0.6304 versus A-LoRA 0.6658
(gap 0.0354; threshold 0.0191). The fallback therefore fires without selecting a
best checkpoint or waiting for E.

The legacy full profile imports the historical forward-KL base loss (`beta=0`),
base/rollout temperature 1, 512-token training rollout, global batch 8, AdamW LR
2e-6 with betas 0.9/0.999 and zero weight decay, and constant LR. No historical
LoRA recipe exists, so the legacy-LoRA profile is explicitly a hybrid that retains
the validated 5e-5 adapter LR while importing the other historical settings. Both
profiles retain the pinned DAPO/MATH datasets, full 5,000-example greedy eval at
steps 0/100/200/300, two-GPU launch, base-loss chunking, local compile caches, and
the 0.20 vLLM reservation. B/C/E differ only in their arm block within each
finetune mode. `--profile` defaults to `current`, and regenerating without the new
flag is byte-identical to the existing current config.

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

Within each finetune mode, the five current configs and the three legacy BCE configs
each differ only in `arm`, `experiment_name`, and `checkpoint_root` inside their
respective profile. `tests/test_spine.py` validates all configs through `OpdTrainer`
and compares the remaining spine byte-for-byte. The dataset revisions, full-MATH
protocol, two-GPU allocation, legacy recipe, and LoRA checkpoint policy are pinned.

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

### Chunk256 / vLLM 0.18 startup attempt

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logits-full-chunk256` | `examples/dapo17k_math_round1_logits_full.jsonc` | `00449a4` | 4208 | failed during vLLM init; replaces 4198 |
| `logits-lora-chunk256` | `examples/dapo17k_math_round1_logits_lora.jsonc` | `9ab65c4` | 4209 | failed during vLLM init; replaces 4199 |
| `logitlens-full-chunk256` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | `c95322d` | 4210 | failed during vLLM init; replaces 4200 |
| `logitlens-lora-chunk256` | `examples/dapo17k_math_round1_logitlens_lora.jsonc` | `9bfd254` | 4211 | failed during vLLM init; replaces 4201 |
| `jlens-full-chunk256` | `examples/dapo17k_math_round1_jlens_full.jsonc` | `eb9d1bf` | 4212 | failed during vLLM init; replaces 4202 |
| `jlens-lora-chunk256` | `examples/dapo17k_math_round1_jlens_lora.jsonc` | `85c1e79` | 4213 | failed during vLLM init; replaces 4203 |
| `hiddenmse-full-chunk256` | `examples/dapo17k_math_round1_hiddenmse_full.jsonc` | `3fa5683` | 4214 | failed during vLLM init; replaces 4204 |
| `hiddenmse-lora-chunk256` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | `2a5c062` | 4215 | failed during vLLM init; replaces 4205 |
| `symjlens-full-chunk256` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | `36dd42b` | 4216 | failed during vLLM init; replaces 4206 |
| `symjlens-lora-chunk256` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | `8391803` | 4217 | failed during vLLM init; replaces 4207 |

At 03:52 HKT, jobs 4208-4213 occupied all 12 available A800s across nodes 1/2;
jobs 4214-4217 followed immediately as slots freed. Slurm comments match every
run/commit pair, and the stored batch scripts for jobs 4208 and 4217 contain the
inherited safe-path controls. Each job therefore executed its recorded immutable
snapshot.

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

The first chunked attempt set vLLM's memory fraction to 0.18. It produced a
343,319-token KV cache and 922 Qwen3.5 Mamba cache blocks, but vLLM requires one
block for each of its default 1,024 maximum concurrent sequences and therefore
failed fast during initialization. Jobs 4208-4217 never reached evaluation or the
chunked loss. The replacement raises the fraction to 0.20, which remains 4 GiB
below the original reservation while clearing this deterministic cache threshold.

One 0.20 job, B-full 4220, then failed alone while five peers initialized. Both
ranks reported `OSError: Stale file handle` while reading the shared NFS
TorchInductor cache, followed by `FXGraphCacheMiss`; this is a compile-cache race,
not a model/config failure. The DAPO runner now gives each Slurm job a node-local
`/tmp` root for both vLLM and TorchInductor caches. Already initialized jobs are
retained; the failed B-full and jobs that had not yet started are replaced under
the cache-isolated runner.

### Chunk256 / vLLM 0.20 replacements

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logits-full-chunk256vllm020` | `examples/dapo17k_math_round1_logits_full.jsonc` | `43f62df` | 4218 | completed; step-300 acc 0.6858; replaces 4208 |
| `logits-lora-chunk256vllm020` | `examples/dapo17k_math_round1_logits_lora.jsonc` | `408ab83` | 4219 | completed; step-300 acc 0.6658; replaces 4209 |
| `logitlens-full-chunk256vllm020` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | `8ae5a7f` | 4220 | failed in shared NFS compile cache; replaces 4210 |
| `logitlens-lora-chunk256vllm020` | `examples/dapo17k_math_round1_logitlens_lora.jsonc` | `fad5d85` | 4221 | completed; step-300 acc 0.6282; replaces 4211 |
| `jlens-full-chunk256vllm020` | `examples/dapo17k_math_round1_jlens_full.jsonc` | `91a0d43` | 4222 | completed; step-300 acc 0.6462; replaces 4212 |
| `jlens-lora-chunk256vllm020` | `examples/dapo17k_math_round1_jlens_lora.jsonc` | `e94f206` | 4223 | completed; step-300 acc 0.6304; replaces 4213 |
| `hiddenmse-full-chunk256vllm020` | `examples/dapo17k_math_round1_hiddenmse_full.jsonc` | `99ac47c` | 4224 | completed; step-300 acc 0.6488; replaces 4214 |
| `hiddenmse-lora-chunk256vllm020` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | `8ece700` | 4225 | cancelled before start for cache-isolated replacement |
| `symjlens-full-chunk256vllm020` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | `cc143ee` | 4226 | cancelled before start for cache-isolated replacement |
| `symjlens-lora-chunk256vllm020` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | `200715c` | 4227 | cancelled before start for cache-isolated replacement |

At 04:04 HKT, jobs 4218-4223 occupied all 12 A800s and jobs 4224-4227 were
ready in the queue. Every Slurm comment records the matching full commit and run
name; each job has its own pre-launch commit and immutable snapshot.

### Cache-isolated replacements

| Run | Config | Commit | Slurm job | State |
|---|---|---|---|---|
| `logitlens-full-chunk256vllm020` | `examples/dapo17k_math_round1_logitlens_full.jsonc` | `b0dc53f` | 4228 | completed; step-300 acc 0.6610; replaces failed 4220 |
| `hiddenmse-lora-chunk256vllm020` | `examples/dapo17k_math_round1_hiddenmse_lora.jsonc` | `0e93d89` | 4229 | completed; step-300 acc 0.6304; replaces unstarted 4225 |
| `symjlens-full-chunk256vllm020` | `examples/dapo17k_math_round1_symjlens_full.jsonc` | `1e2551a` | 4230 | stable at step 269; step-200 acc 0.6614; replaces unstarted 4226 |
| `symjlens-lora-chunk256vllm020` | `examples/dapo17k_math_round1_symjlens_lora.jsonc` | `58ecd62` | 4231 | stable at step 241; step-200 acc 0.6460; replaces unstarted 4227 |

At 04:12 HKT, jobs 4218/4219/4221-4224 occupied all 12 A800s; jobs 4228-4231
were ready in the queue. The three cancelled jobs had no start time and consumed
no GPU. All replacements have matching pre-launch commits, comments, and snapshots.

### Triggered legacy BCE append

| Run | Profile | Config | Commit | Slurm job | State |
|---|---|---|---|---|---|
| `logitlens-full-legacy-chunk256vllm020` | `legacy-bce-full` | `examples/dapo17k_math_round1_logitlens_full_legacy.jsonc` | `7d44085` | 4240 | completed; step-300 acc 0.6410 |
| `logitlens-lora-legacy-chunk256vllm020` | `legacy-bce-lora-hybrid` | `examples/dapo17k_math_round1_logitlens_lora_legacy.jsonc` | `ce36408` | 4241 | completed; step-300 acc 0.6620 |
| `jlens-full-legacy-chunk256vllm020` | `legacy-bce-full` | `examples/dapo17k_math_round1_jlens_full_legacy.jsonc` | `bc45773` | 4242 | completed; step-300 acc 0.6462 |
| `jlens-lora-legacy-chunk256vllm020` | `legacy-bce-lora-hybrid` | `examples/dapo17k_math_round1_jlens_lora_legacy.jsonc` | `25b3920` | 4243 | completed; step-300 acc 0.6390 |
| `symjlens-full-legacy-chunk256vllm020` | `legacy-bce-full` | `examples/dapo17k_math_round1_symjlens_full_legacy.jsonc` | `a522055` | 4244 | completed; step-300 acc 0.6418 |
| `symjlens-lora-legacy-chunk256vllm020` | `legacy-bce-lora-hybrid` | `examples/dapo17k_math_round1_symjlens_lora_legacy.jsonc` | `29298cf` | 4245 | completed; step-300 acc 0.6278 |

The generator/config implementation is commit `7cf7b0e`. Every row then receives
its own empty pre-launch commit so that its immutable snapshot and Slurm comment
have a unique experiment identity even though all six share the same validated
code and profile. The comments record the full commit, run, train/eval datasets,
and `profile=legacy`.

At 04:18 HKT, all six running jobs had completed full-MATH step-0 evaluation at
the identical 0.6454 score and advanced to train step 10-11. The chunked backward
therefore passes the exact point where jobs 4198-4207 OOMed. All losses and gradient
norms are finite with no Trackio alerts. At step 1, A-full and A-LoRA share base
0.04036 exactly; weighted auxiliary/base ratios are 0.83x for B-LoRA, 1.46x for
C-full/LoRA, and 1.60x for D-full, all inside the stable GSM8K range. The 12
allocated GPUs average 85-100% utilization, 334-392 W, and 51-60 GiB memory over
five minutes; no allocated card is idle and the chunked path leaves substantial
headroom.

At 04:48 HKT, the same six jobs had advanced to step 46-55 without a new runtime
error or Trackio alert. Their latest base losses are 0.0352-0.0415 and all total
losses and gradient norms remain finite. Weighted auxiliary/base ratios have
settled to 0.54x for B-LoRA, 0.80x/1.08x for C-full/LoRA, and 0.62x for D-full,
showing neither domination nor collapse. The 12 allocated GPUs average 88-100%
utilization, 326-375 W, and 56-65 GiB memory over five minutes. Jobs 4228-4231
remain queued with their recorded commits, so every released slot will immediately
admit another matrix experiment.

At 05:18 HKT, the jobs had advanced to step 81-97. Latest base losses remain
0.0363-0.0422 with finite total losses and gradient norms; weighted auxiliary/base
ratios continue to settle at 0.46x for B-LoRA, 0.71x/0.88x for C-full/LoRA, and
0.59x for D-full. No alert or new error is present. The 12 GPUs average 92-100%
utilization and 284-379 W over five minutes. The first step-100 evaluations have
not yet completed, and jobs 4228-4231 remain ready to fill the next released slots.

At 05:48 HKT, all six step-100 evaluations and checkpoints were complete. From
the common 0.6454 initialization, A-full/A-LoRA reach 0.6660/0.6638, B-LoRA
0.6426, C-full/C-LoRA 0.6400/0.6290, and D-full 0.6242. D is currently worst as
anticipated; the largest decline is 2.12 percentage points, below the 5-point
collapse threshold. Training resumed to step 112-133 with all metrics finite;
latest weighted auxiliary/base ratios are 0.47x for B, 0.66x/0.87x for C, and
0.58x for D. The 12 GPUs average 95-100% utilization and 318-393 W over five
minutes, while the four recorded replacement jobs remain queued.

At 06:18 HKT, all six jobs continued through step 148-176 with finite metrics and
no new alert or runtime error. The latest weighted auxiliary/base ratios are 0.47x
for B-LoRA, 0.63x/0.81x for C-full/LoRA, and 0.59x for D-full; the lower step-100
C/D scores are not accompanied by loss divergence. The 12 allocated GPUs average
87-100% utilization and 338-374 W over five minutes. Jobs 4228-4231 remain queued
for the next released slots.

At 06:48 HKT, A-full, A-LoRA, and D-full had completed step-200 checkpoints and
evaluations at 0.6782, 0.6572, and 0.6462 respectively. D recovered from its
step-100 low of 0.6242 to the initialization level rather than continuing to
collapse. B-LoRA and C-full/LoRA were still at step 183-195 and had not yet
evaluated step 200. All six latest losses and gradients remain finite with no new
alert or error. The 12 GPUs average 76-100% utilization and 263-381 W over five
minutes; the temporarily lower pair had just completed checkpoint/eval and remained
well above idle power. Jobs 4228-4231 remain queued.

At 07:18 HKT, all step-200 results were available: A-full/A-LoRA
0.6782/0.6572, B-LoRA 0.6394, C-full/C-LoRA 0.6446/0.6322, and D-full
0.6462. D is no longer the worst arm at this checkpoint; C-LoRA is lowest but
only 1.32 percentage points below initialization. Training continued to step
214-257 with every loss and gradient finite; weighted auxiliary/base ratios are
0.42x for B, 0.67x/0.72x for C, and 0.56x for D. The 12 GPUs average 89-100%
utilization and 289-381 W over five minutes, and jobs 4228-4231 remain queued.

At 08:18 HKT, A-full, A-LoRA, B-LoRA, and D-full had completed cleanly. Their
step-300 accuracies are 0.6858, 0.6658, 0.6282, and 0.6488 respectively. B-LoRA's
1.72-point decline from initialization remains below the collapse threshold and
its final weighted auxiliary/base ratio is 0.44x; all final losses and gradients
are finite. C-full was saving/evaluating step 300 and C-LoRA was stable at step
287. Their latest weighted auxiliary/base ratios are 0.67x and 0.78x.

The released slots admitted cache-isolated B-full 4228, D-LoRA 4229, E-full
4230, and then E-LoRA 4231; all ten matrix entries are therefore either complete
or running. The first three replacements reached steps 19/16/4 with finite
metrics and weighted auxiliary/base ratios 0.62x/1.56x/1.37x. Their vLLM and
TorchInductor paths are under `/tmp/opdlens-cache-$SLURM_JOB_ID`, with no stale
file handle or OOM. Step-0 scores differ by at most 0.72 points across these
separately scheduled greedy runs (0.6478/0.6526 versus the first wave's 0.6454),
which is within roughly one full-MATH standard error and is not used for
checkpoint selection. No Trackio alert is present. Over five minutes, the 12
allocated GPUs average 75-100% utilization, 266-386 W, and 53-66 GiB memory; the
lowest-utilization pair belongs to a newly started job and remains far above idle
power.

At 08:48 HKT, C-full and C-LoRA had also completed cleanly at 0.6462 and 0.6304,
leaving six of ten current-profile runs complete. All three LoRA completions retain
their step-100/200/300 checkpoints. The four remaining current jobs reached steps
57/54/38/20 for B-full/D-LoRA/E-full/E-LoRA with finite losses and gradients;
their weighted auxiliary/base ratios are 0.44x/0.75x/0.67x/1.18x. No new runtime
error or Trackio alert is present. Node 1's eight allocated GPUs average 82-99%
utilization, 287-398 W, and 53-67 GiB over five minutes.

The fixed-step B/C gaps then triggered the legacy append described above. At
08:55 HKT, B-full 4240 and B-LoRA 4241 started on node 2 while C/E jobs 4242-4245
remained queued. The four current jobs plus two legacy jobs occupy all 12 available
A800s, and every future two-GPU release already has a recorded experiment waiting.

At 09:18 HKT, legacy B-full/B-LoRA had completed full-MATH step-0 at
0.6526/0.6508 and step-100 at 0.6356/0.6504, then advanced to steps 166/155.
Their step-1 weighted auxiliary/base ratios are 0.160x/0.167x, matching the
historical scale rather than the current profile's much larger contribution. All
losses and gradients remain finite, and the LoRA step-100 checkpoint is retained.
Current B-full/D-LoRA/E-full/E-LoRA reached steps 97/94/74/55 with weighted
auxiliary/base ratios 0.40x/0.64x/0.54x/0.74x and no alert or new runtime error.
Across the 12 allocated GPUs, five-minute utilization is 73-100% and power is
264-372 W; the four unallocated node-2 cards remain near 58-65 W. Legacy C/E jobs
4242-4245 remain queued behind the two running B jobs.

At 09:48 HKT, legacy B-full/B-LoRA completed cleanly at step-300 accuracies
0.6410/0.6620. The full run stays within 1.16 points of initialization; LoRA
finishes 1.12 points above initialization and 3.38 points above current B-LoRA.
Their final weighted auxiliary/base ratios are 0.163x/0.126x, all losses and
gradients are finite, and LoRA retains step-100/200/300 checkpoints. Legacy C-full
4242 and C-LoRA 4243 immediately replaced them and reached steps 57/42 after common
full-MATH initialization at 0.6526/0.6508. Their latest weighted auxiliary/base
ratios are 0.32x/0.22x, again matching the historical scale.

Current B-full/D-LoRA/E-full completed step-100 eval at 0.6568/0.6224/0.6464 and
continued to steps 131/128/105; E-LoRA remained stable at step 90. D-LoRA is the
lowest current result, 3.02 points below its own initialization but still inside
the five-point collapse gate, with finite training metrics. The eight node-1 GPUs
average 84-100% utilization and 280-412 W. The four node-2 GPUs running the small
historical batch average 39-75% and 152-313 W while completing an entire run in
about 40 minutes; none is idle near 100 W. Legacy E jobs 4244/4245 remain queued.

At 10:18 HKT, legacy C-full/C-LoRA completed cleanly at 0.6462/0.6390. Full is
identical to current C-full, while legacy LoRA improves 0.86 points over current
C-LoRA. Both remain within 1.18 points of their own initialization; final losses
and gradients are finite, and LoRA retains all three eval checkpoints. Legacy
E-full 4244 and E-LoRA 4245 immediately started at 10:16/10:18, so all submitted
matrix entries are now complete or running and there is no remaining experiment to
queue in this stage.

Current B-full/D-LoRA/E-full/E-LoRA advanced to steps 170/168/141/119 with finite
metrics and weighted auxiliary/base ratios 0.38x/0.62x/0.47x/0.65x. No alert or
new runtime error is present. Node 1 remains saturated at 88-100% utilization and
305-376 W over five minutes. The node-2 window spans legacy C completion and E
startup, averaging 40-77% and 205-334 W on the four allocated devices; the other
four cards remain at idle power.

At 10:48 HKT, current B-full/D-LoRA completed step-200 eval at 0.6536/0.6314.
D-LoRA recovered 0.90 points from its step-100 low and remains 2.12 points below
initialization, well inside the collapse gate. They continued to steps 203/201;
E-full/E-LoRA reached steps 177/154, with E-LoRA step-100 at 0.6308. Every latest
loss and gradient is finite and there is no alert or new runtime error.

Legacy E-full/E-LoRA advanced to steps 207/200. Their fixed curves are
0.6496/0.6408/0.6370 and 0.6478/0.6388/0.6290 at steps 0/100/200 respectively,
declines of 1.26/1.88 points rather than collapse. The 12 allocated GPUs average
51-100% utilization and 242-376 W over five minutes; every allocated device remains
well above idle power.

At 11:32 HKT, legacy E-full/E-LoRA were confirmed complete at 0.6418/0.6278.
Their final declines from initialization are 0.78/2.00 points, with finite losses
and gradients; LoRA retains step-100/200/300 checkpoints. This completes all six
triggered legacy BCE runs.

Current E-full/E-LoRA completed step-200 eval at 0.6614/0.6460, recovering
1.50 points in each case from step 100. B-full/D-LoRA/E-full/E-LoRA advanced to
steps 262/260/226/200 with every latest metric finite and no alert or new runtime
error. The eight allocated node-1 GPUs average 86-100% utilization and 334-403 W;
node 2 is idle after the legacy matrix completed.

At 12:08 HKT, current B-full and D-LoRA completed cleanly at 0.6610 and 0.6304.
B-full improves 1.32 points over its separately scheduled initialization. D-LoRA
finishes 2.22 points below initialization but recovers from its 0.6224 step-100
low; its final losses and gradients are finite and all three LoRA checkpoints are
retained. E-full/E-LoRA remain as the final two jobs at steps 269/241. Their four
active GPUs average 85-99% utilization and 326-374 W over five minutes; the other
four node-1 devices had just exited B/D final evaluation.
