# Results summary (all rounds)

Cross-round dashboard of the main eval numbers. Per-round detail (dataset identity,
job ids, commits, caveats) lives in the sibling `docs/experiments/<round>.md`.

Arms: **A** `logits` (plain OPD) · **B** `logit_lens` · **C** `jspace` (Jacobian-lens)
· **D** `hidden_mse` (bridge) · **E** `symmetric_jlens`.

Numbers are **final step-300 test accuracy** unless marked `‡` (still training — the
value is the latest logged step, provisional).

- GSM8K test = 1319 examples, se ≈ 0.011. MATH test = 2500 examples, se ≈ 0.009.
  **Treat gaps under ~0.015 as ties.**
- Source: the `eval[...] step N: acc=` lines in `logs/<jobname>_<jobid>.log`. Refresh:
  ```bash
  for f in logs/*.log; do
    base=$(basename "$f" .log); name=${base%_*}
    last=$(grep -hoE 'eval\[[a-z0-9_]+\] step [0-9]+: acc=[0-9.]+' "$f" | tail -1)
    [ -n "$last" ] && printf '%-32s %s\n' "$name" "$last"
  done | sort
  ```
- Last refreshed: **2026-07-12 (evening)**.

## TL;DR

The spine reproduces GKD (~0.83). In **round 1** the aux arms all sat **≤ A**; the
**round-2** standard-OPD recipe (`base_beta=1`, `base_temperature=1`,
`rollout_temperature=1`) pulled them up to **parity with A**, but **no arm yet beats A
robustly** — only scattered wins (B on DAPO, C/D on GSM8K-LoRA). The live
layer / aux_weight / top-k search (§4) is the attempt to turn that into a robust B>A.

## 1. Repro — GKD spine (GSM8K)

| run | finetune | acc | note |
|---|---|---|---|
| student, untrained | — | 0.749 | step-0 anchor |
| GKD repro (lr 2e-5) | full | **0.826** | reproduces ms-swift 0.76→0.83 ✓ |
| GKD repro | lora | 0.820 | |

## 2. Round 1

Aux arms only (A ≈ the repro anchor). "Current recipe" = the round-1 profile; a
separate **legacy-BCE** recipe (lower training volume, not comparable) is footnoted.

### GSM8K — current recipe

| arm | full | lora |
|---|---|---|
| A logits | *(≈0.826)* | *(≈0.820)* |
| B logit_lens | 0.805 | 0.781 |
| C jspace | 0.785 | 0.760 |
| D hidden_mse | 0.805 | 0.772 |
| E sym_jlens | **0.821** | **0.783** |

### DAPO → MATH — current recipe (`c256m20`; step-0 baseline 0.647)

| arm | full | lora |
|---|---|---|
| A logits | **0.686** | **0.666** |
| B logit_lens | 0.661 | 0.628 |
| C jspace | 0.646 | 0.630 |
| D hidden_mse | 0.649 | 0.630 |
| E sym_jlens | 0.657 | 0.639 |

**Reading:** on DAPO, round-1 A dominates and every aux arm trails (some at/below the
0.647 baseline). This motivated the round-2 recipe change.

> Legacy-BCE recipe (separate, not comparable): GSM8K logitlens-full 0.780 /
> jlens-full 0.773 / symjlens-full 0.801; MATH b-full 0.641 / c-full 0.646 /
> e-full 0.642. See `bcde-round1.md` / `dapo17k-math-round1.md`.

## 3. Round 2 — standard OPD (`base_beta=1`, `temp=1`)

### GSM8K (all final step-300)

| arm | full | lora |
|---|---|---|
| **A** logits | **0.836** | 0.814 |
| B logit_lens | 0.819 | 0.814 |
| C jspace | 0.835 | 0.820 |
| D hidden_mse | 0.829 | **0.822** |
| E sym_jlens | 0.836 | 0.816 |

**Reading:** full — A ≈ E ≈ C > D > **B (last)**. lora — **D > C > E > A = B** (on LoRA
the aux actually helps past A).

### DAPO → MATH

| arm | full | lora |
|---|---|---|
| A logits | 0.678 | 0.664 |
| B logit_lens | **0.678** | **0.670** |
| C jspace | **0.678** | 0.660 |
| D hidden_mse | 0.666‡ (s150) | 0.654‡ (s100) |
| E sym_jlens | 0.646‡ (s50) | 0.665‡ (s50) |

**Reading:** B ties/edges A (LoRA B 0.670 vs A 0.664). Clear progress over round 1,
where A dominated. D/E still training.

## 4. Follow-up search (GSM8K, **in progress**)

Driven by the lens-depth prior (logit-lens content is deep, ~0.86 depth → teacher
layer ~27-28; jlens earlier, ~0.66 → ~20-21). Anchors (step-300 finals): **A = 0.836**,
**B[16,24] = 0.819**. Values below are provisional (latest step).

| experiment | config | acc | step |
|---|---|---|---|
| **B layer search (deep)** | l28 | **0.823**‡ | s150 |
| | [16,20,24,28] | 0.820‡ | s150 |
| | l24 | 0.814‡ | s150 |
| | l20 | 0.814‡ | s150 |
| | l26 / l27 / l30 | queued | — |
| **aux_weight** (B[16,24]) | 0.003 | 0.812‡ | s100 |
| | 0.03 | 0.810‡ | s50 |
| | 0.1 / 0.3 | queued | — |
| **top-k** (B[16,24]) | 50 / 200 / 1000 | queued | — |
| **C jlens (earlier layers)** | l20 / [12,20] | queued | — |

**Early signal:** `l28@s150 = 0.823` already exceeds the B[16,24] final (0.819) and is
still climbing — consistent with "logit-lens content lives deep". Not yet a conclusion
(half-trained). The `2f1372c4` cron reads these to their final step every 3h and refines
around winners.
