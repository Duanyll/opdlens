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
- Last refreshed: **2026-07-13**.

## TL;DR

The spine reproduces GKD (~0.83). In **round 1** the aux arms all sat **≤ A**; the
**round-2** standard-OPD recipe (`base_beta=1`, `base_temperature=1`,
`rollout_temperature=1`) pulled them up to **parity with A**. The GSM8K follow-up
search (§4) has now finished its first pass and finds several configs that edge
**above** A — C jlens + reverse-KL 0.846, B at the single deep layer 24 0.845, E
symjlens + reverse-KL 0.841 — but **every gap is ≤ 1 se (0.011), so no arm beats A by
more than a tie-margin yet.** The clearest lever is **reverse-KL on the Jacobian arms**
(C 0.835→0.846, E 0.836→0.841; it does *not* help logit-lens). **Two configs have now
cleared A by more than 1 se — the first robust leads:** **E symjlens [16,24] rev × temp=2
= 0.852** (+0.016) and **C jlens [12,16] × reverse = 0.851** (+0.015). Both are *single
seeds ~0.0002 past the 0.015 threshold*, so promising leads rather than settled wins — the
follow-up grid probes their neighbourhoods for corroboration. Negatives confirmed: top_k
does not stack with reverse-KL; B's knobs are compensatory (best stays plain l24 = 0.845);
temperature=2 is arm-specific (helps E, hurts B, ~neutral C).

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

## 4. Follow-up search (GSM8K, full only — first pass **final**)

Driven by the lens-depth prior (logit-lens content is deep, ~0.86 depth → teacher
layer ~27-28; jlens earlier, ~0.66 → ~20-21). All step-300 finals. Anchors: **A = 0.836**,
**B[16,24] = 0.819**, **C[16,24] = 0.835**, **E[16,24] = 0.836**. se ≈ 0.011; treat gaps
< 0.015 as ties.

### B `logit_lens` — teacher layer (aux 0.01)

| layers | acc | | layers | acc |
|---|---|---|---|---|
| **24** | **0.845** | | 25 / 26 / 27 | 0.832 |
| [16,20,24,28] | 0.836 | | [24,28] | 0.832 |
| [27,30] | 0.836 | | 23 | 0.826 |
| 20 / 28 / 30 | 0.833 | | [16,24] (anchor) | 0.819 |

A single **deep layer 24** is the B peak (0.845 ≈ A+0.008), and a **sharp** one — the
immediate neighbours l23 (0.826) and l25 (0.832) both sit ~0.015–0.02 below it. Pairing
16 with a deep layer, or averaging four layers, dilutes back toward A.

### B knob sweeps (at [16,24])

| aux_weight | acc | | top_k | acc |
|---|---|---|---|---|
| 0.003 | 0.834 | | 200 | **0.839** |
| 0.01 (anchor) | 0.819 | | 1000 | 0.832 |
| 0.03 | 0.824 | | 50 | 0.830 |
| 0.1 | 0.798 | | (none, anchor) | 0.819 |
| 0.3 | 0.779 | | | |

The [16,24] aux *hurts* at full strength; dialing `aux_weight` toward 0, or top-k
filtering the aux target to ~200 tokens, recovers ≈ A. Same story as "layer 24 alone" —
the [16,24] aux was over-weighted / too broad.

### C `jspace` + E `symmetric_jlens` — layer & KL direction

| arm | config | acc | | arm | config | acc |
|---|---|---|---|---|---|---|
| C jlens | [16,24] **reverse** | **0.846** | | E symjlens | [16,24] **reverse** | **0.841** |
| C jlens | [12,20] fwd | 0.841 | | E symjlens | [12,20] fwd | 0.839 |
| C jlens | 20 fwd | 0.837 | | E symjlens | [12,16,20] fwd | 0.832 |
| C jlens | [16,24] fwd (anchor) | 0.835 | | E symjlens | 12 fwd | 0.831 |
| | | | | E symjlens | 16 / 20 fwd | 0.829 |
| | | | | E symjlens | [16,20] fwd | 0.827 |

**Reverse-KL @ [16,24]:** C 0.835→0.846 (+0.011), E 0.836→0.841 (+0.005), B 0.819→0.817
(no help). Reverse-KL helps the **Jacobian** arms, not logit-lens.

### Leaders vs A — first configs to clear the tie-margin

| config | acc | Δ vs A | |
|---|---|---|---|
| **E symjlens [16,24] rev × temp=2** | **0.852** | **+0.016** | > 1 se — robust (single seed) |
| **C jlens [12,16] × reverse** | **0.851** | **+0.015** | > 1 se — robust (single seed) |
| C jlens [16,24] reverse | 0.846 | +0.010 | tie |
| B logit_lens l24 | 0.845 | +0.008 | tie |
| E symjlens [24,28] rev | 0.845‡ (s250) | +0.008 | tie (running) |
| E symjlens [16,24] reverse | 0.841 | +0.005 | tie |
| C jlens [12,20] fwd | 0.841 | +0.005 | tie |

**Two configs finally exceed A by more than 1 se (0.011)** — but both are *single seeds*
~0.0002 over the 0.015 threshold, so treat them as *promising leads, not settled wins*;
the follow-up grid probes their neighbourhoods to see if the peak is real or noise.

### Refine — best layer × best knob per arm

| arm | config | acc | vs baseline |
|---|---|---|---|
| B | l24 × top_k=200 | 0.842 | ≈ plain l24 (0.845) — no gain |
| B | l24 × aux_weight=0.003 | 0.834 | < plain l24 — hurts |
| B | l24 × top_k=200 × aux_weight=0.003 | 0.836 | = A, < plain l24 |
| C | [12,20] × reverse | 0.833 | < [16,24]rev (0.846) **and** [12,20]fwd (0.841) |
| C | l20 × reverse | 0.837 | < [16,24]rev — single layer no better |
| C | [16,24] rev × top_k=200 | 0.838 | < [16,24]rev (0.846) — top_k **hurts** |
| E | [12,20] × reverse | 0.840 | ≈ [16,24]rev (0.841) — tie |
| E | [16,24] rev × top_k=200 | 0.836 | < [16,24]rev (0.841) — top_k **hurts** |

**Reading:** two clean negatives this round. (1) **top_k does not stack with reverse-KL** —
it was a B-only *compensatory* knob (it rescued the harmful [16,24] pair) and it actively
*hurts* both Jacobian champions (C 0.846→0.838, E 0.841→0.836). (2) **Reverse-KL prefers
the deep [16,24] pair for C**: `[12,20]×reverse` (0.833) is worse than *both* its parents,
i.e. the forward layer preference ([12,20] > [16,24]) flips under reverse. So the leaders
are unchanged — **C [16,24]rev = 0.846** and **B plain l24 = 0.845**, both still ties vs A.
Next probe (queued): push C/E reverse-KL *deeper* ([20,24], [24,28]) and tune `aux_weight`
on the champion, since the aux is clearly helping and its weight is untuned for reverse.

### Reverse-KL layer sweep + temperature (C/E)

**C `jspace` reverse-KL, by layer pair** (aux 0.01):

| layers | acc | | layers | acc |
|---|---|---|---|---|
| **[12,16]** | **0.851** | | [12,20] | 0.833 |
| [16,24] | 0.846 | | [12,24] | 0.832 |
| [8,12] | 0.839 | | [16,20] | 0.830 |
| [20,24] / [24,28] | ‡ running | | [20] (single) | 0.837 |

Non-monotonic: the reverse-KL peak is **[12,16]** (an early **workspace-band** pair, matching
the depth prior), *not* the deep [16,24]. [16,20] and [12,24] are surprisingly poor — the
pair has to sit squarely in the 12–16 band.

**Temperature=2 on the aux (softer lens targets):**

| arm · config | temp=1 | temp=2 | Δ |
|---|---|---|---|
| **E symjlens [16,24] rev** | 0.841 | **0.852** | **+0.011** |
| C jlens [16,24] rev | 0.846 | 0.841 | −0.005 |
| B logit_lens l24 | 0.845 | 0.832 | −0.013 |

Temperature is **arm-specific**: softening the aux helps **E** (its readout is on both sides,
so a softer target may reduce over-fitting to student-lens noise) but *hurts* B and is ~neutral
for C. This is the lever that lifted E into a robust win.
