# Results summary (all rounds)

Cross-round dashboard of the main eval numbers. Per-round detail (dataset identity,
job ids, commits, caveats) lives in the sibling `docs/experiments/<round>.md`.

Arms: **A** `logits` (plain OPD) · **B** `logit_lens` · **C** `jspace` (Jacobian-lens)
· **D** `hidden_mse` (bridge) · **E** `symmetric_jlens`.

Numbers are **final step-300 test accuracy** unless marked `‡` (still training — the
value is the latest logged step, provisional).

- GSM8K test = 1319 examples, se ≈ 0.011 — **treat gaps < ~0.015 as ties.** MATH sets: the
  in-training eval is **MATH-500** (`HuggingFaceH4/MATH-500`, n=500, se ≈ 0.022 — MATH-500 gaps
  < ~0.03 are ties); the round-3 finals are the full **MATH-5000** (`hendrycks_math` test, n=5000,
  se ≈ 0.0067) + AIME24/25 + AIMO (Avg@16). The round-1/2 MATH tables below are also the full
  MATH-5000 in-training eval (se ≈ 0.0067; the earlier "2500 / se 0.009" was a doc error — those
  configs ran `type:"math"` with `eval_max_samples:null` = all 5000).
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
(C 0.835→0.846, E 0.836→0.841; it does *not* help logit-lens). **Seed replication (s42–s44)
has now settled the two single-seed leads — and neither survives.** The peaks (E symjlens
[16,24] rev × temp=2 = 0.852, C jlens [12,16] × reverse = 0.851) were upward seed
fluctuations: E's 3-seed mean is **0.842** (+0.006 vs A, well under 1 se) and C's is **0.835**
(≈ A, a tie). **No GSM8K config beats A by more than a tie-margin.** The reverse-KL /
temperature *levers* are real and reshape the depth response, but they do not translate into a
robust GSM8K accuracy win over plain OPD (A). Negatives confirmed: top_k does not stack with
reverse-KL; B's knobs are compensatory (best stays plain l24 = 0.845); temperature=2 is
arm-specific (helps E, hurts B, ~neutral C). Attention now shifts to **round 3 (DAPO→MATH, §5)**,
where the same C/E levers are being tested against a MATH-calibrated lens.

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
(no help). Reverse-KL helps the **Jacobian** arms at this pair — but the B control below
shows it is really a *shallow-layer* lever that helps logit_lens too, just not at B's deep peak.

### B control — is reverse-KL really Jacobian-only? (4395–4399, full)

Retesting reverse-KL on **logit_lens** across depths shows the "[16,24] no help" above was a
layer artifact — the effect is **depth-dependent**:

| B logit_lens config | acc | vs |
|---|---|---|
| [12,16] reverse | **0.842** | > [12,16] fwd 0.828 (**+0.014** — reverse *helps* shallow) |
| [12,16] forward | 0.828 | the shallow-pair baseline |
| [12,16] reverse × temp=2 | 0.837 | < temp=1 (0.842) — temp=2 hurts B (again) |
| [24,28] reverse | 0.835 | deep pair ≈ A |
| l24 reverse | 0.825 | ≪ plain l24 fwd 0.845 (**−0.020** — reverse *hurts* the deep peak) |

**Reading:** reverse-KL is **not** Jacobian-specific — it is a **shallow / workspace-band**
lever. It lifts logit_lens at [12,16] (0.828→0.842) just as it lifts C jlens there, but *hurts*
B's deep l24 content (0.845→0.825). Since l24-forward (0.845) still beats every reverse-B, **B's
champion is unchanged: plain deep l24, forward-KL.** Unifying prior: **forward-KL for deep
content (B), reverse-KL for the shallow workspace band (C, and B at [12,16])**.

### Leaders vs A — first configs to clear the tie-margin

| config | acc (s42) | Δ vs A | |
|---|---|---|---|
| **E symjlens [16,24] rev × temp=2** | 0.852 | +0.016 | **seed 42 only — deflates on replication** |
| **C jlens [12,16] × reverse** | 0.851 | +0.015 | **seed 42 only — deflates on replication** |
| C jlens [16,24] reverse | 0.846 | +0.010 | tie |
| B logit_lens l24 | 0.845 | +0.008 | tie |
| E symjlens [24,28] rev | 0.848 | +0.012 | tie |
| E symjlens [16,24] reverse | 0.841 | +0.005 | tie |
| C jlens [12,20] fwd | 0.841 | +0.005 | tie |

Two seed-42 configs cleared A by > 1 se, so we ran the **seed replication below** to check whether
the peaks were real. **They were not** — see the next subsection: both means fall back to a tie with A.

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
| [20,24] | 0.842 | | [16,20] | 0.830 |
| [8,12] | 0.839 | | [24,28] | 0.829 |
| [20] (single) | 0.837 | | | |

Non-monotonic: C's reverse-KL peak is **[12,16]** (an early **workspace-band** pair, matching
the depth prior), *not* the deep pairs — acc falls off monotonically as the pair moves deeper
([12,16] 0.851 → [16,24] 0.846 → [20,24] 0.842 → [24,28] 0.829).

**E `symmetric_jlens` reverse-KL, by layer pair** (aux 0.01):

| layers | acc | | layers | acc |
|---|---|---|---|---|
| **[24,28]** | **0.848** | | [12,20] | 0.840 |
| [20,24] | 0.847‡ (s200) | | [12,24] | 0.838 |
| [16,24] | 0.841 | | | |

**Opposite of C:** E reverse-KL likes the **deep** band ([24,28] > [20,24] > [16,24]), where C
likes the early [12,16]. The symmetric (both-sided) readout apparently wants the deeper teacher
signal that C's teacher-only readout does not. `aux_weight` tuning on the champions is a dead
knob — moving it off 0.01 (to 0.005/0.02) hurts both C and E (≈0.82, provisional).

**Temperature=2 on the aux (softer lens targets):**

| arm · config | temp=1 | temp=2 | Δ |
|---|---|---|---|
| **E symjlens [16,24] rev** | 0.841 | **0.852** | **+0.011** |
| C jlens [16,24] rev | 0.846 | 0.841 | −0.005 |
| B logit_lens l24 | 0.845 | 0.832 | −0.013 |

Temperature is **arm-specific**: softening the aux helps **E** (its readout is on both sides,
so a softer target may reduce over-fitting to student-lens noise) but *hurts* B and is ~neutral
for C. This is the lever that lifted E's *single seed* to 0.852 — see the seed check below.

### Seed replication — do the two leads survive? (final)

The two seed-42 leads were re-run at seeds 43 and 44 (jobs 4445/4446 = C, 4447/4448 = E; all full,
step-300). A **robust** win needs the 3-seed mean to clear **A + 1 se = 0.847**.

| lead | s42 | s43 | s44 | **mean** | Δ vs A | verdict |
|---|---|---|---|---|---|---|
| **C jlens [12,16] rev** | 0.851 | 0.831 | 0.823 | **0.835** | −0.001 | **not robust — ties A** |
| **E symjlens [16,24] rev × t2** | 0.852 | 0.837 | 0.836 | **0.842** | +0.006 | **not robust — < 1 se** |

**Neither lead survives.** Both seed-42 peaks were **upward fluctuations**: C's mean (0.835) lands
right on A, and E's (0.842) keeps only a sub-1-se edge (+0.006, need +0.011). E is at least the
*tighter* lead (seeds 0.852/0.837/0.836, range 0.016) while C is noisier (0.851/0.831/0.823, range
0.028). **Conclusion for GSM8K: no config beats plain OPD (A = 0.836) by more than a tie-margin.**
The reverse-KL and temp=2 levers are genuine (they shift the depth response and the seed-42 draw),
but on GSM8K they do not buy a replicable accuracy gain over A. The direction question moves to MATH
(§5), where the lens is being recalibrated to the domain.

## 5. Round 3 — DAPO-Math-17k → MATH (reverse-KL × lens calibration)

Round 3 ports the two GSM8K Jacobian leads (C jlens [12,16], E symjlens [16,24] × temp=2) onto the
DAPO→MATH spine and runs a **2×2 that separates the KL direction from the lens calibration**:

- **Wave A** — GSM8K-domain lens (`qwen3p5_9b_v2/lens.pt`), the *same* artifact the GSM8K search used
  (jobs 4419–4422). This carries the GSM8K recipe over verbatim.
- **Wave B** — **MATH-calibrated** lens (`qwen3p5-{9b,2b}-jlens-math.pt`, fit on hendrycks_math),
  crossed with **forward vs reverse KL** (jobs 4440–4443). Wave B isolates the lens-calibration gain
  from the KL-direction gain.

In-training eval = **MATH-500** (n=500, se ≈ 0.022 → treat MATH-500 gaps < ~0.03 as ties). Untrained
anchor **0.646**. Finals (queued once a leader settles) = MATH-5000 + AIME24/25 + AIMO Avg@16;
teacher ceilings GSM8K 0.953 / MATH-5000 0.849.

### MATH-500 in-training (‡ = still training, value is latest logged step)

| arm · config | lens | KL | acc | status |
|---|---|---|---|---|
| **A logits** (baseline) | — | — | **0.684** | step-300 final |
| B logit_lens l24 | v2 | fwd | 0.670 | step-300 final |
| C jlens [12,16] | v2 (Wave A) | reverse | 0.664 | step-300 final |
| E symjlens [16,24] t2 | v2 (Wave A) | reverse | 0.668 | step-300 final |
| C jlens [12,16] | math (Wave B) | reverse | **0.682** | step-300 final |
| C jlens [12,16] | math (Wave B) | forward | 0.662 | step-300 final |
| E symjlens [16,24] t2 | math (Wave B) | reverse | 0.672‡ (s200) | 4442 running |
| E symjlens [16,24] t2 | math (Wave B) | forward | 0.664‡ (s200) | 4443 running |

**Reading (Wave A final; Wave B C final, Wave B E still training).** Plain **A `logits` (0.684)**
remains the MATH reference. The headline update: with the **MATH-calibrated lens, C [12,16] reverse-KL
recovers to 0.682** — parity with A and **+0.018 over the identical config on the stale GSM8K lens**
(Wave-A C-rev 0.664). So the Wave-A reverse-KL "drag" was **largely the stale lens, not reverse-KL
itself**: recalibrating the lens to the domain lifts C-rev back to A-level. Within Wave B, reverse also
edges forward for C (rev 0.682 vs fwd 0.662) — but the two crossed over between steps (fwd led 0.676 vs
0.654 at s200), so the rev-vs-fwd margin is inside MATH-500 noise (se 0.022) on a single seed. **E's
2×2 is still training** (rev 0.672 / fwd 0.664 @ s200); the direction verdict holds until E's step-300.
**Net so far: the calibrated lens closes C up to A on MATH — but no aux config has yet *beaten* A**, and
the one parity result (C-rev-math 0.682 ≈ A 0.684) is a single seed within 1 se, i.e. a tie, not a lead.
