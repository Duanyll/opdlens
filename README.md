# opdlens

On-policy distillation with intermediate-layer (**lens**) supervision.

Teacher → student knowledge distillation on self-sampled (on-policy) sequences,
with an auxiliary term that supervises an intermediate layer. Five arms share a
single training/eval spine and differ **only** in the auxiliary loss:

| arm | `arm.type` | auxiliary supervision |
|-----|------------|-----------------------|
| A   | `logits`     | none — plain OPD (final-logit KL) |
| B   | `logit_lens` | teacher **logit-lens** readout → vocab-space KL |
| C   | `jspace`     | teacher **Jacobian-lens** readout `unembed(J·h)` → vocab-space KL |
| D   | `hidden_mse` | OPRD-style raw-hidden MSE (through a frozen bridge) |
| E   | `symmetric_jlens` | **symmetric Jacobian-lens** readouts on teacher and student → vocab-space KL |

The scientific question: does a **lens readout** (B/C/E) beat plain OPD (A) and
raw-hidden MSE (D), does the Jacobian-lens (C) beat the logit-lens (B), and does
symmetric representational readout (E) improve on C's asymmetric readout?

## Design

One `OpdTrainer` composed from cohesive mixins (rollout / teacher / generation /
optim / eval / checkpoint / logging). The five arms are a flat discriminated
union (`opdlens/arms.py`); benchmarks (GSM8K / MATH / MATH-500 / AIME) are a
behavior-rich union owning prompt construction **and** grading. Colocated
in-process vLLM samples on-policy; multi-GPU DDP now, FSDP2 next.

By design, with `aux_weight=0` every arm reduces to identical base OPD, and
experiment configs differ only in the `arm` block — fair comparison is
structural, not a convention.

## Layout

- `opdlens/` — the package (trainer, arms, benchmarks, losses, offline-fit tooling).
- `examples/` — minimal, illustrative configs only: one per arm (`arm_a`…`arm_e`)
  plus `smoke.jsonc`. Meant to be read and copied, not to record a study.
- `experiments/` — the actual research config matrices, **one subdirectory per
  round** (`experiments/gsm8k_round1/`, `experiments/dapo17k_math_round1/`,
  `experiments/repro_gkd/`). Each `.jsonc` is a committed, self-contained run spec
  and is the source of truth — the scripts that generated them are intentionally
  **not** tracked (recover from git history if a matrix ever needs regenerating).
  Configs are only comparable **within one recipe family**: the round dirs and
  `repro_gkd/` share the current GKD recipe (β=0.5, global batch 96, cosine),
  whereas `experiments/legacy/<bench>/` holds the earlier, far-shorter legacy-BCE
  recipe (β=0, global batch 8, constant LR) — kept apart precisely so the two are
  never mixed in one comparison.
- `docs/experiments/<round>.md` — the write-up and per-run log (job ids, commits,
  results) for the matching `experiments/<round>/`.
- `docs/experiments/results.md` — cross-round results dashboard (repro / round1 /
  round2 / live search), refreshed from the `logs/` eval lines.
- `scripts/run.sbatch <config.jsonc> [<commit>]` — the **single** launcher. It
  archives the repo at `<commit>` (default `HEAD`) into an immutable snapshot and
  trains from it, exporting `OPDLENS_EXPERIMENT_COMMIT` so the code version lands
  in trackio; torchrun is sized to the GPUs Slurm allocated. Override resources on
  the sbatch line, e.g.
  `sbatch -J logitlens-full scripts/run.sbatch experiments/dapo17k_math_round1/logitlens_full.jsonc`.
  (`scripts/` otherwise holds only offline-fit helpers such as `fit_student_jlens.sh`.)
