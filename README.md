# opdlens

On-policy distillation with intermediate-layer (**lens**) supervision.

Teacher → student knowledge distillation on self-sampled (on-policy) sequences,
with an auxiliary term that supervises an intermediate layer. Four arms share a
single training/eval spine and differ **only** in the auxiliary loss:

| arm | `arm.type` | auxiliary supervision |
|-----|------------|-----------------------|
| A   | `logits`     | none — plain OPD (final-logit KL) |
| B   | `logit_lens` | teacher **logit-lens** readout → vocab-space KL |
| C   | `jspace`     | teacher **Jacobian-lens** readout `unembed(J·h)` → vocab-space KL |
| D   | `hidden_mse` | OPRD-style raw-hidden MSE (through a frozen bridge) |

The scientific question: does a **lens readout** (B/C) beat plain OPD (A) and
raw-hidden MSE (D), and does the Jacobian-lens (C) beat the logit-lens (B)?

## Design

One `OpdTrainer` composed from cohesive mixins (rollout / teacher / generation /
optim / eval / checkpoint / logging). The four arms are a flat discriminated
union (`opdlens/arms.py`); benchmarks (GSM8K / MATH / MATH-500 / AIME) are a
behavior-rich union owning prompt construction **and** grading. Colocated
in-process vLLM samples on-policy; multi-GPU DDP now, FSDP2 next.

`tests/test_spine.py` pins the invariant: with `aux_weight=0` every arm reduces
to identical base OPD, and experiment configs differ only in the `arm` block —
fair comparison is structural, not a convention.
