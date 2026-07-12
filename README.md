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

**Arm E (symmetric J-lens; formerly planned as C′).** Like C, but the *student*
is also read through its own **offline-fit** Jacobian-lens:
`KL(unembed_t(J_t·h_t) ‖ unembed_s(J_s·h_s))`, vs C's asymmetric
`KL(teacher_jlens ‖ student_logit_lens)`. `J_s` is fit once on the frozen init
checkpoint at the mapped student layers (loaded, never fit inline). This tests
*representational-content* alignment, so it is a **parallel axis**, not a
one-variable step on the A→B→C ladder. Dynamic re-fit of `J_s` is out of scope
(cost dominates training; a moving target makes the loss non-stationary).

## Design

One `OpdTrainer` composed from cohesive mixins (rollout / teacher / generation /
optim / eval / checkpoint / logging). The five arms are a flat discriminated
union (`opdlens/arms.py`); benchmarks (GSM8K / MATH / MATH-500 / AIME) are a
behavior-rich union owning prompt construction **and** grading. Colocated
in-process vLLM samples on-policy; multi-GPU DDP now, FSDP2 next.

`tests/test_spine.py` pins the invariant: with `aux_weight=0` every arm reduces
to identical base OPD, and experiment configs differ only in the `arm` block —
fair comparison is structural, not a convention.
