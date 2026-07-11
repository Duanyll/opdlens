# Instructions for agents working on `opdlens`

`opdlens` studies **on-policy distillation with intermediate-layer (lens)
supervision**. Its scientific core is a **four-arm experiment** (A logits / B
logit-lens / C jacobian-lens / D hidden-mse). The arms share one training + eval
spine and differ **only** in the auxiliary loss.

This project is deliberately small and built to be extended by agents. The rules
below are what keep parallel work from diverging into incomparable results.

## The one design law

**Store configuration in pydantic models; pass data through function arguments.**
Nearly every class is a `BaseModel` with a `type: Literal[...]` discriminator and
its behavior as methods; data (tensors, batches) flows as function args. Each knob
is declared on the class whose behavior consumes it — no threading through call
layers. Standalone `XxxConfig` only at the entrypoint (`config.py`).

## The single seam — do not widen it

- The four arms differ **only** in `arm.aux_loss` (`opdlens/arms.py`, four bodies
  stacked side by side). B↔C differ by one line (the teacher readout operator).
- The base OPD loss, rollout, teacher forward, optimizer, eval, and checkpointing
  are the **shared spine** (`opdlens/losses.py`, `opdlens/training/`). They are
  identical for every arm by construction.
- **An experiment is a config, not code.** A new experiment = a new
  `examples/*.jsonc` that differs from the others only in the `arm` block. Never
  edit the train/eval loop to change an experiment — that is a *scientific* bug
  (it confounds cross-arm comparison), not a style choice.

## Extension points (all plain discriminated unions — no registry/plugin)

Use the pre-`faa22ec` flow-control style: `Annotated[Annotated[A, Tag("a")] | ...,
Discriminator("type")]` + a module-level `TypeAdapter` + `parse_x()`.

- **New arm**: add a member to `opdlens/arms.py` (declare `teacher_layers`, one
  `aux_loss`), add it to the `Arm` union. Shared masked-KL machinery lives in
  `losses.lens_kl` — reuse it.
- **New benchmark**: add a member to `opdlens/benchmarks/` — a behavior class
  owning `build_prompt` **and** `grade`. The *same* `build_prompt`/`grade` serve
  training rollout, in-loop eval, and reported eval; there is exactly one grader
  and one prompt per dataset (never select a checkpoint under a different metric
  than you report).
- Swappable choices are `Literal` fields, not new unions, unless a genuinely new
  behavior family appears.

## Fair comparison is structural, not a convention

`tests/test_spine.py` pins it: with `aux_weight == 0` every arm gates off its aux
(no hidden capture, no RNG use) and reduces to the single shared `opd_base_loss`;
and the four example configs differ only in the `arm` block. Keep it green. Eval
runs at fixed steps — no best-step selection.

## Package management & checks

Python 3.12; **torch is held to 2.11 + vllm 0.24** because the colocated in-process
vLLM must share the training env (see the design plan / §9). Use `uv`:

- `uv add` / `uv remove` for dependencies — never edit `pyproject.toml` deps by hand.
- `uv run pyright opdlens` and `uv run ruff format opdlens && uv run ruff check --fix opdlens` — keep both clean.
- `uv run pytest tests` — keep green.
- `uv run opdlens schema` — regenerate `schema/opd.schema.json` after editing config models.

## Running

- Train: `uv run opdlens launch examples/arm_a.jsonc` (torchrun parent → colocated
  vLLM + multi-GPU DDP; FSDP2 is the next step, isolated to
  `GenerationMixin.sync_weights`).
- Offline artifacts (before arms C/D): `uv run python -m opdlens.fit.fit_jlens ...`
  and `... fit_bridge ...`. The trainer only *loads* `lens.pt` / `bridge.pt`.
- The `opdlens/jlens/` package is vendored third-party (Anthropic, Apache-2.0) —
  keep its license headers; do not add headers to opdlens's own code.

## Cluster

No GPU in the interactive/agent session. Check `nvidia-smi`; if absent, grab a GPU
with `srun --gpus=N uv run ...` (default `-p compute -q interactive`, 4 GPU / 4 h).
Use `sbatch` for long runs. Never `find`/`grep` broadly over `/gdata` or `/home`.

## Logging

Never `print` in library code. `from opdlens.utils.logging import get_logger,
console`; pass `console` to Rich progress bars.
