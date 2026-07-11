# Instructions for agents working on `opdlens`

`opdlens` studies **on-policy distillation with intermediate-layer (lens)
supervision**: several distillation arms that share one training + eval spine and
differ **only** in an auxiliary loss. It is deliberately small and built to be
extended in parallel by many agents. The rules below are what keep that parallel
work from diverging into incomparable results — follow them exactly.

## The scientific discipline (read first)

These are laws, not style preferences. Breaking one is a *scientific* bug — it
confounds cross-arm comparison — not a matter of taste.

1. **An experiment is a config, not code.** A new experiment is a new example
   config that differs from the others only in its arm block. Never edit the
   train/eval loop, rollout, teacher forward, optimizer, or checkpointing to
   change an experiment.
2. **One seam, kept narrow.** The arms differ only in the auxiliary loss;
   everything else is identical for every arm by construction. Do not widen the
   seam, and reuse the shared loss machinery rather than reimplementing it per arm.
3. **Fair comparison is structural.** With the aux weight set to zero, every arm
   must gate its aux completely off (no hidden capture, no RNG use) and reduce to
   the shared base loss; the example configs must differ only in the arm block.
   The spine tests pin this — keep them green.
4. **One grader, one prompt per dataset.** The same prompt-builder and grader serve
   training rollout, in-loop eval, and reported eval. Never select a checkpoint
   under a different metric than you report.
5. **Eval at fixed steps.** No best-step selection.

## The one design law

**Store configuration in Pydantic models; pass data through function arguments.**

1. Nearly every class extends `pydantic.BaseModel` with its behavior as methods;
   data (tensors, batches) flows as function arguments. Avoid untyped kwargs and
   loose dicts.
2. Declare each knob on the class whose behavior consumes it — do not thread config
   through call layers. Use a standalone `XxxConfig` only at the entrypoint.
3. Prefer composition and mixins over inheritance.
4. Extension points are plain discriminated unions (`Tag` + `Discriminator` + a
   module-level `TypeAdapter` + a `parse_x()` helper) — no registry or plugin
   system. To add an arm or a benchmark, add a member to the relevant union and
   follow the existing members. Swappable choices are `Literal` fields, not new
   unions, unless a genuinely new behavior family appears.
5. Regenerate the jsonc schema after editing any config model: `uv run opdlens schema`.

## Package management

Use `uv`. Never edit `pyproject.toml` dependencies by hand.

1. `uv add` / `uv remove` to manage dependencies.
2. Run scripts with `uv run <script>` or `uv run -m <module>`, never bare `python`.
3. torch and vLLM are pinned *together* (see `pyproject.toml`) because the
   colocated in-process vLLM shares the training env — do not bump either alone.

## Type hints and linting

Keep both clean:

```bash
uv run pyright opdlens
uv run ruff format opdlens && uv run ruff check --fix opdlens
```

1. Maintain type hints. Python 3.12 — prefer `dict`, `list`, `|` over
   `typing.Dict` / `List` / `Union`.
2. Format first, then `check --fix`, then fix the rest by hand. Avoid `# noqa`.
3. IDE diagnostics may be stale — trust the CLI output.

## Logging

1. Never `print()` in library code. Use:
   ```python
   from opdlens.utils.logging import get_logger, console
   logger = get_logger(__name__)
   ```
2. Pass the `console` instance explicitly for any rich printing, especially
   progress bars.

## Vendored code

The vendored `jlens` package is third-party (Anthropic, Apache-2.0) — keep its
license headers, and do not add headers to opdlens's own code.

## Running

1. Train from a config: `uv run opdlens launch examples/<arm>.jsonc`.
2. Some arms depend on offline-fit artifacts. Fit them with the fit tooling; the
   trainer only *loads* these artifacts, it never fits them inline.

## Cluster

No GPU in the interactive/agent session. Check `nvidia-smi`; if absent, grab one
with `srun --gpus=N uv run ...` (default `-p compute -q interactive`, 4 GPU / 4 h),
and `sbatch` for long runs. Never `find` / `grep` broadly over `/gdata` or `/home`.

## Workflow

1. Read the existing member of the same union (arm, benchmark, …) before adding a
   new one, and follow that pattern.
2. Keep the spine tests green: `uv run pytest tests`.
3. Fix all type and lint errors, and regenerate the schema, before finishing.
4. To run a full training / eval pipeline past the Bash timeout, use `sbatch` (or
   `tmux` if Slurm is unavailable).
