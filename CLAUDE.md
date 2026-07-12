# Instructions for agents working on `opdlens`

`opdlens` studies **on-policy distillation with intermediate-layer (lens)
supervision**: several distillation arms that share one training + eval spine and
differ **only** in an auxiliary loss. It is deliberately small and built to be
extended in parallel by many agents. The rules below are what keep that parallel
work from diverging into incomparable results — follow them exactly.

## The scientific discipline (read first)

The framework has settled. Two rules keep results comparable:

1. **Change behavior by adding a knob, never by mutating the old path.** To adjust
   how an experiment behaves, add an option in the right place or a new pydantic
   type, and default it to the existing behavior — so every old config keeps
   producing its old result.
2. **Never touch eval without explicit consent.** Do not change the grader,
   prompt-builder, eval metric, or eval schedule unless the user has explicitly
   agreed.

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
