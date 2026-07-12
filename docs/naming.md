# Experiment naming & trackio conventions

opdlens experiments are compared **across arms on a shared dataset**, so trackio is
organized to make that comparison one glance in the dashboard. Follow this for every
run — it is what lets arms A–E line up side by side.

## 1. Project = dataset

One trackio project per **dataset**, named `opdlens-<dataset>`:

| dataset | `trackio_project` |
|---------|-------------------|
| GSM8K   | `opdlens-gsm8k`   |
| MATH    | `opdlens-math`    |

Set it in the config's `trackio_project`. Every arm (A–E) and finetune variant on a
dataset lands in the same project, so their curves overlay directly.

## 2. Run name = `experiment_name`

The trackio run name is the config's `experiment_name` (not a timestamp-hash). Format:

```
<arm>-<finetune>[-<sweep>][-s<seed>]
```

**`<arm>`** — one tag per `arm.type` (the single scientific seam between experiments):

| arm | `arm.type`        | tag         |
|-----|-------------------|-------------|
| A — plain OPD (final-logit KL / GKD) | `logits`          | `logits`    |
| B — teacher logit-lens               | `logit_lens`      | `logitlens` |
| C — teacher Jacobian-lens            | `jspace`          | `jlens`     |
| D — raw-hidden MSE (bridge)          | `hidden_mse`      | `hiddenmse` |
| E — symmetric Jacobian-lens          | `symmetric_jlens` | `symjlens`  |

**`<finetune>`** — `lora` or `full`.

**`<sweep>`** *(optional)* — only the knob you are varying in **this** comparison,
compressed: `lr2e5`, `beta0.3`, `T0.9`, `r16`, `aux0.5`. Everything else is in the
logged config (§4), so don't cram the full config into the name — just the axis.

**`-s<seed>`** *(optional)* — seed replicate: `s0`, `s1`, …

### Examples

| run name              | meaning                                   |
|-----------------------|-------------------------------------------|
| `logits-lora`         | arm A, LoRA — the GKD baseline            |
| `logits-full`         | arm A, full fine-tune                     |
| `logits-full-lr2e5`   | arm A, full FT, one lr-sweep point        |
| `jlens-lora`          | arm C, LoRA                               |
| `jlens-lora-aux0.5`   | arm C, LoRA, aux weight 0.5               |
| `logitlens-lora-s1`   | arm B, LoRA, seed 1                       |

## 3. Uniqueness & resume

`experiment_name` is the run's identity in trackio, opened with `resume="allow"`:

- **Same name → resumes/appends** to that run. This is what lets checkpoint
  auto-resume continue one curve — intended.
- For a **fresh, independent** run (new seed, a redo), use a **distinct** name (add
  `-s<seed>` or a sweep tag), or delete the old run first. Never point two different
  configs at the same name.

## 4. Hyperparameters are logged

`trackio.init` records the full trainer config (`model_dump`) on every run — the arm
block, `base_beta`/`base_temperature`, optimizer, LoRA spec, batch, steps, etc. — so
they are queryable and you don't need them in the run name. It also stamps
`opdlens_version` (package version + git commit), so every run is traceable to the
exact code that produced it — the launch scripts pin this via
`OPDLENS_EXPERIMENT_COMMIT`, so the recorded commit is the snapshot's, not repo HEAD:

```bash
trackio get run   --project opdlens-gsm8k --run logits-lora --json      # includes config
trackio query project --project opdlens-gsm8k \
  --sql "SELECT run_name, MAX(step) AS last_step FROM metrics GROUP BY run_name"
```

## 5. Where the data lives

trackio's store is **in-project** at `runs/trackio/` (one `<project>.db` per project) —
`init_tracker` defaults `TRACKIO_DIR` there, not the global `~/.cache/huggingface/trackio`.
Launch the dashboard from this repo's venv:

```bash
TRACKIO_DIR=$PWD/runs/trackio uv run trackio show --host 0.0.0.0
```

## 6. Housekeeping

Delete aborted / short runs so they don't clutter a comparison — the raw metrics also
survive in `runs/<experiment_name>/<run_id>/metrics.jsonl`:

```python
from trackio.sqlite_storage import SQLiteStorage
SQLiteStorage.delete_run("opdlens-gsm8k", "logits-lora-badseed")
```
