# Layer Lens Visualizer

Interactive tool that visualizes, **for every layer of a language model**, the
vocabulary tokens the layer is disposed to emit — under two lenses:

- **LogitLens** — decode a layer's raw residual through the model's own final
  norm + unembedding (`unembed(h_l)`). No fitting required.
- **Jacobian lens (J-lens)** — first linearly transport the residual into the
  final-layer basis with the fitted average input–output Jacobian
  (`unembed(J_l @ h_l)`), then decode. Surfaces interpretable tokens several
  layers earlier than the LogitLens. Requires a one-time fit of the `J_l`
  matrices (cached to disk).

Each cell shows the **Top-K tokens and their probabilities**. The final layer's
LogitLens row is the model's actual output distribution.

This is a rewrite of the Anthropic `jacobian-lens` reference repo's static slice
page into a live, interactive per-layer read-out. It reuses the `jlens` core
(model wrapper, activation hooks, `J_l` transport) unchanged.

---

## Quick start

Everything runs in a self-contained `uv` environment (Python 3.12). `uv` was
installed to `~/.local/bin`.

```bash
cd /Users/bytedance/Desktop/JOPD/visual_workspace
export PATH="$HOME/.local/bin:$PATH"

# 1. (one-time, slow) fit + cache the Jacobian lens for the demo model
uv run python fit_lens.py

# 2. launch the app (loads the cached lens instantly)
uv run python app.py
```

Then open the printed local URL (default http://127.0.0.1:7860).

If you skip step 1, launch with LogitLens only:

```bash
uv run python app.py --no-fit
```

The app will also auto-fit on first launch if no cached lens exists (adds a few
minutes to startup).

## Using the app

1. Type any **prompt**.
2. Drag the **position** slider to choose which token to read out (defaults to
   the last token).
3. Set **Top-K** (tokens shown per layer).
4. Toggle **LogitLens** / **Jacobian lens**.
5. Click **Analyze**.

The table shows one row per layer (deepest at top), with the top-K tokens and
probability bars for each lens side by side.

## Model

The demo uses **`Qwen/Qwen3.5-2B`** (24 layers, d_model 2048). Pick another
HuggingFace decoder with:

```bash
uv run python app.py --model Qwen/Qwen3-0.6B
# or
LENS_MODEL=Qwen/Qwen3-0.6B uv run python app.py
```

Fit its lens first with `uv run python fit_lens.py --model <name>`.

## Files

| File | Purpose |
|---|---|
| `lens_engine.py` | Loads the model, runs both lenses, returns Top-K read-outs. |
| `app.py` | Gradio UI. |
| `fit_lens.py` | One-time J-lens fit, caches `lenses/<model>__jlens.pt`. |
| `lenses/` | Cached fitted `J_l` matrices. |

## Notes on fitting cost

J-lens fitting runs `ceil(d_model / dim_batch)` backward passes per prompt over
a retained autograd graph. On Apple Silicon (MPS) the cost is dominated by the
**lowest** fitted layer (it sets how much of the network the graph spans), not
the number of layers — so fitting several layers at once is nearly free once you
pay for the lowest one.

Because Qwen3.5 uses a linear-attention path with an O(seq²) buffer, keep
`--dim-batch` modest (16 is safe on 36 GB unified memory) and `--max-seq-len`
short; larger values can exhaust MPS memory.

Defaults: layers `[6, 10, 14, 17, 20]`, 6 prompts, `dim_batch=16`,
`max_seq_len=64`. LogitLens needs no fitting and is always available at every
layer.
