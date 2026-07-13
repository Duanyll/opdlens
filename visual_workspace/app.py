# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
"""Gradio UI for the per-layer LogitLens / Jacobian-lens vocabulary visualizer.

Run:

    uv run python app.py                 # loads Qwen/Qwen3.5-2B by default
    LENS_MODEL=Qwen/Qwen3-0.6B uv run python app.py

The user picks a prompt, a token position, and a Top-K, and sees — for every
layer of the model — the top-K vocabulary tokens the layer is disposed to emit,
under both the LogitLens and the Jacobian lens, each with its probability.
"""

from __future__ import annotations

import argparse
import html
import logging
import os

import gradio as gr

from lens_engine import AnalysisResult, LensEngine, LayerReadout

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

DEFAULT_MODEL = os.environ.get("LENS_MODEL", "Qwen/Qwen3.5-2B")

DEFAULT_PROMPT = (
    "Fact: The capital of Japan is Tokyo.\n"
    "Fact: The currency used in the country shaped like a boot is"
)

# The engine is loaded once at startup and shared across requests.
ENGINE: LensEngine | None = None


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
def _prob_bar(prob: float) -> str:
    """A thin inline bar whose width encodes probability."""
    pct = max(0.0, min(1.0, prob)) * 100
    return (
        f'<div style="background:#eee;border-radius:3px;height:8px;width:60px;'
        f'display:inline-block;vertical-align:middle;overflow:hidden">'
        f'<div style="background:#f0a;height:100%;width:{pct:.1f}%"></div></div>'
    )


def _readout_cell(row: LayerReadout) -> str:
    """One layer's top-K list as an HTML fragment."""
    if not row.available:
        return '<span style="color:#bbb">— not fitted at this layer —</span>'
    items = []
    for rank, tp in enumerate(row.top):
        tok = html.escape(repr(tp.token_str)[1:-1]) or "∅"
        items.append(
            f'<div style="display:flex;align-items:center;gap:6px;'
            f'padding:1px 0;font:12px/1.4 ui-monospace,monospace">'
            f'<span style="color:#999;width:1.6em;text-align:right">{rank}</span>'
            f'<span style="min-width:8ch;color:#111;font-weight:600">{tok}</span>'
            f'{_prob_bar(tp.prob)}'
            f'<span style="color:#555;width:5ch;text-align:right">{tp.prob*100:.1f}%</span>'
            f'</div>'
        )
    return "".join(items)


def _render_tables(result: AnalysisResult, show_logit: bool, show_jlens: bool) -> str:
    """Build the side-by-side per-layer comparison table as HTML."""
    show_jlens = show_jlens and result.jlens_available
    logit_by_layer = {r.layer: r for r in result.logitlens}
    jlens_by_layer = {r.layer: r for r in result.jlens}
    layers = sorted(set(logit_by_layer) | set(jlens_by_layer), reverse=True)

    cols = []
    if show_logit:
        cols.append("LogitLens")
    if show_jlens:
        cols.append("Jacobian lens")
    if not cols:
        return "<p>Select at least one lens.</p>"

    header = (
        '<tr style="border-bottom:2px solid #ccc;text-align:left">'
        '<th style="padding:6px 10px;width:4em">Layer</th>'
        + "".join(
            f'<th style="padding:6px 10px">{c}</th>' for c in cols
        )
        + "</tr>"
    )

    body_rows = []
    for layer in layers:
        is_final = layer == result.n_layers - 1
        layer_label = f"L{layer}"
        if is_final:
            layer_label += '<br><span style="color:#f0a;font-size:10px">output</span>'
        cells = [
            f'<td style="padding:6px 10px;vertical-align:top;color:#333;'
            f'font-weight:600;border-right:1px solid #eee">{layer_label}</td>'
        ]
        if show_logit:
            cells.append(
                f'<td style="padding:6px 10px;vertical-align:top;'
                f'border-right:1px solid #eee">'
                f'{_readout_cell(logit_by_layer[layer])}</td>'
            )
        if show_jlens:
            cells.append(
                f'<td style="padding:6px 10px;vertical-align:top">'
                f'{_readout_cell(jlens_by_layer[layer])}</td>'
            )
        bg = "#fff6fb" if is_final else ("#fafafa" if layer % 2 else "#fff")
        body_rows.append(
            f'<tr style="background:{bg};border-bottom:1px solid #f0f0f0">'
            + "".join(cells)
            + "</tr>"
        )

    note = ""
    if show_jlens and not result.jlens_available:
        note = (
            '<p style="color:#c60">Jacobian lens not available — no fitted '
            "lens loaded.</p>"
        )
    elif not result.jlens_available:
        note = (
            '<p style="color:#999;font-size:12px">Jacobian lens is fitted only '
            "at a subset of layers; other layers show &ldquo;not fitted&rdquo;.</p>"
        )

    return (
        f'<div style="font:13px system-ui,-apple-system,sans-serif">'
        f'{note}'
        f'<table style="border-collapse:collapse;width:100%">{header}'
        f"{''.join(body_rows)}</table></div>"
    )


def _render_prompt_tokens(token_strs: list[str], position: int) -> str:
    """Show the tokenized prompt with the read-out position highlighted."""
    spans = []
    for i, tok in enumerate(token_strs):
        disp = html.escape(repr(tok)[1:-1]) or "␣"
        if i == position:
            style = (
                "background:#f0a;color:#fff;font-weight:600;"
                "padding:1px 3px;border-radius:3px"
            )
        else:
            style = "background:#f3f3f3;padding:1px 3px;border-radius:3px;color:#333"
        spans.append(
            f'<span title="pos {i}" style="{style};margin:1px;'
            f'display:inline-block;font:12px ui-monospace,monospace">{disp}</span>'
        )
    return (
        '<div style="line-height:2;">'
        f'<div style="color:#666;font-size:12px;margin-bottom:4px">'
        f"Tokenized prompt ({len(token_strs)} tokens) — highlighted = read-out "
        f"position {position}:</div>{''.join(spans)}</div>"
    )


# --------------------------------------------------------------------------- #
# Event handlers
# --------------------------------------------------------------------------- #
def on_analyze(
    prompt: str,
    position: int,
    top_k: int,
    lenses: list[str],
):
    assert ENGINE is not None
    show_logit = "LogitLens" in lenses
    show_jlens = "Jacobian lens" in lenses
    if not (show_logit or show_jlens):
        show_logit = True

    result = ENGINE.analyze(
        prompt,
        position=int(position),
        top_k=int(top_k),
        use_logitlens=show_logit,
        use_jlens=show_jlens,
    )
    tables = _render_tables(result, show_logit, show_jlens)
    tokens = _render_prompt_tokens(result.token_strs, result.position)
    summary = (
        f"Model **{result.model_name}** · {result.n_layers} layers · "
        f"d_model {result.d_model} · device `{result.device}` · "
        f"read-out at position **{result.position}** "
        f"(token `{result.position_token!r}`) · Top-{result.top_k}"
    )
    return tables, tokens, summary


def on_position_slider_setup(prompt: str):
    """Update the position slider's range to match the token count."""
    assert ENGINE is not None
    n = len(ENGINE.tokenize_preview(prompt))
    # keep the last token selected by default
    return gr.update(minimum=0, maximum=max(0, n - 1), value=max(0, n - 1))


# --------------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------------- #
def build_ui() -> gr.Blocks:
    assert ENGINE is not None
    jlens_note = (
        "✅ Jacobian lens loaded"
        if ENGINE.lens is not None
        else "⚠️ Jacobian lens not fitted (LogitLens only)"
    )
    fitted = (
        f"fitted layers: {ENGINE.lens.source_layers}"
        if ENGINE.lens is not None
        else ""
    )

    with gr.Blocks(title="Layer Lens Visualizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            f"# 🔍 Layer Lens Visualizer\n"
            f"Per-layer **LogitLens** and **Jacobian lens** read-outs — the "
            f"vocabulary tokens each layer is disposed to emit, with "
            f"probabilities.\n\n"
            f"**{ENGINE.model_name}** · {jlens_note} {fitted}"
        )

        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    lines=4,
                    placeholder="Type any text…",
                )
            with gr.Column(scale=2):
                lenses = gr.CheckboxGroup(
                    choices=["LogitLens", "Jacobian lens"],
                    value=["LogitLens", "Jacobian lens"]
                    if ENGINE.lens is not None
                    else ["LogitLens"],
                    label="Lenses to show",
                )
                top_k = gr.Slider(
                    1, 25, value=8, step=1, label="Top-K tokens per layer"
                )
                position = gr.Slider(
                    0,
                    10,
                    value=0,
                    step=1,
                    label="Read-out position (token index; last token by default)",
                )
                run = gr.Button("Analyze", variant="primary")

        summary = gr.Markdown()
        tokens_view = gr.HTML()
        tables_view = gr.HTML()

        # Wire up: set position range when the prompt changes.
        prompt.change(on_position_slider_setup, inputs=prompt, outputs=position)

        run.click(
            on_analyze,
            inputs=[prompt, position, top_k, lenses],
            outputs=[tables_view, tokens_view, summary],
        )
        # On load, first snap the position slider to the last token, then run
        # the analysis so the user sees a correct read-out immediately.
        demo.load(
            on_position_slider_setup, inputs=prompt, outputs=position
        ).then(
            on_analyze,
            inputs=[prompt, position, top_k, lenses],
            outputs=[tables_view, tokens_view, summary],
        )

        gr.Markdown(
            "---\n"
            "**How to read this:** each row is a layer (top = deepest). "
            "The **LogitLens** decodes a layer's raw residual through the "
            "model's final norm + unembedding. The **Jacobian lens** first "
            "transports the residual into the final-layer basis with the fitted "
            "`J_l` matrix, which typically surfaces interpretable tokens several "
            "layers earlier than the LogitLens. The bottom row (pink) is the "
            "model's actual output distribution."
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--share", action="store_true", help="Gradio public link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip J-lens fit if not cached (LogitLens only)",
    )
    parser.add_argument(
        "--fit-layers",
        type=int,
        default=6,
        help="How many uniformly-spaced layers to fit the J-lens at",
    )
    parser.add_argument(
        "--fit-prompts", type=int, default=16, help="Prompts to fit the J-lens on"
    )
    args = parser.parse_args()

    global ENGINE
    ENGINE = LensEngine(args.model)
    ENGINE.load_or_fit_lens(
        source_layers=ENGINE.uniform_source_layers(args.fit_layers),
        n_prompts=args.fit_prompts,
        fit_if_missing=not args.no_fit,
    )

    demo = build_ui()
    demo.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
