"""CLI for fitting and caching a Jacobian lens."""

from __future__ import annotations

import argparse
import os

from lens_engine import LensEngine

from opdlens.utils.logging import console

DEFAULT_MODEL = os.environ.get("LENS_MODEL", "Qwen/Qwen3.5-2B")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--fit-layers",
        type=int,
        default=5,
        help="How many uniformly-spaced source layers to fit",
    )
    parser.add_argument(
        "--fit-prompts",
        type=int,
        default=6,
        help="How many fallback prompts to average over",
    )
    parser.add_argument(
        "--dim-batch",
        type=int,
        default=None,
        help="Output dimensions per backward pass",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=64,
        help="Prompt truncation length during fitting",
    )
    args = parser.parse_args()

    engine = LensEngine(args.model)
    source_layers = engine.uniform_source_layers(args.fit_layers)
    engine.fit_lens(
        source_layers=source_layers,
        n_prompts=args.fit_prompts,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
    )
    console.print(f"Saved lens to {engine.default_lens_path()}")


if __name__ == "__main__":
    main()
