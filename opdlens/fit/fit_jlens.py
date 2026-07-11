"""Offline Jacobian-lens fit → ``lens.pt`` (arm C/E artifact).

Wraps ``opdlens.jlens.fit`` over a calibration prompt set. Run once per model
model + source-layer choice::

    uv run python -m opdlens.fit.fit_jlens --model Qwen/Qwen3-8B \
        --prompts data/gsm8k_train.jsonl --source-layers 8,12,16 --out lens.pt
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from ..data import read_jsonl
from ..jlens import fit, from_hf
from ..models import LanguageModel
from ..utils.logging import console, get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a Jacobian lens for a teacher or initial student model."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path.")
    parser.add_argument(
        "--prompts", required=True, help="Calibration prompts (.jsonl)."
    )
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument(
        "--source-layers",
        required=True,
        help="Comma-separated block indices, e.g. 8,12,16.",
    )
    parser.add_argument("--out", default="lens.pt")
    parser.add_argument("--n-prompts", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--dim-batch", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = LanguageModel(model_id=args.model, dtype="bf16")
    lm.load(device, trainable=False)

    lens_model: Any = from_hf(lm.model, lm.tokenizer)
    rows = read_jsonl(args.prompts)[: args.n_prompts]
    prompts = [str(row[args.prompt_key]) for row in rows]
    source_layers = [int(x) for x in args.source_layers.split(",")]
    logger.info("Fitting lens over %d prompts, layers %s", len(prompts), source_layers)

    lens = fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        max_seq_len=args.max_seq_len,
        dim_batch=args.dim_batch,
        checkpoint_path=args.out,
    )
    lens.save(args.out)
    console.print(f"[green]Saved Jacobian lens to {args.out}[/green] {lens!r}")


if __name__ == "__main__":
    main()
