"""Offline Jacobian-lens fit → ``lens.pt`` (arm C/E artifact).

Wraps ``opdlens.jlens.fit`` over a calibration prompt set. Run once per model
model + source-layer choice::

    uv run python -m opdlens.fit.fit_jlens --model Qwen/Qwen3-8B \
        --prompts data/gsm8k_train.jsonl --source-layers 8,12,16 --out lens.pt
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import torch

from ..data import read_hf, read_jsonl
from ..jlens import fit, from_hf
from ..models import LanguageModel
from ..utils.logging import console, get_logger

logger = get_logger(__name__)


def _calibration_texts(
    rows: Sequence[dict[str, Any]],
    tokenizer: Any,
    *,
    prompt_key: str,
    answer_key: str,
    include_completion: bool,
) -> list[str]:
    if not include_completion:
        return [str(row[prompt_key]) for row in rows]

    texts: list[str] = []
    for row in rows:
        messages = [
            {"role": "user", "content": str(row[prompt_key])},
            {"role": "assistant", "content": str(row[answer_key])},
        ]
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(messages, tokenize=False)
        texts.append(str(text))
    return texts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit a Jacobian lens for a teacher or initial student model."
    )
    parser.add_argument("--model", required=True, help="HF model id or local path.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompts", help="Calibration prompts (.jsonl).")
    source.add_argument("--hf-id", help="Hugging Face calibration dataset id.")
    parser.add_argument("--hf-name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument("--answer-key", default="answer")
    parser.add_argument(
        "--include-completion",
        action="store_true",
        help="Render user prompt plus gold assistant completion via the chat template.",
    )
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument(
        "--source-layers",
        required=True,
        help="Comma-separated block indices, e.g. 8,12,16.",
    )
    parser.add_argument("--out", default="lens.pt")
    parser.add_argument("--n-prompts", type=int, default=256)
    parser.add_argument("--max-seq-len", type=int, default=384)
    parser.add_argument("--dim-batch", type=int, default=16)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=8,
        help="Persist resumable fit state every N prompts.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lm = LanguageModel(model_id=args.model, dtype="bf16")
    lm.load(device, trainable=False)

    lens_model: Any = from_hf(lm.model, lm.tokenizer)
    if args.prompts:
        all_rows = read_jsonl(args.prompts)
    else:
        all_rows = list(read_hf(args.hf_id, args.split, name=args.hf_name))
    rows = all_rows[args.prompt_offset : args.prompt_offset + args.n_prompts]
    prompts = _calibration_texts(
        rows,
        lm.tokenizer,
        prompt_key=args.prompt_key,
        answer_key=args.answer_key,
        include_completion=args.include_completion,
    )
    source_layers = [int(x) for x in args.source_layers.split(",")]
    logger.info(
        "Fitting lens over %d prompts (offset=%d, completion=%s), layers %s",
        len(prompts),
        args.prompt_offset,
        args.include_completion,
        source_layers,
    )

    lens = fit(
        lens_model,
        prompts,
        source_layers=source_layers,
        max_seq_len=args.max_seq_len,
        dim_batch=args.dim_batch,
        checkpoint_path=args.out,
        checkpoint_every=args.checkpoint_every,
    )
    lens.save(args.out)
    console.print(f"[green]Saved Jacobian lens to {args.out}[/green] {lens!r}")


if __name__ == "__main__":
    main()
