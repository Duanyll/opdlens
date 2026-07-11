"""Offline OPRD bridge fit → ``bridge.pt`` (arm D artifact).

Fits a frozen affine map ``h_S ≈ (h_T - b_x) @ W + b_y`` (teacher basis → student
basis) by ridge least squares on paired teacher/student hidden states, pooled over
the supervised layer pairs. Run once per teacher/student pair + layer choice::

    uv run python -m opdlens.fit.fit_bridge --teacher Qwen/Qwen3-8B \
        --student Qwen/Qwen3-1.7B --prompts data/gsm8k_train.jsonl \
        --teacher-layers 8,12,16 --out bridge.pt
"""

from __future__ import annotations

import argparse

import torch

from ..data import read_jsonl
from ..losses import map_student_layer
from ..models import LanguageModel
from ..utils.logging import console, get_logger

logger = get_logger(__name__)


def _collect_pairs(
    teacher: LanguageModel,
    student: LanguageModel,
    prompts: list[str],
    pairs: list[tuple[int, int]],
    max_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    teacher_layers = tuple(lt for lt, _ in pairs)
    student_layers = tuple(ls for _, ls in pairs)
    teacher_rows: list[torch.Tensor] = []
    student_rows: list[torch.Tensor] = []
    total = 0
    for text in prompts:
        ids = (
            student.tokenizer(text, return_tensors="pt")
            .input_ids[0]
            .to(student.model.device)
        )
        with torch.no_grad():
            t_out = teacher.forward_capture(
                ids, layers=teacher_layers, need_logits=False
            )
            s_out = student.forward_capture(
                ids, layers=student_layers, need_logits=False
            )
        for lt, ls in pairs:
            teacher_rows.append(t_out.hidden[lt].float().cpu())
            student_rows.append(s_out.hidden[ls].float().cpu())
        total += ids.shape[0] * len(pairs)
        if total >= max_tokens:
            break
    return torch.cat(teacher_rows), torch.cat(student_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit an OPRD teacher→student bridge.")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--student", required=True)
    parser.add_argument(
        "--prompts", required=True, help="Calibration prompts (.jsonl)."
    )
    parser.add_argument("--prompt-key", default="question")
    parser.add_argument(
        "--teacher-layers", required=True, help="Comma-separated block indices."
    )
    parser.add_argument("--out", default="bridge.pt")
    parser.add_argument("--n-prompts", type=int, default=256)
    parser.add_argument("--max-tokens", type=int, default=50000)
    parser.add_argument("--ridge", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = LanguageModel(model_id=args.teacher, dtype="bf16")
    student = LanguageModel(model_id=args.student, dtype="bf16")
    teacher.load(device, trainable=False)
    student.load(device, trainable=False)

    teacher_layers = [int(x) for x in args.teacher_layers.split(",")]
    pairs = [
        (lt, map_student_layer(lt, student.num_layers, teacher.num_layers))
        for lt in teacher_layers
    ]
    rows = read_jsonl(args.prompts)[: args.n_prompts]
    prompts = [str(row[args.prompt_key]) for row in rows]
    logger.info("Collecting activations over %d prompts, pairs %s", len(prompts), pairs)

    teacher_h, student_h = _collect_pairs(
        teacher, student, prompts, pairs, args.max_tokens
    )
    logger.info("Collected %d paired token activations", teacher_h.shape[0])

    b_x = teacher_h.mean(dim=0)
    b_y = student_h.mean(dim=0)
    centered_t = teacher_h - b_x
    centered_s = student_h - b_y
    d_t = centered_t.shape[1]
    gram = centered_t.T @ centered_t + args.ridge * torch.eye(d_t)
    weight = torch.linalg.solve(gram, centered_t.T @ centered_s)  # [d_t, d_s]

    torch.save({"W": weight, "b_x": b_x, "b_y": b_y}, args.out)
    console.print(
        f"[green]Saved bridge to {args.out}[/green] "
        f"(W={tuple(weight.shape)}, ridge={args.ridge})"
    )


if __name__ == "__main__":
    main()
