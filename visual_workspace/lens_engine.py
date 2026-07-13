"""Model + lens utilities for the Gradio layer visualizer."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class TokenProb:
    token_id: int
    token_str: str
    prob: float


@dataclass(frozen=True)
class LayerReadout:
    layer: int
    top: list[TokenProb]
    available: bool


@dataclass(frozen=True)
class AnalysisResult:
    model_name: str
    n_layers: int
    d_model: int
    device: str
    position: int
    position_token: str
    top_k: int
    token_strs: list[str]
    logitlens: list[LayerReadout]
    jlens: list[LayerReadout]
    jlens_available: bool


class LensEngine:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = pick_device()
        self.hf_model, self.tokenizer = self._load_hf_model(model_name, self.device)
        self.model = jlens.from_hf(self.hf_model, self.tokenizer)
        self.lens: JacobianLens | None = None

    @staticmethod
    def _load_hf_model(
        model_name: str, device: torch.device
    ) -> tuple[torch.nn.Module, transformers.PreTrainedTokenizerBase]:
        logger.info("Loading model %s on %s", model_name, device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
        load_kwargs = {"trust_remote_code": True, "torch_dtype": dtype}
        try:
            hf_model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        except Exception as exc:
            logger.warning(
                "AutoModelForCausalLM failed for %s (%s); falling back to AutoModel",
                model_name,
                exc,
            )
            hf_model = AutoModel.from_pretrained(model_name, **load_kwargs)
        hf_model.to(device)
        hf_model.eval()
        return hf_model, tokenizer

    def default_lens_path(self) -> Path:
        safe_name = self.model_name.replace("/", "__")
        return Path(__file__).with_name("lenses") / f"{safe_name}__jlens.pt"

    def load_or_fit_lens(
        self,
        *,
        source_layers: list[int] | None = None,
        n_prompts: int = 16,
        fit_if_missing: bool = True,
    ) -> JacobianLens | None:
        lens_path = self.default_lens_path()
        if lens_path.exists():
            logger.info("Loading cached lens from %s", lens_path)
            self.lens = JacobianLens.load(str(lens_path))
            return self.lens
        if not fit_if_missing:
            logger.info("No cached lens at %s; continuing with LogitLens only", lens_path)
            self.lens = None
            return None
        return self.fit_lens(
            source_layers=source_layers or self.uniform_source_layers(5),
            n_prompts=n_prompts,
            lens_path=lens_path,
        )

    def uniform_source_layers(self, n_layers_to_fit: int) -> list[int]:
        n_layers_to_fit = max(1, min(n_layers_to_fit, self.model.n_layers - 1))
        candidates = torch.linspace(
            0, self.model.n_layers - 2, steps=n_layers_to_fit + 2
        )[1:-1]
        layers = sorted({int(round(x.item())) for x in candidates})
        return [layer for layer in layers if 0 <= layer < self.model.n_layers - 1]

    def fit_lens(
        self,
        *,
        source_layers: list[int],
        n_prompts: int,
        lens_path: str | Path | None = None,
        dim_batch: int | None = None,
        max_seq_len: int = 64,
    ) -> JacobianLens:
        prompts = self._fallback_prompts(n_prompts)
        out_path = Path(lens_path) if lens_path is not None else self.default_lens_path()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_path.with_suffix(".fit.ckpt.pt")

        if dim_batch is None:
            dim_batch = 16 if self.device.type in {"cuda", "mps"} else 4

        logger.info(
            "Fitting J-lens: layers=%s prompts=%d dim_batch=%d max_seq_len=%d",
            source_layers,
            len(prompts),
            dim_batch,
            max_seq_len,
        )
        self.lens = jlens.fit(
            self.model,
            prompts,
            source_layers=source_layers,
            dim_batch=dim_batch,
            max_seq_len=max_seq_len,
            checkpoint_path=str(checkpoint_path),
            checkpoint_every=1,
            resume=True,
        )
        self.lens.save(str(out_path))
        logger.info("Saved fitted lens to %s", out_path)
        return self.lens

    def _fallback_prompts(self, n_prompts: int) -> list[str]:
        base = [
            "Fact: The capital of Japan is Tokyo.\nFact: The currency used in the country shaped like a boot is",
            "The old painting hung crookedly on the wall while the evening light faded through the window.",
            "Write a concise review of this Python function and point out any obvious bug you notice first.",
            "A short story begins with a traveler arriving at a quiet seaside town just before dawn.",
            "The experiment compares baseline decoding, logit lens, Jacobian lens, and hidden-state alignment.",
            "List three reasons why residual representations can become easier to decode in deeper layers.",
            "The recipe calls for flour, butter, sugar, eggs, and a pinch of salt before baking.",
            "A student solves a geometry problem by drawing an auxiliary line and checking similar triangles.",
            "The city council debated whether to expand the subway line to the growing northern district.",
            "In the lab notebook, the researcher marked the unexpected measurement and repeated the calibration.",
            "Translate the sentence into Chinese and preserve the tone of a formal academic abstract.",
            "The detective noticed that the letter on the desk had been folded twice instead of three times.",
            "A musician practices the same passage slowly until the phrasing becomes natural and controlled.",
            "The model should explain its reasoning carefully without jumping to the final conclusion too early.",
            "An astronomer observed a faint signal near the edge of the image and scheduled a longer exposure.",
            "The conference paper argues that intermediate representations can reveal planning before final token emission.",
        ]
        if n_prompts <= len(base):
            return base[:n_prompts]
        prompts = list(base)
        for idx in range(len(base), n_prompts):
            prompts.append(base[idx % len(base)] + f"\nExample variant {idx}.")
        return prompts

    @torch.no_grad()
    def analyze(
        self,
        prompt: str,
        *,
        position: int = -1,
        top_k: int = 8,
        use_logitlens: bool = True,
        use_jlens: bool = True,
        max_seq_len: int = 512,
    ) -> AnalysisResult:
        input_ids = self.model.encode(prompt, max_length=max_seq_len)
        token_ids = input_ids[0].tolist()
        token_strs = [self.tokenizer.decode([token_id]) for token_id in token_ids]
        seq_len = len(token_ids)
        if seq_len == 0:
            raise ValueError("prompt produced zero tokens")

        if position < 0:
            position += seq_len
        position = max(0, min(position, seq_len - 1))

        with ActivationRecorder(self.model.layers, at=range(self.model.n_layers)) as rec:
            self.model.forward(input_ids)
            activations = {
                layer: rec.activations[layer][0, position].detach()
                for layer in range(self.model.n_layers)
            }

        logit_rows: list[LayerReadout] = []
        if use_logitlens:
            for layer in range(self.model.n_layers):
                logits = self.model.unembed(activations[layer].unsqueeze(0))[0].float().cpu()
                logit_rows.append(
                    LayerReadout(layer=layer, top=self._topk(logits, top_k), available=True)
                )

        jlens_rows: list[LayerReadout] = []
        jlens_available = self.lens is not None
        if use_jlens:
            fitted = set(self.lens.source_layers) if self.lens is not None else set()
            for layer in range(self.model.n_layers):
                if self.lens is None or layer not in fitted:
                    jlens_rows.append(LayerReadout(layer=layer, top=[], available=False))
                    continue
                transported = self.lens.transport(activations[layer].float(), layer)
                logits = self.model.unembed(transported.unsqueeze(0))[0].float().cpu()
                jlens_rows.append(
                    LayerReadout(layer=layer, top=self._topk(logits, top_k), available=True)
                )

        return AnalysisResult(
            model_name=self.model_name,
            n_layers=self.model.n_layers,
            d_model=self.model.d_model,
            device=str(self.device),
            position=position,
            position_token=token_strs[position],
            top_k=top_k,
            token_strs=token_strs,
            logitlens=logit_rows,
            jlens=jlens_rows,
            jlens_available=jlens_available,
        )

    def _topk(self, logits: torch.Tensor, top_k: int) -> list[TokenProb]:
        probs = torch.softmax(logits.float(), dim=-1)
        values, indices = torch.topk(probs, k=min(top_k, probs.shape[-1]))
        return [
            TokenProb(
                token_id=int(token_id),
                token_str=self.tokenizer.decode([int(token_id)]),
                prob=float(prob),
            )
            for prob, token_id in zip(values.tolist(), indices.tolist(), strict=True)
        ]

    def tokenize_preview(self, prompt: str, *, max_seq_len: int = 512) -> list[str]:
        input_ids = self.model.encode(prompt, max_length=max_seq_len)[0].tolist()
        return [self.tokenizer.decode([token_id]) for token_id in input_ids]
