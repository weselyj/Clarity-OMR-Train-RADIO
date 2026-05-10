"""Single-source-of-truth loader for Stage B checkpoints used by inference.

Factored from src/eval/evaluate_stage_b_checkpoint.py:285-345 — keeps the
8-step Stage B inference loading sequence in one place so callers
(SystemInferencePipeline, debugging notebooks, future refactors) cannot
get the order or arguments wrong.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn  # noqa: F401
    from src.tokenizer.vocab import OMRVocabulary
    from src.train.model_factory import ModelFactoryConfig


@dataclass
class StageBInferenceBundle:
    """Fully-loaded, eval-mode, inference-ready Stage B model bundle."""
    model: "nn.Module"
    decode_model: "nn.Module"
    vocab: "OMRVocabulary"
    token_to_idx: Dict[str, int]
    use_fp16: bool
    factory_cfg: "ModelFactoryConfig"


import torch  # noqa: E402

from src.checkpoint_io import load_stage_b_checkpoint  # noqa: E402
from src.inference.decoder_runtime import _prepare_model_for_inference  # noqa: E402
from src.tokenizer.vocab import build_default_vocabulary  # noqa: E402
from src.train.model_factory import (  # noqa: E402
    ModelFactoryConfig,
    build_stage_b_components,
    model_factory_config_from_checkpoint_payload,
)


def load_stage_b_for_inference(
    checkpoint_path: Path,
    device,
    *,
    use_fp16: bool = False,
    quantize: bool = False,
) -> StageBInferenceBundle:
    """Build vocab → torch.load payload → infer ModelFactoryConfig →
    build_stage_b_components → load_stage_b_checkpoint → eval() →
    _prepare_model_for_inference.

    Returns the fully-prepared inference-ready bundle.
    """
    vocab = build_default_vocabulary()
    payload = torch.load(str(checkpoint_path), map_location=device)
    fallback = ModelFactoryConfig(stage_b_vocab_size=vocab.size)
    factory_cfg = model_factory_config_from_checkpoint_payload(
        payload, vocab_size=vocab.size, fallback=fallback,
    )
    components = build_stage_b_components(factory_cfg)
    model = components["model"]
    ckpt_result = load_stage_b_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        dora_config=components.get("dora_config"),
        min_coverage=0.50,
    )
    model = ckpt_result["_model"]
    model.eval()
    decode_model, use_fp16_resolved = _prepare_model_for_inference(
        model, device, use_fp16=use_fp16, quantize=quantize,
    )
    token_to_idx = {token: idx for idx, token in enumerate(vocab.tokens)}
    return StageBInferenceBundle(
        model=model,
        decode_model=decode_model,
        vocab=vocab,
        token_to_idx=token_to_idx,
        use_fp16=use_fp16_resolved,
        factory_cfg=factory_cfg,
    )
