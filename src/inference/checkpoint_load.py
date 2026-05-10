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
