"""Tier-aware fields on StageTrainingConfig — round-trip and validation."""
from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path

import pytest


def test_legacy_yaml_loads_with_tier_fields_none():
    """Stage 1/2 YAML (no tier fields) loads with tier fields = None and tier_grouped_sampling=False."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage2-test
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 1000
        batch_size: 2
        grad_accumulation_steps: 8
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 100
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        dataset_mix:
          - dataset: grandstaff_systems
            ratio: 1.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        cfg = load_stage_config(path)
    finally:
        path.unlink()

    assert cfg.tier_grouped_sampling is False
    assert cfg.b_cached is None
    assert cfg.b_live is None
    assert cfg.grad_accumulation_steps_cached is None
    assert cfg.grad_accumulation_steps_live is None
    assert cfg.cached_data_ratio is None
    assert cfg.cache_root is None
    assert cfg.cache_hash16 is None


def test_stage3_yaml_loads_tier_fields():
    """Stage 3 YAML with tier_grouped_sampling=true populates all 7 tier fields."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage3-test
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 7653
        batch_size: 1
        grad_accumulation_steps: 1
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 500
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        tier_grouped_sampling: true
        b_cached: 16
        b_live: 2
        grad_accumulation_steps_cached: 1
        grad_accumulation_steps_live: 8
        cached_data_ratio: 0.9
        cache_root: data/cache/encoder
        cache_hash16: ac8948ae4b5be3e9
        dataset_mix:
          - dataset: synthetic_systems
            ratio: 0.7778
            split: train
            required: true
          - dataset: grandstaff_systems
            ratio: 0.1111
            split: train
            required: true
          - dataset: primus_systems
            ratio: 0.1111
            split: train
            required: true
          - dataset: cameraprimus_systems
            ratio: 0.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        cfg = load_stage_config(path)
    finally:
        path.unlink()

    assert cfg.tier_grouped_sampling is True
    assert cfg.b_cached == 16
    assert cfg.b_live == 2
    assert cfg.grad_accumulation_steps_cached == 1
    assert cfg.grad_accumulation_steps_live == 8
    assert cfg.cached_data_ratio == pytest.approx(0.9)
    assert cfg.cache_root == "data/cache/encoder"
    assert cfg.cache_hash16 == "ac8948ae4b5be3e9"


def test_tier_grouped_true_requires_all_tier_fields():
    """tier_grouped_sampling=true with missing tier fields raises ValueError."""
    from src.train.train import load_stage_config

    yaml_text = textwrap.dedent("""
        stage_name: stage3-broken
        stage_b_encoder: radio_h
        epochs: 1
        effective_samples_per_epoch: 1000
        batch_size: 1
        grad_accumulation_steps: 1
        max_sequence_length: 512
        lr_dora: 0.0005
        lr_new_modules: 0.0003
        warmup_steps: 100
        schedule: cosine
        weight_decay: 0.01
        label_smoothing: 0.01
        contour_loss_weight: 0.01
        checkpoint_every_steps: 500
        validate_every_steps: 500
        tier_grouped_sampling: true
        b_cached: 16
        # missing: b_live, grad_accum_*, cached_data_ratio, cache_root, cache_hash16
        dataset_mix:
          - dataset: synthetic_systems
            ratio: 1.0
            split: train
            required: true
    """)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(yaml_text)
        path = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="tier_grouped_sampling=true requires"):
            load_stage_config(path)
    finally:
        path.unlink()
