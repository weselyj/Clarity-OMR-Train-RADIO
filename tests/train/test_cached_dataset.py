"""Tests for the cached-path extension to StageBDataset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


from src.train.train import _CACHED_DATASETS as CACHED_DATASETS


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e) + "\n")


def _make_fake_image(tmp_path: Path, name: str = "img.png") -> Path:
    """Create a tiny white PNG for use as a live-tier image."""
    import numpy as np
    from PIL import Image
    img = Image.fromarray((255 * torch.ones(32, 64, dtype=torch.uint8).numpy()), mode="L")
    p = tmp_path / name
    img.save(p)
    return p


def _write_cache_entries(cache_root: Path, hash16: str, entries: list[dict]) -> None:
    """Pre-populate cache with fake tensors for testing."""
    from src.data.encoder_cache import _sanitize_sample_key, write_cache_entry
    for e in entries:
        ds = e["dataset"]
        key = _sanitize_sample_key(e["sample_id"])
        t = torch.randn(8, 1280, dtype=torch.bfloat16)
        write_cache_entry(cache_root, hash16, ds, key, t, h16=2, w16=4)


def _make_minimal_stage_config(datasets: list[str]):
    """Build a minimal StageTrainingConfig-like namespace for StageBDataset."""
    import types
    stage = types.SimpleNamespace()
    stage.dataset_mix = [
        types.SimpleNamespace(dataset=ds, split="train", ratio=1.0 / len(datasets))
        for ds in datasets
    ]
    return stage


def test_cached_getitem_returns_tier_cached(tmp_path: Path) -> None:
    """__getitem__ for a cached-tier entry must return dict with 'tier'='cached'."""
    from src.train.train import StageBDataset
    from src.data.encoder_cache import _sanitize_sample_key

    hash16 = "test0000test0000"
    cache_root = tmp_path / "cache"

    entries = [
        {"sample_id": "synthetic_systems:page001__sys00", "dataset": "synthetic_systems",
         "split": "train", "image_path": None, "token_sequence": ["<bos>", "<eos>"]}
    ]
    _write_cache_entries(cache_root, hash16, entries)

    stage = _make_minimal_stage_config(["synthetic_systems"])
    grouped = {("synthetic_systems", "train"): entries}
    ds = StageBDataset(
        stage=stage,
        grouped_entries=grouped,
        split="train",
        project_root=tmp_path,
        cache_root=cache_root,
        cache_hash16=hash16,
    )
    sample = ds[0]
    assert sample["tier"] == "cached"
    assert "encoder_hidden" in sample
    assert sample["encoder_hidden"].shape == (8, 1280)
    assert sample["encoder_hidden"].dtype == torch.bfloat16
    assert "images" not in sample


def test_cached_getitem_raises_on_missing_cache(tmp_path: Path) -> None:
    """__getitem__ for cached-tier entry with no cache file must raise CacheMiss."""
    from src.train.train import StageBDataset
    from src.data.encoder_cache import CacheMiss

    hash16 = "test0000test0000"
    cache_root = tmp_path / "cache"
    # Do NOT write any cache entries

    entries = [
        {"sample_id": "synthetic_systems:page001__sys00", "dataset": "synthetic_systems",
         "split": "train", "image_path": None, "token_sequence": ["<bos>", "<eos>"]}
    ]
    stage = _make_minimal_stage_config(["synthetic_systems"])
    grouped = {("synthetic_systems", "train"): entries}
    ds = StageBDataset(
        stage=stage,
        grouped_entries=grouped,
        split="train",
        project_root=tmp_path,
        cache_root=cache_root,
        cache_hash16=hash16,
    )
    with pytest.raises(CacheMiss):
        _ = ds[0]


def test_live_getitem_returns_tier_live(tmp_path: Path) -> None:
    """__getitem__ for a live-tier entry must return dict with 'tier'='live'."""
    from src.train.train import StageBDataset

    img_path = _make_fake_image(tmp_path, "live_img.png")
    entries = [
        {"sample_id": "cameraprimus_systems:sample001", "dataset": "cameraprimus_systems",
         "split": "train", "image_path": str(img_path.relative_to(tmp_path)),
         "token_sequence": ["<bos>", "<eos>"]}
    ]
    stage = _make_minimal_stage_config(["cameraprimus_systems"])
    grouped = {("cameraprimus_systems", "train"): entries}
    ds = StageBDataset(
        stage=stage,
        grouped_entries=grouped,
        split="train",
        project_root=tmp_path,
        cache_root=None,
        cache_hash16=None,
    )
    sample = ds[0]
    assert sample["tier"] == "live"
    assert "images" in sample
    assert "encoder_hidden" not in sample


def test_collate_fn_cached_batches_stack_encoder_hidden(tmp_path: Path) -> None:
    """collate_fn on all-cached samples must stack encoder_hidden tensors."""
    from src.train.train import StageBDataset

    samples = [
        {"tier": "cached", "encoder_hidden": torch.randn(8, 1280, dtype=torch.bfloat16),
         "_h16": 2, "_w16": 4,
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(0, dtype=torch.long)},
        {"tier": "cached", "encoder_hidden": torch.randn(8, 1280, dtype=torch.bfloat16),
         "_h16": 2, "_w16": 4,
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(1, dtype=torch.long)},
    ]
    batch = StageBDataset.collate_fn(samples)
    assert batch["tier"] == "cached"
    assert batch["encoder_hidden"].shape == (2, 8, 1280)
    assert batch["_h16"] == 2
    assert batch["_w16"] == 4
    assert "images" not in batch


def test_collate_fn_live_batches_stack_images(tmp_path: Path) -> None:
    """collate_fn on all-live samples must stack image tensors."""
    from src.train.train import StageBDataset

    samples = [
        {"tier": "live", "images": torch.rand(1, 32, 64),
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(0, dtype=torch.long),
         "content_widths": torch.tensor(64, dtype=torch.long)},
        {"tier": "live", "images": torch.rand(1, 32, 64),
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(1, dtype=torch.long),
         "content_widths": torch.tensor(64, dtype=torch.long)},
    ]
    batch = StageBDataset.collate_fn(samples)
    assert batch["tier"] == "live"
    assert batch["images"].shape == (2, 1, 32, 64)
    assert "encoder_hidden" not in batch


def test_collate_fn_raises_on_mixed_tiers() -> None:
    """collate_fn must raise ValueError if samples mix cached and live tiers."""
    from src.train.train import StageBDataset

    samples = [
        {"tier": "cached", "encoder_hidden": torch.randn(8, 1280, dtype=torch.bfloat16),
         "_h16": 2, "_w16": 4,
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(0, dtype=torch.long)},
        {"tier": "live", "images": torch.rand(1, 32, 64),
         "decoder_inputs": torch.zeros(10, dtype=torch.long),
         "labels": torch.zeros(10, dtype=torch.long),
         "contour_targets": torch.tensor(0, dtype=torch.long),
         "content_widths": torch.tensor(64, dtype=torch.long)},
    ]
    with pytest.raises(ValueError, match="[Mm]ixed"):
        StageBDataset.collate_fn(samples)
