"""Tests for the StageBDataset + custom sampler (issue #2 Tier 1 #1+#2).

TDD: These tests are written FIRST and define the required API.
All tests must pass on CPU-only hardware (no CUDA required).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train.train import (
    DatasetMix,
    StageTrainingConfig,
    StageBDataset,
    build_stage_b_sampler,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grouped_entries(tmp_path):
    """Build a tiny in-memory grouped_entries dict with synthetic PNG images."""
    from PIL import Image

    grouped = {}
    for ds_name, count in [("primus", 100), ("cameraprimus", 50), ("grandstaff", 50)]:
        rows = []
        for i in range(count):
            img_path = tmp_path / ds_name / f"{i}.png"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("L", (250, 250), color=128).save(img_path)
            rows.append(
                {
                    "sample_id": f"{ds_name}:s{i}",
                    "dataset": ds_name,
                    "split": "train",
                    "image_path": str(img_path),
                    "token_sequence": [
                        "<bos>",
                        "<staff_start>",
                        "clef-G2",
                        "<staff_end>",
                        "<eos>",
                    ],
                }
            )
        grouped[(ds_name, "train")] = rows
    return grouped


def _make_stage(mix, batch_size=2):
    return StageTrainingConfig(
        stage_name="test",
        epochs=1,
        effective_samples_per_epoch=100,
        batch_size=batch_size,
        grad_accumulation_steps=1,
        max_sequence_length=64,
        lr_dora=1e-4,
        lr_new_modules=1e-4,
        loraplus_lr_ratio=2.0,
        warmup_steps=0,
        schedule="linear",
        weight_decay=0.0,
        label_smoothing=0.0,
        contour_loss_weight=0.0,
        checkpoint_every_steps=1000,
        validate_every_steps=500,
        dataset_mix=mix,
    )


# ---------------------------------------------------------------------------
# Core API tests (RED: must fail before StageBDataset exists)
# ---------------------------------------------------------------------------


def test_dataset_returns_correct_keys(tmp_path):
    """StageBDataset.__getitem__ must return dict with the five required keys.

    augment=False: torchvision is unavailable in the CPU-only test venv.
    The production GPU host has torchvision; augmentation is tested separately.
    """
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    item = ds[0]
    for k in ("images", "decoder_inputs", "labels", "contour_targets", "content_widths"):
        assert k in item, f"missing key: {k}"


def test_dataset_len(tmp_path):
    """StageBDataset.__len__ should match total flattened entry count for the mix."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    assert len(ds) == 100  # primus has 100 entries


def test_dataset_multi_source_len(tmp_path):
    """With multiple datasets, __len__ covers all entries in the mix."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (
        DatasetMix(dataset="primus", ratio=0.5, split="train", required=True),
        DatasetMix(dataset="cameraprimus", ratio=0.5, split="train", required=True),
    )
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    assert len(ds) == 150  # 100 + 50


def test_dataset_item_tensor_shapes(tmp_path):
    """Each item tensor should have the expected shapes.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    item = ds[0]
    # image: (1, H, W) — grayscale channel first
    assert item["images"].shape == (1, 64, 128), f"Unexpected image shape: {item['images'].shape}"
    # decoder_inputs and labels: (max_sequence_length - 1,)
    assert item["decoder_inputs"].shape == (63,), f"Unexpected decoder_inputs shape: {item['decoder_inputs'].shape}"
    assert item["labels"].shape == (63,), f"Unexpected labels shape: {item['labels'].shape}"
    # contour_targets: scalar (0-dim) or (1,)
    assert item["contour_targets"].numel() == 1, f"Expected scalar, got {item['contour_targets']}"
    # content_widths: scalar
    assert item["content_widths"].numel() == 1, f"Expected scalar, got {item['content_widths']}"


def test_dataset_item_dtypes(tmp_path):
    """images should be float32; decoder_inputs/labels/contour_targets/content_widths should be long.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    item = ds[0]
    assert item["images"].dtype == torch.float32
    assert item["decoder_inputs"].dtype == torch.long
    assert item["labels"].dtype == torch.long
    assert item["contour_targets"].dtype == torch.long
    assert item["content_widths"].dtype == torch.long


def test_dataset_entries_attribute(tmp_path):
    """dataset.entries should be a flat list of dicts with at minimum 'dataset' key."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    assert hasattr(ds, "entries")
    assert all("dataset" in e for e in ds.entries)


# ---------------------------------------------------------------------------
# Sampler tests
# ---------------------------------------------------------------------------


def test_sampler_preserves_dataset_mix_ratios(tmp_path):
    """build_stage_b_sampler must yield the configured ratio within ±2% over 10000 draws."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (
        DatasetMix(dataset="primus", ratio=0.10, split="train", required=True),
        DatasetMix(dataset="cameraprimus", ratio=0.10, split="train", required=True),
        DatasetMix(dataset="grandstaff", ratio=0.80, split="train", required=True),
    )
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=10000, seed=0)

    counts: dict[str, int] = {"primus": 0, "cameraprimus": 0, "grandstaff": 0}
    for idx in sampler:
        ds_name = ds.entries[idx]["dataset"]
        counts[ds_name] += 1

    total = sum(counts.values())
    assert total == 10000

    expected = {"primus": 0.10, "cameraprimus": 0.10, "grandstaff": 0.80}
    for k, ev in expected.items():
        realized = counts[k] / total
        assert abs(realized - ev) <= 0.02, (
            f"{k}: expected {ev:.2f}, realized {realized:.4f}"
        )


def test_sampler_with_zero_ratio_dataset_drops_it(tmp_path):
    """A dataset with ratio=0 should never appear in sampled indices."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (
        DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),
        DatasetMix(dataset="cameraprimus", ratio=0.0, split="train", required=False),
    )
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=1000, seed=0)
    seen_datasets = {ds.entries[idx]["dataset"] for idx in sampler}
    assert "cameraprimus" not in seen_datasets
    assert "primus" in seen_datasets


def test_sampler_total_samples(tmp_path):
    """Sampler must yield exactly total_samples indices."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    for n in (1, 50, 500):
        sampler = build_stage_b_sampler(stage, ds, total_samples=n, seed=42)
        assert len(list(sampler)) == n, f"Expected {n} samples"


def test_sampler_seed_reproducibility(tmp_path):
    """Same seed must produce the same index sequence."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (
        DatasetMix(dataset="primus", ratio=0.5, split="train", required=True),
        DatasetMix(dataset="cameraprimus", ratio=0.5, split="train", required=True),
    )
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    s1 = list(build_stage_b_sampler(stage, ds, total_samples=200, seed=7))
    s2 = list(build_stage_b_sampler(stage, ds, total_samples=200, seed=7))
    assert s1 == s2, "Same seed must give identical sequences"


def test_sampler_different_seeds_differ(tmp_path):
    """Different seeds should (very likely) produce different sequences."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    s1 = list(build_stage_b_sampler(stage, ds, total_samples=200, seed=1))
    s2 = list(build_stage_b_sampler(stage, ds, total_samples=200, seed=2))
    assert s1 != s2, "Different seeds should produce different sequences"


# ---------------------------------------------------------------------------
# DataLoader + collate tests
# ---------------------------------------------------------------------------


def test_dataloader_collate_produces_expected_tensor_shapes(tmp_path):
    """Standard DataLoader with collate_fn should produce batched tensors of the right shape.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=10, seed=0)

    loader = DataLoader(
        ds,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        collate_fn=ds.collate_fn,
    )
    batch = next(iter(loader))
    assert batch["images"].shape[0] == 2, "Batch size should be 2"
    assert batch["images"].shape[1] == 1, "Should have 1 grayscale channel"
    assert batch["images"].shape[2] == 64, "Image height should be 64"
    assert batch["images"].shape[3] == 128, "Image width should be 128"


def test_dataloader_collate_all_keys_present(tmp_path):
    """Collated batch must contain all five keys.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=10, seed=0)
    loader = DataLoader(ds, batch_size=2, sampler=sampler, num_workers=0, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    for k in ("images", "decoder_inputs", "labels", "contour_targets", "content_widths"):
        assert k in batch, f"Batch missing key: {k}"


def test_dataloader_collate_batch_tensor_shapes(tmp_path):
    """All tensors in a collated batch should have correct shapes.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=6, seed=0)
    loader = DataLoader(ds, batch_size=3, sampler=sampler, num_workers=0, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    B = 3
    assert batch["images"].shape == (B, 1, 64, 128)
    assert batch["decoder_inputs"].shape == (B, 63)
    assert batch["labels"].shape == (B, 63)
    assert batch["contour_targets"].shape == (B,)
    assert batch["content_widths"].shape == (B,)


def test_persistent_workers_zero_workers_smoke(tmp_path):
    """With num_workers=0 (Windows fallback), DataLoader works without errors.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=10, seed=0)
    loader = DataLoader(
        ds, batch_size=2, sampler=sampler, num_workers=0, collate_fn=ds.collate_fn
    )
    # Two iterations exercise the iterator state machine
    batches = []
    for batch in loader:
        assert "images" in batch
        batches.append(batch)
    assert len(batches) == 5  # 10 samples / batch_size=2


def test_dataloader_multi_epoch_iteration(tmp_path):
    """Iterating the DataLoader twice (simulating two epochs) should produce valid batches.

    augment=False to avoid torchvision dependency in CPU-only test venv.
    """
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )

    for epoch in range(2):
        sampler = build_stage_b_sampler(stage, ds, total_samples=10, seed=epoch)
        loader = DataLoader(
            ds, batch_size=2, sampler=sampler, num_workers=0, collate_fn=ds.collate_fn
        )
        count = 0
        for batch in loader:
            assert "images" in batch
            count += 1
        assert count == 5, f"Epoch {epoch}: expected 5 batches, got {count}"


# ---------------------------------------------------------------------------
# Collate_fn attribute test
# ---------------------------------------------------------------------------


def test_dataset_has_collate_fn(tmp_path):
    """StageBDataset must expose a collate_fn callable."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    assert callable(ds.collate_fn), "StageBDataset must have a callable collate_fn"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_dataset_with_only_one_entry(tmp_path):
    """Dataset with a single entry must still be iterable and return correct structure."""
    from PIL import Image

    img_path = tmp_path / "tiny" / "0.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (200, 200), color=200).save(img_path)

    grouped = {
        ("tiny", "train"): [
            {
                "sample_id": "tiny:0",
                "dataset": "tiny",
                "split": "train",
                "image_path": str(img_path),
                "token_sequence": ["<bos>", "clef-G2", "<eos>"],
            }
        ]
    }
    mix = (DatasetMix(dataset="tiny", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=32,
        image_width=64,
        max_sequence_length=16,
        augment=False,
    )
    assert len(ds) == 1
    item = ds[0]
    assert "images" in item


def test_sampler_single_sample_no_crash(tmp_path):
    """Sampler with total_samples=1 should not crash."""
    from PIL import Image

    img_path = tmp_path / "tiny" / "0.png"
    img_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", (200, 200), color=200).save(img_path)

    grouped = {
        ("tiny", "train"): [
            {
                "sample_id": "tiny:0",
                "dataset": "tiny",
                "split": "train",
                "image_path": str(img_path),
                "token_sequence": ["<bos>", "clef-G2", "<eos>"],
            }
        ]
    }
    mix = (DatasetMix(dataset="tiny", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=32,
        image_width=64,
        max_sequence_length=16,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=1, seed=0)
    indices = list(sampler)
    assert len(indices) == 1
    assert indices[0] == 0


def test_augmentation_skipped_when_disabled(tmp_path):
    """With augment=False, images should be deterministic across two identical calls."""
    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
        augment=False,
    )
    item1 = ds[0]
    item2 = ds[0]
    assert torch.allclose(item1["images"], item2["images"]), (
        "Without augmentation, same item should return identical images"
    )


# ---------------------------------------------------------------------------
# Non-blocking H2D transfer (smoke test — cannot test overlap on CPU)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pin_memory_non_blocking_h2d(tmp_path):
    """On CUDA, pin_memory=True + non_blocking=True should work without error."""
    from torch.utils.data import DataLoader

    grouped = _make_grouped_entries(tmp_path)
    mix = (DatasetMix(dataset="primus", ratio=1.0, split="train", required=True),)
    stage = _make_stage(mix)
    ds = StageBDataset(
        stage,
        grouped,
        project_root=tmp_path,
        image_height=64,
        image_width=128,
        max_sequence_length=64,
    )
    sampler = build_stage_b_sampler(stage, ds, total_samples=4, seed=0)
    loader = DataLoader(
        ds,
        batch_size=2,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
        collate_fn=ds.collate_fn,
    )
    device = torch.device("cuda")
    batch = next(iter(loader))
    images = batch["images"].to(device, non_blocking=True)
    torch.cuda.synchronize()
    assert images.device.type == "cuda"
