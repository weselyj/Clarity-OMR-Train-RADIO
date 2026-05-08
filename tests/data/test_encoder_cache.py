"""Unit tests for src/data/encoder_cache.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Hash determinism tests
# ---------------------------------------------------------------------------

def test_compute_cache_hash_deterministic(tmp_path: Path) -> None:
    """Same inputs → same 16-char hex string, called twice."""
    from src.data.encoder_cache import compute_cache_hash

    weights_file = tmp_path / "weights.pt"
    weights_file.write_bytes(b"fake-weights-content")
    preproc_cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
                   "normalize_std": None, "pad_color": 1.0}
    arch = "c-radio_v4-h"

    h1 = compute_cache_hash(weights_file, preproc_cfg, arch, git_head_sha="abc123")
    h2 = compute_cache_hash(weights_file, preproc_cfg, arch, git_head_sha="abc123")
    assert h1 == h2
    assert len(h1) == 16
    assert all(c in "0123456789abcdef" for c in h1)


def test_compute_cache_hash_changes_on_weights(tmp_path: Path) -> None:
    from src.data.encoder_cache import compute_cache_hash

    w1 = tmp_path / "w1.pt"
    w2 = tmp_path / "w2.pt"
    w1.write_bytes(b"weights-v1")
    w2.write_bytes(b"weights-v2")
    cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
           "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(w1, cfg, "c-radio_v4-h", git_head_sha="abc")
    h2 = compute_cache_hash(w2, cfg, "c-radio_v4-h", git_head_sha="abc")
    assert h1 != h2


def test_compute_cache_hash_changes_on_preproc_cfg(tmp_path: Path) -> None:
    from src.data.encoder_cache import compute_cache_hash

    wf = tmp_path / "w.pt"
    wf.write_bytes(b"weights")
    cfg_a = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
             "normalize_std": None, "pad_color": 1.0}
    cfg_b = {"image_height": 300, "image_width": 2500, "normalize_mean": None,
             "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(wf, cfg_a, "c-radio_v4-h", git_head_sha="abc")
    h2 = compute_cache_hash(wf, cfg_b, "c-radio_v4-h", git_head_sha="abc")
    assert h1 != h2


def test_compute_cache_hash_changes_on_arch_version(tmp_path: Path) -> None:
    from src.data.encoder_cache import compute_cache_hash

    wf = tmp_path / "w.pt"
    wf.write_bytes(b"weights")
    cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
           "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha="abc")
    h2 = compute_cache_hash(wf, cfg, "c-radio_v4-b", git_head_sha="abc")
    assert h1 != h2


def test_compute_cache_hash_changes_on_git_sha(tmp_path: Path) -> None:
    from src.data.encoder_cache import compute_cache_hash

    wf = tmp_path / "w.pt"
    wf.write_bytes(b"weights")
    cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
           "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha="abc123")
    h2 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha="def456")
    assert h1 != h2


def test_compute_cache_hash_ignore_git_sha_is_stable(tmp_path: Path) -> None:
    """When git_head_sha=None (--ignore-git-sha mode) hash is stable across SHA values."""
    from src.data.encoder_cache import compute_cache_hash

    wf = tmp_path / "w.pt"
    wf.write_bytes(b"weights")
    cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
           "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha=None)
    h2 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha=None)
    assert h1 == h2


def test_compute_cache_hash_whitespace_in_cfg_doesnt_change_hash(tmp_path: Path) -> None:
    """Hashing the parsed dict (not raw YAML bytes) means whitespace is irrelevant."""
    from src.data.encoder_cache import compute_cache_hash

    wf = tmp_path / "w.pt"
    wf.write_bytes(b"weights")
    # Same logical dict, different whitespace if it came from YAML
    cfg = {"image_height": 250, "image_width": 2500, "normalize_mean": None,
           "normalize_std": None, "pad_color": 1.0}
    h1 = compute_cache_hash(wf, cfg, "c-radio_v4-h", git_head_sha="abc")
    # Reorder keys: sorted-key JSON should still produce same hash
    cfg2 = {"pad_color": 1.0, "image_height": 250, "normalize_mean": None,
            "normalize_std": None, "image_width": 2500}
    h2 = compute_cache_hash(wf, cfg2, "c-radio_v4-h", git_head_sha="abc")
    assert h1 == h2


# ---------------------------------------------------------------------------
# Sample key sanitization tests
# ---------------------------------------------------------------------------

def test_sanitize_sample_key_strips_dataset_prefix() -> None:
    from src.data.encoder_cache import _sanitize_sample_key
    assert _sanitize_sample_key("synthetic_systems:Abbott__p001__sys00") == "Abbott__p001__sys00"


def test_sanitize_sample_key_replaces_slash() -> None:
    from src.data.encoder_cache import _sanitize_sample_key
    assert _sanitize_sample_key("primus:dir/sub/file") == "dir__sub__file"


def test_sanitize_sample_key_replaces_backslash() -> None:
    from src.data.encoder_cache import _sanitize_sample_key
    assert _sanitize_sample_key("grandstaff_systems:dir\\sub\\file") == "dir__sub__file"


def test_sanitize_sample_key_no_op_on_clean_key() -> None:
    from src.data.encoder_cache import _sanitize_sample_key
    assert _sanitize_sample_key("Abbott__p001__sys00") == "Abbott__p001__sys00"


def test_sanitize_sample_key_colon_in_body_replaced() -> None:
    """A colon that is NOT the dataset-prefix separator (e.g. after the first colon) is replaced."""
    from src.data.encoder_cache import _sanitize_sample_key
    # "ds:body:extra" → strip "ds:", then replace ":" in "body:extra"
    assert _sanitize_sample_key("ds:body:extra") == "body__extra"


# ---------------------------------------------------------------------------
# Write / read / hit-miss / collision tests
# ---------------------------------------------------------------------------

def _make_fake_tensor(seq_tokens: int = 20, hidden_dim: int = 1280) -> "torch.Tensor":
    import torch
    return torch.randn(seq_tokens, hidden_dim, dtype=torch.bfloat16)


def test_cache_entry_exists_false_before_write(tmp_path: Path) -> None:
    from src.data.encoder_cache import cache_entry_exists
    assert not cache_entry_exists(tmp_path, "abcd1234abcd1234", "synthetic_systems", "sample_001")


def test_write_then_exists_returns_true(tmp_path: Path) -> None:
    from src.data.encoder_cache import cache_entry_exists, write_cache_entry
    t = _make_fake_tensor()
    write_cache_entry(tmp_path, "abcd1234abcd1234", "synthetic_systems", "sample_001", t, h16=2, w16=10)
    assert cache_entry_exists(tmp_path, "abcd1234abcd1234", "synthetic_systems", "sample_001")


def test_read_returns_correct_tensor(tmp_path: Path) -> None:
    import torch
    from src.data.encoder_cache import read_cache_entry, write_cache_entry
    t = _make_fake_tensor(seq_tokens=12)
    write_cache_entry(tmp_path, "abcd1234abcd1234", "synthetic_systems", "sample_001", t, h16=3, w16=4)
    tensor, h16, w16 = read_cache_entry(tmp_path, "abcd1234abcd1234", "synthetic_systems", "sample_001")
    assert tensor.shape == (12, 1280)
    assert tensor.dtype == torch.bfloat16
    assert h16 == 3
    assert w16 == 4
    assert torch.allclose(tensor.float(), t.float(), atol=1e-3)


def test_read_raises_cache_miss_on_absent_key(tmp_path: Path) -> None:
    from src.data.encoder_cache import CacheMiss, read_cache_entry
    with pytest.raises(CacheMiss):
        read_cache_entry(tmp_path, "abcd1234abcd1234", "synthetic_systems", "does_not_exist")


def test_two_sample_keys_do_not_overwrite(tmp_path: Path) -> None:
    """Two different sample keys under same hash → distinct files, no collision."""
    from src.data.encoder_cache import read_cache_entry, write_cache_entry
    t1 = _make_fake_tensor(seq_tokens=5)
    t2 = _make_fake_tensor(seq_tokens=7)
    write_cache_entry(tmp_path, "hash0000hash0000", "primus", "sample_A", t1, h16=1, w16=5)
    write_cache_entry(tmp_path, "hash0000hash0000", "primus", "sample_B", t2, h16=1, w16=7)
    r1, _, _ = read_cache_entry(tmp_path, "hash0000hash0000", "primus", "sample_A")
    r2, _, _ = read_cache_entry(tmp_path, "hash0000hash0000", "primus", "sample_B")
    assert r1.shape == (5, 1280)
    assert r2.shape == (7, 1280)


def test_write_cache_metadata_creates_json(tmp_path: Path) -> None:
    from src.data.encoder_cache import write_cache_metadata
    meta = {"encoder_weights_path": "/fake/path.pt", "hidden_dim": 1280,
            "dtype": "bfloat16", "sample_count": 42, "total_bytes": 1000000}
    write_cache_metadata(tmp_path, "abcd1234abcd1234", meta)
    md_path = tmp_path / "abcd1234abcd1234" / "metadata.json"
    assert md_path.exists()
    loaded = json.loads(md_path.read_text())
    assert loaded["sample_count"] == 42
    assert loaded["hidden_dim"] == 1280


def test_write_returns_correct_path(tmp_path: Path) -> None:
    from src.data.encoder_cache import write_cache_entry
    t = _make_fake_tensor()
    p = write_cache_entry(tmp_path, "hash0000hash0000", "grandstaff_systems", "my_sample", t, h16=4, w16=8)
    assert p == tmp_path / "hash0000hash0000" / "grandstaff_systems" / "my_sample.pt"
    assert p.exists()


def test_write_cache_entry_does_not_serialize_full_batch_storage(tmp_path: Path) -> None:
    """Regression: writing a sliced tensor must not bloat the file by the batch factor.

    `feature_map[i]` from a batched encoder forward is a slice that shares storage
    with the full batch. Without `.clone()` in the writer, torch.save would serialize
    the entire batch's underlying storage per sample, bloating each file ~B× on disk.
    This test fakes that scenario and asserts the file size matches the tensor data
    size (with small torch-pickle header overhead), NOT the full batch storage.
    """
    import torch
    from src.data.encoder_cache import write_cache_entry

    batch_size = 8
    h16, w16 = 4, 6
    seq_tokens = h16 * w16
    hidden_dim = 1280

    # Simulate the build script's pattern: a (B, C, H, W) batch tensor, then
    # take a per-sample slice and reshape to (seq_tokens, C).
    full_batch = torch.randn(batch_size, hidden_dim, h16, w16, dtype=torch.bfloat16)
    sample_slice = full_batch[3].flatten(1).transpose(0, 1)  # (seq_tokens, 1280)
    # The slice shares storage with the full batch:
    assert sample_slice.untyped_storage().nbytes() == full_batch.numel() * 2

    p = write_cache_entry(tmp_path, "h" * 16, "synthetic_systems", "s0", sample_slice, h16=h16, w16=w16)

    actual_bytes = p.stat().st_size
    data_bytes = seq_tokens * hidden_dim * 2  # bf16
    # Allow up to 32 KB of pickle/header overhead, but reject 8× bloat (full batch).
    assert actual_bytes < data_bytes + 32_768, (
        f"file size {actual_bytes} bytes exceeds tensor data size {data_bytes} + 32KB header. "
        f"Likely the slice's full-batch storage was serialized — clone before save."
    )


# ---------------------------------------------------------------------------
# Resumability test (mocked encoder)
# ---------------------------------------------------------------------------

def test_builder_skips_already_cached_entries(tmp_path: Path) -> None:
    """If 5 of 10 entries are already cached, builder calls encoder only 5 times."""
    from unittest.mock import MagicMock, patch
    from src.data.encoder_cache import write_cache_entry, _sanitize_sample_key

    hash16 = "test0000test0000"
    cache_root = tmp_path / "cache"

    # Pre-write 5 entries
    for i in range(5):
        t = _make_fake_tensor(seq_tokens=8)
        key = _sanitize_sample_key(f"synthetic_systems:sample_{i:03d}")
        write_cache_entry(cache_root, hash16, "synthetic_systems", key, t, h16=2, w16=4)

    # Simulate 10 manifest entries
    entries = [
        {"sample_id": f"synthetic_systems:sample_{i:03d}", "dataset": "synthetic_systems",
         "image_path": str(tmp_path / f"img_{i}.png")}
        for i in range(10)
    ]
    # Create fake image files
    import numpy as np
    from PIL import Image
    for i in range(10):
        img = Image.fromarray(np.ones((32, 64), dtype=np.uint8) * 200)
        img.save(tmp_path / f"img_{i}.png")

    import torch

    encoder_call_count = [0]

    def fake_encode(image_batch):
        encoder_call_count[0] += image_batch.shape[0]
        B = image_batch.shape[0]
        return torch.ones(B, 1280, 2, 4, dtype=torch.bfloat16)

    # Import the core builder loop. The repo root must be on sys.path so
    # `scripts/` is discoverable; pytest is normally invoked from the repo root,
    # which satisfies this. If the import fails, add a `scripts/__init__.py`
    # or run pytest with `PYTHONPATH=.` from the repo root.
    from scripts.build_encoder_cache import _build_cache_for_entries
    _build_cache_for_entries(
        entries=entries,
        cache_root=cache_root,
        hash16=hash16,
        encode_fn=fake_encode,
        project_root=tmp_path,
        image_height=32,
        image_width=64,
        batch_size=2,
        dry_run=False,
    )

    assert encoder_call_count[0] == 5, (
        f"Expected 5 encoder calls (5 cache hits skipped), got {encoder_call_count[0]}"
    )
