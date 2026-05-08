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
