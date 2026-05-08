# Stage 3 Phase 0 — Encoder Cache Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, validate, and benchmark a content-addressed encoder feature cache for the 90% cached training tier (215,985 samples across synthetic_systems + grandstaff_systems + primus_systems), plus the two-tier dataloader and model forward-branching code that reads cached features for the cached tier and runs live encoder passes for cameraprimus.

**Architecture:** Cache identity is a 16-character hex string derived from SHA-256(encoder_weights_sha + preproc_cfg_sha + arch_version + git_head_sha); each sample is stored as a per-sample bf16 `.pt` file at `data/cache/encoder/<hash16>/<tier>/<sample_key>.pt` (shape `(seq_tokens, 1280)`, no batch dim). At training time the tier-grouped sampler guarantees batches are 100% cached or 100% live, and `RadioStageB.forward` dispatches on a `cached_features` keyword argument — bypassing `RadioEncoder` and reshaping the flat cache tensor back to `(B, C, H, W)` before `deformable_attention`.

**Tech Stack:** PyTorch (bf16), Python 3.14, pytest, `torch.save`/`torch.load`, `hashlib`, `yaml`, RADIO C-RADIO v4-H, SSH to GPU box at 10.10.1.29.

---

## Decisions locked at plan time

1. **Git HEAD SHA in cache hash (Open Q #1):** Included as 4th hash component. Default-on. `--ignore-git-sha` CLI flag skips it for fresh-clone CI environments. Rationale: low cost, prevents silent drift from the alignment-fix branch; consistent with the spec's "any change to encoder weights, preprocessing config, or RADIO architecture version produces a fresh directory" principle.

2. **Preprocessing config source (Open Q #2):** Formalized as `configs/preproc_stage3.yaml`. Hash is computed from the parsed dict serialized with sorted keys to JSON then SHA-256 (whitespace-insensitive). Fields enumerated: `image_height: 250`, `image_width: 2500`, `normalize_mean: null`, `normalize_std: null`, `pad_color: 1.0`. Values cross-checked against `StageBDataset.__init__` at `train.py:554–555` (`image_height=250`, `image_width=2500`); no explicit normalize in the dataset code — pad-to-white is the convention (pixel `1.0`).

3. **DoRA adapter location (Open Q #3):** Encoder-side AND decoder-side. Confirmed from `src/train/model_factory.py:list_radio_dora_target_modules()` (lines 59–72): targets `qkv`, `proj`, `fc1`, `fc2` on the RADIO ViT encoder, plus `q_proj`, `k_proj`, `v_proj`, `out_proj`, `cross_attn_*` on the decoder. **Cache stores `RadioEncoder.forward()` output including any frozen encoder-side DoRA adapter transformations.** The `encode_staff` call path (`radio_stage_b.py:142`) runs `self.encoder(images)`, which passes through all frozen encoder layers including DoRA adapters. No special handling needed — the cache naturally captures their output because the encoder (+ adapters) is frozen. Verified: no Phase 0a verification task needed.

4. **ViT positional-embedding interpolation (Open Q #4):** Validation only. Task 1 (dry-run) prints the `get_nearest_supported_resolution` snapped shapes for 20 diverse system crops and asserts none are degenerate. The dry-run runs on the GPU box as its first operational step. No production code change needed.

5. **`sample_key` sanitization (Open Q #5):** Strip `<dataset>:` prefix from manifest `sample_id`; replace any remaining `/`, `:`, `\` with `__`. Codified in `_sanitize_sample_key(sample_id: str) -> str` in `src/data/encoder_cache.py`. Unit test covers colon-prefix strip, slash replace, backslash replace, and no-op clean key.

6. **Contour logits (Open Q #6):** Not cached. The cache stores `RadioEncoder.forward()` output only (shape `(B, 1280, H/16, W/16)` → flattened to `(seq_tokens, 1280)` per sample). `contour_logits` are computed at training time from `positional_bridge`'s `memory` output via `contour_head` — the same computation happens in both cached and live paths. No difference in contour loss behavior.

7. **Batch dimension in cache (Open Q #7):** Per-sample `.pt` files store shape `(seq_tokens, 1280)` — **no batch dim**. The `deformable_attention` in `encode_staff` (`radio_stage_b.py:149`) accepts `(B, seq, C)` (sequence form after `flatten(2).transpose(1,2)`). The cached-features forward branch reshapes the collated `(B, seq_tokens, 1280)` batch back to `(B, 1280, H/16, W/16)` using stored height/width metadata before feeding `deformable_attention`. Because seq_tokens = H/16 × W/16, the reshape is lossless. The cache writer stores `(seq_tokens, 1280)` plus `(H_16, W_16)` shape fields in a companion sidecar or within the `.pt` file as a tuple `(tensor, h16, w16)`.

---

## Files to create or modify

**New files:**
- `configs/preproc_stage3.yaml` — preprocessing config (Task 0)
- `src/data/encoder_cache.py` — cache I/O library: `compute_cache_hash`, `_sanitize_sample_key`, `write_cache_entry`, `write_cache_metadata`, `read_cache_entry`, `cache_entry_exists`, `CacheMiss` (Tasks 1–2)
- `tests/data/test_encoder_cache.py` — unit tests for cache I/O and hash (Tasks 1–2)
- `scripts/build_encoder_cache.py` — offline cache builder with `--dry-run` (Task 5)
- `tests/models/test_radio_stage_b_cached.py` — cached-features forward branch test (Task 4)
- `src/train/tier_sampler.py` — `build_tier_grouped_sampler` (Task 8)
- `tests/train/test_tier_grouped_sampler.py` — sampler unit tests (Task 8)
- `tests/train/test_cached_dataset.py` — dataset cached-path tests (Task 9)
- `scripts/measure_encoder_cache_throughput.py` — throughput + VRAM sweep (Task 12)
- `docs/superpowers/handoffs/2026-05-08-radio-stage3-phase0-complete-handoff.md` — final handoff (Task 14)

**Modified files:**
- `src/models/radio_stage_b.py` — add `cached_features` arg to `forward()`, add `forward_from_cache()` helper (Task 3)
- `src/train/train.py` — extend `StageBDataset.__getitem__` and `collate_fn` for cached/live tier dispatch (Task 9)

---

## Phase 0 Exit Criteria (from spec §"0d")

All five must hold before Phase 1 launch:

1. **Disk math:** Phase 0a measurement ≤ 2 TB; GPU box NVMe has ≥ 200 GB headroom after the build.
2. **Cache built:** Total on-disk size matches Phase 0a measurement ±5%.
3. **Correctness:** 100 random cached samples loaded → cached-path decoder forward matches live-encoder + decoder forward to ≤ 1e-3 max absolute element-wise diff (bf16 tolerance).
4. **Throughput:** Cached dataloader saturates GPU at training-shape batch size; dataloader is not the new bottleneck.
5. **Memory sweep:** Run b=4, 8, 16, 32 on 200-batch sample. Largest b with VRAM ≤ 80% defines `b_cached` for Phase 1. Live path at b=2 still fits when interleaved with cached batches at chosen `b_cached`.

---

## Tasks

### Task 0: Create branch and preprocessing config

**Files:**
- Create: `configs/preproc_stage3.yaml`

**Why this task:** Establishes the feature branch and formalizes the preprocessing config that is a required input to the cache hash. Must exist before any cache I/O code is written.

- [ ] **Step 1: Create feature branch**

Run:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git checkout -b feat/stage3-encoder-cache feat/system-level-rebuild
```
Expected: `Switched to a new branch 'feat/stage3-encoder-cache'`

- [ ] **Step 2: Create `configs/preproc_stage3.yaml`**

```yaml
# Preprocessing config for Stage 3 encoder cache.
# Hash is computed from the parsed dict (sorted keys → JSON → SHA-256),
# not from raw file bytes, so whitespace edits do not invalidate the cache.
#
# Fields must enumerate every parameter that affects pixel values entering
# RadioEncoder.forward(). Cross-checked against StageBDataset.__init__
# (train.py:554-555): image_height=250, image_width=2500, pad-to-white=1.0.
# No explicit normalize is applied in the current pipeline (RADIO expects
# pixels in [0,1]; clamping is done in RadioEncoder.forward at line 73).

image_height: 250
image_width: 2500
normalize_mean: null
normalize_std: null
pad_color: 1.0
```

- [ ] **Step 3: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add configs/preproc_stage3.yaml
git commit -m "feat(data): add preproc_stage3.yaml for encoder cache hash"
```

> **Review:** Confirm YAML fields match hardcoded values in `StageBDataset.__init__` at `train.py:554–555`. Confirm branch is off `feat/system-level-rebuild` (not `main`).

---

### Task 1: Cache hash + sample key sanitization (TDD)

**Files:**
- Create: `src/data/encoder_cache.py` (partial — hash + sanitize only)
- Create: `tests/data/test_encoder_cache.py` (hash + sanitize tests)

**Why this task:** `compute_cache_hash` and `_sanitize_sample_key` are the foundation all other cache I/O functions depend on. Getting them green and committed first ensures later tasks build on verified primitives.

- [ ] **Step 1: Write failing tests**

Create `tests/data/test_encoder_cache.py`:

```python
"""Unit tests for src/data/encoder_cache.py."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError: cannot import name 'compute_cache_hash' from 'src.data.encoder_cache'` (file does not exist yet).

- [ ] **Step 3: Implement hash + sanitize in `src/data/encoder_cache.py`**

Create `src/data/encoder_cache.py`:

```python
"""Encoder feature cache I/O library for Stage 3.

Provides content-addressed storage for RadioEncoder.forward() output tensors.
Cache identity is derived from four inputs:
  1. SHA-256 of the encoder checkpoint file bytes.
  2. SHA-256 of the preprocessing config dict (sorted-keys JSON).
  3. RADIO architecture version string.
  4. Git HEAD SHA (optional; omit with git_head_sha=None for CI environments).

Storage layout:
  <cache_root>/<hash16>/<tier>/<sample_key>.pt
    where each .pt file is a tuple (tensor, h16, w16) saved via torch.save.
    tensor shape: (seq_tokens, 1280), dtype=bfloat16.
    h16, w16: spatial dimensions of the encoder output before flattening.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------

class CacheMiss(FileNotFoundError):
    """Raised when the cache does not contain an entry for the given key."""


# ---------------------------------------------------------------------------
# Hash + sanitization helpers
# ---------------------------------------------------------------------------

def _sanitize_sample_key(sample_id: str) -> str:
    """Derive a filesystem-safe filename stem from a manifest sample_id.

    Rules:
      1. If sample_id contains ':', strip everything up to and including the
         first ':' (removes the '<dataset>:' prefix).
      2. Replace any remaining '/', ':', '\\' with '__'.
    """
    if ":" in sample_id:
        sample_id = sample_id.split(":", 1)[1]
    sample_id = sample_id.replace("/", "__").replace(":", "__").replace("\\", "__")
    return sample_id


def compute_cache_hash(
    encoder_weights_path: Path,
    preproc_cfg: dict,
    radio_arch_version: str,
    *,
    git_head_sha: Optional[str],
) -> str:
    """Return a 16-character hex string used as the cache directory name.

    Args:
        encoder_weights_path: Path to the Stage 2 v2 checkpoint file. Its
            full bytes are SHA-256'd so any weight change invalidates the cache.
        preproc_cfg: Preprocessing config dict. Hashed via sorted-key JSON so
            key ordering and whitespace changes in YAML do not invalidate.
        radio_arch_version: String like "c-radio_v4-h". Hashed as UTF-8 bytes.
        git_head_sha: Current git HEAD SHA (hex string). Pass None to skip
            (e.g. fresh-clone CI environments where git state is unstable).

    Returns:
        First 16 hex characters of the combined SHA-256 digest.
    """
    # Component 1: encoder weights file bytes
    weights_sha = hashlib.sha256(
        Path(encoder_weights_path).read_bytes()
    ).hexdigest()

    # Component 2: preprocessing config (sorted-key JSON, whitespace-insensitive)
    preproc_json = json.dumps(preproc_cfg, sort_keys=True, default=str)
    preproc_sha = hashlib.sha256(preproc_json.encode("utf-8")).hexdigest()

    # Component 3: RADIO architecture version
    arch_sha = hashlib.sha256(radio_arch_version.encode("utf-8")).hexdigest()

    # Component 4: git HEAD SHA (optional drift protection)
    components = [weights_sha, preproc_sha, arch_sha]
    if git_head_sha is not None:
        git_sha = hashlib.sha256(git_head_sha.encode("utf-8")).hexdigest()
        components.append(git_sha)

    combined = hashlib.sha256("".join(components).encode("utf-8")).hexdigest()
    return combined[:16]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py -v -k "hash or sanitize"`
Expected: all 11 tests PASS.

- [ ] **Step 5: Run full test suite for regressions**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green; no regressions.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/encoder_cache.py tests/data/test_encoder_cache.py
git commit -m "feat(data): encoder cache hash and sample key sanitization"
```

> **Review:** Confirm `compute_cache_hash` signature matches exactly what `build_encoder_cache.py` will call in Task 5. Confirm `_sanitize_sample_key("ds:body:extra") == "body__extra"` (double-colon edge case).

---

### Task 2: Cache write / read / hit-miss / collision (TDD)

**Files:**
- Modify: `src/data/encoder_cache.py` — add `write_cache_entry`, `write_cache_metadata`, `read_cache_entry`, `cache_entry_exists`
- Modify: `tests/data/test_encoder_cache.py` — add write/read/hit-miss/collision tests

**Why this task:** The write and read primitives are consumed by the builder (Task 5) and dataset (Task 9). Green tests here confirm the on-disk format is round-trip stable before any downstream code is written.

- [ ] **Step 1: Append failing tests to `tests/data/test_encoder_cache.py`**

Append to the end of `tests/data/test_encoder_cache.py`:

```python
# ---------------------------------------------------------------------------
# Write / read / hit-miss / collision tests
# ---------------------------------------------------------------------------

def _make_fake_tensor(seq_tokens: int = 20, hidden_dim: int = 1280) -> torch.Tensor:
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py -v -k "write or read or exists or miss or collision or metadata or returns_correct"`
Expected: FAIL with `ImportError: cannot import name 'write_cache_entry'` (functions not defined yet).

- [ ] **Step 3: Implement write/read/metadata functions in `src/data/encoder_cache.py`**

Append to `src/data/encoder_cache.py` (after the existing hash functions):

```python
# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_cache_entry(
    cache_root: Path,
    hash16: str,
    tier: str,
    sample_key: str,
    tensor: torch.Tensor,
    *,
    h16: int,
    w16: int,
) -> Path:
    """Write a per-sample encoder feature tensor to disk.

    Stores a tuple (tensor, h16, w16) via torch.save so the reader can
    reconstruct the original (B, C, H/16, W/16) shape for deformable_attention.

    Args:
        cache_root: Root directory for all cache versions.
        hash16: 16-char hex cache identity string.
        tier: Dataset tier name, e.g. "synthetic_systems".
        sample_key: Sanitized sample identifier (no colons or slashes).
        tensor: bf16 tensor of shape (seq_tokens, 1280). Must be on CPU.
        h16: Height dimension of the encoder spatial output (H/16).
        w16: Width dimension of the encoder spatial output (W/16).

    Returns:
        Path to the written .pt file.
    """
    dest_dir = Path(cache_root) / hash16 / tier
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sample_key}.pt"
    payload = (tensor.cpu().to(torch.bfloat16), int(h16), int(w16))
    torch.save(payload, dest)
    return dest


def write_cache_metadata(
    cache_root: Path,
    hash16: str,
    metadata: dict,
) -> None:
    """Write or update metadata.json at cache_root/<hash16>/metadata.json."""
    dest_dir = Path(cache_root) / hash16
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "metadata.json"
    dest.write_text(json.dumps(metadata, indent=2, default=str))


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------

def cache_entry_exists(
    cache_root: Path,
    hash16: str,
    tier: str,
    sample_key: str,
) -> bool:
    """Return True if the .pt file exists on disk (path-stat only, no load)."""
    p = Path(cache_root) / hash16 / tier / f"{sample_key}.pt"
    return p.exists()


def read_cache_entry(
    cache_root: Path,
    hash16: str,
    tier: str,
    sample_key: str,
) -> Tuple[torch.Tensor, int, int]:
    """Load and return the cached bf16 tensor plus spatial shape.

    Returns:
        (tensor, h16, w16) where tensor has shape (seq_tokens, 1280) and
        h16 * w16 == seq_tokens.

    Raises:
        CacheMiss: If the .pt file does not exist.
    """
    p = Path(cache_root) / hash16 / tier / f"{sample_key}.pt"
    if not p.exists():
        raise CacheMiss(
            f"Cache miss: no entry for tier={tier!r} key={sample_key!r} "
            f"under hash {hash16!r} in {cache_root}"
        )
    payload = torch.load(p, weights_only=True, map_location="cpu")
    tensor, h16, w16 = payload
    return tensor, int(h16), int(w16)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/data/encoder_cache.py tests/data/test_encoder_cache.py
git commit -m "feat(data): encoder cache write/read/exists/metadata I/O"
```

> **Review:** Confirm `.pt` files store a 3-tuple `(tensor, h16, w16)` — not just the tensor. Confirm `read_cache_entry` uses `weights_only=True`. Confirm `CacheMiss` is a subclass of `FileNotFoundError`.

---

### Task 3: `RadioStageB.forward` cached-features branch (TDD)

**Files:**
- Modify: `src/models/radio_stage_b.py` — add `cached_features` kwarg to `forward()`, implement bypass path
- Create: `tests/models/test_radio_stage_b_cached.py`

**Why this task:** The `cached_features` branch is what makes encoder-free training possible. It must be correct (output matches live path to ≤ 1e-3) before the dataset and training-loop changes depend on it.

- [ ] **Step 1: Write failing test**

Create `tests/models/test_radio_stage_b_cached.py`:

```python
"""Tests for RadioStageB.forward cached_features branch.

These tests run on CPU with a tiny dummy model (vocab_size=10, 2 decoder layers)
to avoid requiring the RADIO hub download. We mock RadioEncoder to return a
deterministic feature map.
"""
from __future__ import annotations

import torch
import pytest


def _build_tiny_model():
    """Build a RadioStageB with stub encoder that doesn't call torch.hub."""
    import torch.nn as nn
    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig

    config = RadioStageBConfig(
        decoder_dim=64,
        decoder_layers=2,
        decoder_heads=4,
        vocab_size=10,
        max_decode_len=16,
        contour_classes=3,
    )
    model = RadioStageB.__new__(RadioStageB)
    nn.Module.__init__(model)
    model.config = config

    # Stub encoder: returns (B, 1280, 2, 4) feature map deterministically
    class _StubEncoder(nn.Module):
        hidden_dim = 1280
        def forward(self, x):
            B = x.shape[0]
            return torch.ones(B, 1280, 2, 4, dtype=x.dtype, device=x.device)

    from src.models.davit_stage_b import DecoderBlock, DeformableContextBlock, PositionalBridge, RMSNorm
    model.encoder = _StubEncoder()
    model.deformable_attention = DeformableContextBlock(dim=1280, heads=4)
    model.positional_bridge = PositionalBridge(encoder_dim=1280, decoder_dim=64)
    model.token_embedding = nn.Embedding(10, 64)
    model.decoder_blocks = nn.ModuleList([DecoderBlock(64, 4) for _ in range(2)])
    model.decoder_norm = RMSNorm(64)
    model.lm_head = nn.Linear(64, 10)
    model.contour_head = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 3))
    model.max_decode_length = 16
    model.eval()
    return model


def test_forward_cached_features_signature_accepted() -> None:
    """forward() must accept cached_features without raising TypeError."""
    model = _build_tiny_model()
    B, seq_tokens, C = 2, 8, 1280
    cached = torch.randn(B, seq_tokens, C)
    tgt = torch.zeros(B, 4, dtype=torch.long)
    with torch.no_grad():
        out = model.forward(cached_features=cached, tgt=tgt, _h16=2, _w16=4)
    assert "logits" in out
    assert "contour_logits" in out


def test_forward_cached_matches_live_to_1e3() -> None:
    """Cached path output must match live path output to ≤ 1e-3 max abs diff.

    Both paths use the same stub encoder (deterministic ones). We:
      1. Run live forward: image → encoder → deformable_attn → bridge → decoder
      2. Capture the encoder output (feature_map from stub: ones tensor)
      3. Flatten to (seq_tokens, 1280), run cached forward
      4. Assert logits match to ≤ 1e-3
    """
    model = _build_tiny_model()
    B, H, W = 1, 32, 64  # dummy image; stub encoder ignores content
    image = torch.rand(B, 1, H, W)
    tgt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    with torch.no_grad():
        live_out = model.forward(image=image, tgt=tgt)
        # Manually extract encoder output for the cached path
        feature_map = model.encoder(image)  # (B, 1280, 2, 4)
        h16, w16 = feature_map.shape[2], feature_map.shape[3]
        cached_tensor = feature_map.flatten(2).transpose(1, 2)  # (B, 8, 1280)
        cached_out = model.forward(cached_features=cached_tensor, tgt=tgt, _h16=h16, _w16=w16)

    diff = (live_out["logits"].float() - cached_out["logits"].float()).abs().max().item()
    assert diff <= 1e-3, f"max abs diff {diff} exceeds 1e-3 tolerance"


def test_forward_raises_without_image_or_cached() -> None:
    """forward() with neither image nor cached_features must raise ValueError."""
    model = _build_tiny_model()
    tgt = torch.zeros(1, 3, dtype=torch.long)
    with pytest.raises(ValueError, match="requires"):
        model.forward(tgt=tgt)


def test_cached_features_skips_encoder_call() -> None:
    """When cached_features is provided, encoder.forward must NOT be called."""
    model = _build_tiny_model()
    call_count = [0]
    original_forward = model.encoder.forward

    def counting_forward(x):
        call_count[0] += 1
        return original_forward(x)

    model.encoder.forward = counting_forward

    cached = torch.randn(1, 8, 1280)
    tgt = torch.zeros(1, 3, dtype=torch.long)
    with torch.no_grad():
        model.forward(cached_features=cached, tgt=tgt, _h16=2, _w16=4)

    assert call_count[0] == 0, "encoder.forward was called despite cached_features being provided"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/models/test_radio_stage_b_cached.py -v`
Expected: FAIL with `TypeError: RadioStageB.forward() got an unexpected keyword argument 'cached_features'` or `TypeError: forward() got an unexpected keyword argument '_h16'`.

- [ ] **Step 3: Modify `src/models/radio_stage_b.py` to add cached-features branch**

The current `forward()` method at lines 189–209 reads:
```python
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        *,
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict:
```

Replace the entire `forward` method (lines 189–209) with:

```python
    def forward(
        self,
        image: Optional[torch.Tensor] = None,
        tgt: Optional[torch.Tensor] = None,
        *,
        cached_features: Optional[torch.Tensor] = None,  # (B, seq_tokens, 1280) bf16
        _h16: Optional[int] = None,   # spatial height before flatten (required if cached_features given)
        _w16: Optional[int] = None,   # spatial width before flatten (required if cached_features given)
        pixel_values: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        **_: object,
    ) -> dict:
        if image is None:
            image = pixel_values
        if tgt is None:
            tgt = decoder_input_ids if decoder_input_ids is not None else input_ids

        if cached_features is not None:
            # --- Cached path: bypass RadioEncoder entirely ---
            # cached_features: (B, seq_tokens, 1280) — the raw encoder spatial output
            # stored per-sample as (seq_tokens, 1280) and collated to (B, seq_tokens, 1280).
            # Reshape back to (B, 1280, H/16, W/16) for deformable_attention.
            if _h16 is None or _w16 is None:
                raise ValueError(
                    "RadioStageB.forward: _h16 and _w16 are required when cached_features is provided. "
                    "These encode the original spatial dimensions (H/16, W/16) before flattening."
                )
            if tgt is None:
                raise ValueError(
                    "RadioStageB.forward requires a target token tensor (tgt) when using cached_features."
                )
            B, seq_tokens, C = cached_features.shape
            # Reshape: (B, seq_tokens, C) → (B, C, H/16, W/16)
            feature_map = cached_features.transpose(1, 2).reshape(B, C, int(_h16), int(_w16))
            # Run trainable deformable_attention + positional_bridge (same as live path)
            batch, channels, height, width = feature_map.shape
            sequence = feature_map.flatten(2).transpose(1, 2)
            sequence = self.deformable_attention(sequence, height, width)
            sequence = sequence.transpose(1, 2).reshape(batch, channels, height, width)
            memory, _ = self.positional_bridge(sequence)
            contour_logits = self.contour_head(memory.mean(dim=1))
            logits, _, _ = self.decode_tokens(tgt, memory)
            return {"logits": logits, "contour_logits": contour_logits}

        # --- Live path: run encoder ---
        if image is None or tgt is None:
            raise ValueError(
                "RadioStageB.forward requires an image tensor and a target/input token tensor."
            )
        memory, contour_logits = self.encode_staff(image)
        logits, _, _ = self.decode_tokens(tgt, memory)
        return {"logits": logits, "contour_logits": contour_logits}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/models/test_radio_stage_b_cached.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green; no regressions in existing model tests.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/models/radio_stage_b.py tests/models/test_radio_stage_b_cached.py
git commit -m "feat(models): add cached_features branch to RadioStageB.forward"
```

> **Review:** Confirm the reshape logic: `cached_features.transpose(1,2).reshape(B, C, h16, w16)` produces `(B, 1280, H/16, W/16)` — matching the shape `encode_staff` would have produced. Confirm encoder.forward is not called in the cached path (test `test_cached_features_skips_encoder_call` verifies this). Confirm `_h16`/`_w16` are passed through the collate_fn (verified in Task 9).

---

### Task 4: DoRA verification + ViT positional-embedding Phase 0a validation (operational)

**Files:**
- No new files — this is an operational verification task on the GPU box.

**Why this task:** Confirms the DoRA adapter location (already established from code inspection) and validates that RADIO's positional-embedding interpolation handles typical system-crop aspect ratios without degeneracy. Must be done before full cache build so we know encoder output quality is sound.

- [ ] **Step 1: Verify DoRA adapter location from code (local, already done)**

Grep confirms DoRA adapters are encoder-side AND decoder-side:
```bash
grep -n "list_radio_dora_target_modules\|qkv\|fc1\|fc2\|proj" /home/ari/work/Clarity-OMR-Train-RADIO/src/train/model_factory.py | head -20
```
Expected: see `qkv`, `proj`, `fc1`, `fc2` listed as RADIO ViT targets (lines 61–64). **Decision confirmed from code:** encoder-side DoRA adapters are frozen with the encoder. Cache captures their output naturally. No code change needed.

- [ ] **Step 2: Push branch to GPU box**

Run:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
```

Pull on GPU box:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; git fetch origin; git checkout feat/stage3-encoder-cache; git pull origin feat/stage3-encoder-cache"'
```
Expected: branch checked out at current HEAD.

- [ ] **Step 3: Validate ViT positional-embedding interpolation on 20 diverse system crops**

Run on GPU box:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe -c \"
import torch, sys
sys.path.insert(0, \".\")
from src.models.radio_stage_b import RadioEncoder
enc = RadioEncoder()
enc.eval()
# Typical system-crop aspect ratios: wide/short (3-staff), medium (6-staff), near-square (1-staff)
test_shapes = [
    (96, 2496), (128, 2496), (176, 2496), (224, 2496), (304, 2496),
    (96, 1248), (128, 1248), (256, 1248), (384, 1248),
    (96, 624),  (128, 624),  (256, 624),
    (96, 312),  (128, 312),
    (176, 2496), (304, 2496), (400, 2496), (512, 2496), (704, 2496), (992, 2496),
]
print(f\'Testing {len(test_shapes)} shapes:\')
for h, w in test_shapes:
    snapped = enc.model.get_nearest_supported_resolution(h, w)
    sh, sw = snapped
    seq = (sh // 16) * (sw // 16)
    ok = sh > 0 and sw > 0 and seq > 0
    print(f\'  input({h:4d},{w:4d}) -> snapped({sh:4d},{sw:4d}) seq={seq:5d} [OK={ok}]\')
print(\'All shapes validated.\')
\""'
```

Expected: all 20 lines print `[OK=True]`. Common snapped shapes for system crops:
- `(96, 2496)` → approx `(96, 2496)` or nearest supported resolution; seq_tokens ≈ 6×156 = 936
- `(304, 2496)` → approx `(304, 2496)`; seq_tokens ≈ 19×156 = 2,964

Assert none have `seq=0` or negative dimensions. If any shape produces a degenerate result (`seq=0`, `sh=0`, `sw=0`), halt and escalate — the fixed-pad-to-(1000,2500) fallback becomes mandatory.

- [ ] **Step 4: Record findings**

Record the snapped shape for `(250, 2500)` (the StageBDataset default) specifically — this is the shape the cache will predominantly use. Note the resulting `seq_tokens` value; this is the correct token count for the per-sample disk math in Task 5.

> **Review:** Confirm no degenerate shapes. Confirm the snapped shape for `(250, 2500)` is recorded — this value feeds Task 5's disk projection.

---

### Task 5: Cache builder script with `--dry-run` (TDD)

**Files:**
- Create: `scripts/build_encoder_cache.py`
- Modify: `tests/data/test_encoder_cache.py` — add resumability test

**Why this task:** The dry-run mode provides the Phase 0a sizing measurement. The full builder is used in Task 6 (full cache build). Resumability test locks the skip-if-present behavior.

- [ ] **Step 1: Append resumability test to `tests/data/test_encoder_cache.py`**

Append to `tests/data/test_encoder_cache.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py::test_builder_skips_already_cached_entries -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.build_encoder_cache'` or `ImportError: cannot import name '_build_cache_for_entries'`.

- [ ] **Step 3: Implement `scripts/build_encoder_cache.py`**

Create `scripts/build_encoder_cache.py`:

```python
#!/usr/bin/env python3
"""Offline encoder feature cache builder for Stage 3.

Iterates the combined Stage 3 manifest, filters to the 90% cached tier
(synthetic_systems, grandstaff_systems, primus_systems), runs RadioEncoder
under torch.no_grad() + bf16 autocast, and writes per-sample .pt files to
data/cache/encoder/<hash16>/<tier>/<sample_key>.pt.

Usage (dry-run for Phase 0a sizing):
    python scripts/build_encoder_cache.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --batch-size 8 \\
        --device cuda \\
        --dry-run

Usage (full build with resume):
    python scripts/build_encoder_cache.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --batch-size 8 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data.encoder_cache import (
    _sanitize_sample_key,
    cache_entry_exists,
    compute_cache_hash,
    write_cache_entry,
    write_cache_metadata,
)

CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
DRY_RUN_SAMPLE_LIMIT = 1_000


def _load_manifest_entries(manifest_path: Path, cached_only: bool = True) -> list[dict]:
    """Load JSONL manifest, optionally filtering to cached-tier datasets."""
    entries = []
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if cached_only and entry.get("dataset") not in CACHED_TIER_DATASETS:
                continue
            # Skip entries with null image_path (filtered staves from alignment fix)
            if entry.get("image_path") is None:
                continue
            entries.append(entry)
    return entries


def _load_image_tensor(
    image_path: Path,
    project_root: Path,
    image_height: int,
    image_width: int,
) -> Optional[torch.Tensor]:
    """Load and resize a single image to (1, H, W) float32 tensor in [0, 1]."""
    import torchvision.transforms.functional as TF
    from PIL import Image

    full_path = project_root / image_path if not Path(image_path).is_absolute() else Path(image_path)
    if not full_path.exists():
        return None
    try:
        img = Image.open(full_path).convert("L")
        # Resize to target height, pad width to image_width
        scale = image_height / img.height
        new_w = min(int(img.width * scale), image_width)
        img = img.resize((new_w, image_height), Image.LANCZOS)
        # Create white canvas and paste
        canvas = Image.new("L", (image_width, image_height), color=255)
        canvas.paste(img, (0, 0))
        tensor = TF.to_tensor(canvas)  # (1, H, W) float32 in [0, 1]
        return tensor
    except Exception as exc:
        print(f"[builder] WARNING: failed to load {full_path}: {exc}", file=sys.stderr)
        return None


def _build_cache_for_entries(
    entries: list[dict],
    cache_root: Path,
    hash16: str,
    encode_fn: Callable[[torch.Tensor], torch.Tensor],
    project_root: Path,
    image_height: int,
    image_width: int,
    batch_size: int,
    dry_run: bool,
) -> dict:
    """Core builder loop. Returns stats dict.

    Args:
        entries: Manifest entries to process (already filtered to cached tier).
        cache_root: Root directory for cache.
        hash16: Cache identity hash (16 hex chars).
        encode_fn: Callable[image_batch_cpu] -> (B, 1280, H/16, W/16) tensor.
            image_batch_cpu is (B, 1, H, W) float32 on CPU. Output must be
            CPU bf16 with shape (B, 1280, h16, w16).
        project_root: Repo root for resolving relative image paths.
        image_height: Target image height in pixels.
        image_width: Target image width in pixels.
        batch_size: Number of samples per encoder forward pass.
        dry_run: If True, limit to DRY_RUN_SAMPLE_LIMIT entries and don't write.

    Returns:
        Dict with keys: written, skipped_cached, skipped_load_fail, oom_count,
        total_bytes, samples_processed.
    """
    oom_log_path = Path(cache_root) / hash16 / "oom_log.jsonl"

    stats = {
        "written": 0,
        "skipped_cached": 0,
        "skipped_load_fail": 0,
        "oom_count": 0,
        "total_bytes": 0,
        "samples_processed": 0,
    }

    limit = DRY_RUN_SAMPLE_LIMIT if dry_run else len(entries)
    entries_to_process = entries[:limit]

    # Sort by image_path for filesystem locality
    entries_to_process = sorted(
        entries_to_process,
        key=lambda e: str(e.get("image_path", "")),
    )

    # Batch iteration
    i = 0
    while i < len(entries_to_process):
        batch_entries = entries_to_process[i: i + batch_size]
        i += batch_size

        # Check which entries in this batch still need caching
        pending = []
        for entry in batch_entries:
            ds = str(entry.get("dataset", ""))
            sid = str(entry.get("sample_id", ""))
            key = _sanitize_sample_key(sid)
            if cache_entry_exists(cache_root, hash16, ds, key):
                stats["skipped_cached"] += 1
                continue
            pending.append((entry, ds, key))

        if not pending:
            continue

        # Load images for pending entries
        images = []
        valid_pending = []
        for entry, ds, key in pending:
            img_path = entry.get("image_path")
            tensor = _load_image_tensor(
                Path(str(img_path)), project_root, image_height, image_width
            )
            if tensor is None:
                stats["skipped_load_fail"] += 1
                continue
            images.append(tensor)
            valid_pending.append((entry, ds, key))

        if not images:
            continue

        # Stack into batch
        image_batch = torch.stack(images, dim=0)  # (B, 1, H, W)

        if dry_run:
            # In dry-run mode: run encoder on first real batch to measure sizes,
            # then count remaining samples for projection
            try:
                feature_map = encode_fn(image_batch)  # (B, 1280, h16, w16)
                h16 = feature_map.shape[2]
                w16 = feature_map.shape[3]
                seq_tokens = h16 * w16
                bytes_per_sample = seq_tokens * 1280 * 2  # bf16
                stats["written"] += len(valid_pending)
                stats["total_bytes"] += bytes_per_sample * len(valid_pending)
                stats["samples_processed"] += len(valid_pending)
            except torch.cuda.OutOfMemoryError:
                stats["oom_count"] += len(valid_pending)
            continue

        # Run encoder forward with OOM protection
        try:
            feature_map = encode_fn(image_batch)  # (B, 1280, h16, w16)
        except torch.cuda.OutOfMemoryError:
            stats["oom_count"] += len(valid_pending)
            oom_log_path.parent.mkdir(parents=True, exist_ok=True)
            with oom_log_path.open("a") as fh:
                for entry, ds, key in valid_pending:
                    fh.write(json.dumps({"sample_id": entry.get("sample_id"), "oom": True}) + "\n")
            continue

        h16 = feature_map.shape[2]
        w16 = feature_map.shape[3]

        # Write per-sample files
        for b_idx, (entry, ds, key) in enumerate(valid_pending):
            tensor = feature_map[b_idx].cpu().to(torch.bfloat16)
            flat = tensor.flatten(1).transpose(0, 1)  # (h16*w16, 1280) = (seq_tokens, 1280)
            p = write_cache_entry(cache_root, hash16, ds, key, flat, h16=h16, w16=w16)
            stats["written"] += 1
            stats["total_bytes"] += p.stat().st_size
            stats["samples_processed"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/data/manifests/token_manifest_stage3.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/full_radio_stage2_systems_v2/"
                                     "stage2-radio-systems-polyphonic_best.pt"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/encoder"))
    parser.add_argument("--preproc-cfg", type=Path, default=Path("configs/preproc_stage3.yaml"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dry-run", action="store_true",
                        help=f"Process only first {DRY_RUN_SAMPLE_LIMIT} samples; print disk projection.")
    parser.add_argument("--ignore-git-sha", action="store_true",
                        help="Omit git HEAD SHA from cache hash (for CI environments).")
    args = parser.parse_args()

    t0 = time.time()

    # Load preprocessing config
    with args.preproc_cfg.open() as fh:
        preproc_cfg = yaml.safe_load(fh)
    print(f"[builder] preproc_cfg: {preproc_cfg}", flush=True)

    # Get git HEAD SHA
    git_head_sha: Optional[str] = None
    if not args.ignore_git_sha:
        try:
            git_head_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip()
            print(f"[builder] git HEAD SHA: {git_head_sha}", flush=True)
        except Exception as exc:
            print(f"[builder] WARNING: could not get git HEAD SHA: {exc}. Use --ignore-git-sha to suppress.", flush=True)
            raise

    # Compute cache hash
    print(f"[builder] computing cache hash from {args.checkpoint}...", flush=True)
    hash16 = compute_cache_hash(
        args.checkpoint, preproc_cfg, "c-radio_v4-h", git_head_sha=git_head_sha
    )
    print(f"[builder] cache hash: {hash16}", flush=True)
    print(f"[builder] cache directory: {args.cache_root / hash16}", flush=True)

    # Pre-flight disk check
    free_bytes = shutil.disk_usage(args.cache_root.parent if not args.cache_root.exists() else args.cache_root).free
    print(f"[builder] free disk: {free_bytes / 1e9:.1f} GB", flush=True)

    # Load manifest
    entries = _load_manifest_entries(args.manifest, cached_only=True)
    print(f"[builder] cached-tier entries: {len(entries)}", flush=True)

    if args.dry_run:
        print(f"[builder] DRY RUN: processing first {DRY_RUN_SAMPLE_LIMIT} entries", flush=True)

    # Load model
    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig
    print(f"[builder] loading checkpoint: {args.checkpoint}", flush=True)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    # Strip compile wrapper prefix if present
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    config = RadioStageBConfig()
    model = RadioStageB(config)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[builder] load_state_dict: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    device = torch.device(args.device)
    model.encoder.to(device)
    print(f"[builder] encoder on {device}", flush=True)

    def encode_fn(image_batch_cpu: torch.Tensor) -> torch.Tensor:
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Expand grayscale to 3-channel
            if image_batch_cpu.shape[1] == 1:
                image_batch_cpu = image_batch_cpu.repeat(1, 3, 1, 1)
            batch_gpu = image_batch_cpu.to(device)
            feat = model.encoder(batch_gpu)  # (B, 1280, h16, w16)
            return feat.cpu().to(torch.bfloat16)

    stats = _build_cache_for_entries(
        entries=entries,
        cache_root=args.cache_root,
        hash16=hash16,
        encode_fn=encode_fn,
        project_root=ROOT,
        image_height=preproc_cfg.get("image_height", 250),
        image_width=preproc_cfg.get("image_width", 2500),
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    elapsed = time.time() - t0
    print(f"\n[builder] === {'DRY RUN ' if args.dry_run else ''}COMPLETE ===", flush=True)
    print(f"[builder] entries_total:       {len(entries)}", flush=True)
    print(f"[builder] samples_processed:   {stats['samples_processed']}", flush=True)
    print(f"[builder] written:             {stats['written']}", flush=True)
    print(f"[builder] skipped_cached:      {stats['skipped_cached']}", flush=True)
    print(f"[builder] skipped_load_fail:   {stats['skipped_load_fail']}", flush=True)
    print(f"[builder] oom_count:           {stats['oom_count']}", flush=True)
    print(f"[builder] total_bytes_sampled: {stats['total_bytes'] / 1e9:.3f} GB", flush=True)
    print(f"[builder] elapsed_sec:         {elapsed:.1f}", flush=True)

    if args.dry_run and stats["samples_processed"] > 0:
        per_sample_bytes = stats["total_bytes"] / stats["samples_processed"]
        projected_total = per_sample_bytes * len(entries)
        with_overhead = projected_total * 1.5
        print(f"\n[builder] === DISK PROJECTION ===", flush=True)
        print(f"[builder] per_sample_bytes:    {per_sample_bytes / 1e6:.2f} MB", flush=True)
        print(f"[builder] projected_total:     {projected_total / 1e12:.3f} TB ({projected_total / 1e9:.1f} GB)", flush=True)
        print(f"[builder] with_1.5x_overhead:  {with_overhead / 1e12:.3f} TB ({with_overhead / 1e9:.1f} GB)", flush=True)
        print(f"[builder] free_disk:           {free_bytes / 1e9:.1f} GB", flush=True)
        if with_overhead > free_bytes:
            print(f"[builder] WARNING: projected size EXCEEDS free disk. Stop and reassess.", flush=True)
        elif projected_total > 2e12:
            print(f"[builder] WARNING: projected total > 2 TB. Review spec §0a sizing table.", flush=True)
        elif projected_total > 1e12:
            print(f"[builder] CAUTION: 1 TB – 2 TB band. Consider dropping primus from cache.", flush=True)
        elif projected_total > 5e11:
            print(f"[builder] INFO: 500 GB – 1 TB. Verify free disk; proceed if headroom exists.", flush=True)
        else:
            print(f"[builder] INFO: ≤ 500 GB. Proceed with full cache build.", flush=True)
        return 0

    # Write metadata for full build
    if not args.dry_run:
        write_cache_metadata(args.cache_root, hash16, {
            "encoder_weights_path": str(args.checkpoint),
            "preproc_cfg": preproc_cfg,
            "radio_arch_version": "c-radio_v4-h",
            "git_head_sha": git_head_sha,
            "hash16": hash16,
            "hidden_dim": 1280,
            "dtype": "bfloat16",
            "sample_count": stats["written"],
            "total_bytes": stats["total_bytes"],
            "oom_count": stats["oom_count"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        print(f"[builder] metadata written to {args.cache_root / hash16 / 'metadata.json'}", flush=True)

    return 0 if stats["oom_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run resumability test to verify it passes**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/test_encoder_cache.py::test_builder_skips_already_cached_entries -v`
Expected: PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/build_encoder_cache.py tests/data/test_encoder_cache.py
git commit -m "feat(scripts): build_encoder_cache.py with dry-run and resumability"
```

> **Review:** Confirm `_build_cache_for_entries` is a public function (importable from tests). Confirm dry-run mode does NOT write any `.pt` files. Confirm OOM handler writes to `oom_log.jsonl` and continues (does not crash). Confirm `--ignore-git-sha` flag is tested indirectly via `test_compute_cache_hash_ignore_git_sha_is_stable`.

---

### Task 6: Phase 0a — Dry-run on GPU box + disk gate decision (operational)

**Files:**
- No new code files — pure operational task on GPU box.

**Why this task:** The dry-run is a hard gate. The disk projection it produces determines which path in the spec's sizing table applies before a single full cache entry is committed to disk.

- [ ] **Step 1: Push branch to GPU box**

Run:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; git pull origin feat/stage3-encoder-cache"'
```
Expected: GPU box is at the same HEAD as local.

- [ ] **Step 2: Run dry-run (1,000-sample sizing measurement)**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe scripts\build_encoder_cache.py --manifest src\data\manifests\token_manifest_stage3.jsonl --checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt --cache-root data\cache\encoder --batch-size 8 --device cuda --dry-run 2>&1 | Tee-Object -FilePath logs\encoder_cache_dryrun_2026-05-08.log"'
```

Expected output includes lines like:
```
[builder] cache hash: <16 hex chars>
[builder] cached-tier entries: 215985
[builder] DRY RUN: processing first 1000 entries
[builder] per_sample_bytes:    X.XX MB
[builder] projected_total:     X.XXX TB (XXXX.X GB)
[builder] with_1.5x_overhead:  X.XXX TB (XXXX.X GB)
[builder] free_disk:           XXXX.X GB
```

- [ ] **Step 3: Read the dry-run log and record measurements**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\encoder_cache_dryrun_2026-05-08.log\" | Select-String -Pattern \"per_sample|projected|overhead|free_disk|cache hash\""'
```

Record:
- `per_sample_bytes`: actual median bytes per sample (not theoretical)
- `projected_total` (TB)
- `with_1.5x_overhead` (TB)
- `free_disk` (GB)
- `cache_hash`: 16-char hex string (needed for all subsequent operations)

- [ ] **Step 4: Apply spec §0a sizing table to decide next step**

| Measured projected_total | Action |
|---|---|
| ≤ 500 GB | Proceed with full build at bf16. No changes. |
| 500 GB – 1 TB | Verify free_disk ≥ projected_total × 1.5 + 200 GB headroom. If yes, proceed. |
| 1 TB – 2 TB | Evaluate: (a) proceed if free_disk sufficient, or (b) drop primus from cache (reduces by ~40%). Record decision here before proceeding to Task 7. |
| > 2 TB | STOP. Pivot to no-caching design. Do not proceed to Task 7 without explicit user sign-off. |

Record decision inline (edit this task's step 4 with the actual numbers and decision before marking done).

> **Review:** Confirm `cache_hash` is recorded. Confirm the sizing table decision is documented in this step before marking task done. If projected > 2 TB, halt the plan and escalate to user before any further steps.

---

### Task 7: Full cache build on GPU box (operational)

**Files:**
- No new code — operational. Output: `data\cache\encoder\<hash16>\` directory tree + `metadata.json`.

**Why this task:** Builds the complete on-disk cache for all 215,985 cached-tier samples. Takes several hours. Must complete before the dataset extension (Task 9) can be tested end-to-end.

**Prerequisite:** Task 6 Step 4 sizing gate must have passed (projected ≤ 2 TB, free disk sufficient).

- [ ] **Step 1: Verify free disk before starting**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-PSDrive C | Select-Object Used, Free"'
```
Expected: Free (in GB after unit conversion) ≥ `projected_total × 1.5 + 200 GB`. If not, stop and investigate disk.

- [ ] **Step 2: Launch full cache build as background job**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; Start-Job -Name encoder_cache_build -ScriptBlock { cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe scripts\build_encoder_cache.py --manifest src\data\manifests\token_manifest_stage3.jsonl --checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt --cache-root data\cache\encoder --batch-size 8 --device cuda 2>&1 | Tee-Object -FilePath logs\encoder_cache_fullbuild_2026-05-08.log }"'
```

Monitor progress:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Job -Name encoder_cache_build | Select-Object State, HasMoreData"'
ssh 10.10.1.29 'powershell -NoProfile -Command "Receive-Job -Name encoder_cache_build -Keep 2>&1 | Select-Object -Last 10"'
```

Or monitor the log file:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\encoder_cache_fullbuild_2026-05-08.log\" -Wait -Last 5"'
```

- [ ] **Step 3: After completion, verify final stats**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\encoder_cache_fullbuild_2026-05-08.log\" | Select-String -Pattern \"written|skipped|oom|total_bytes|COMPLETE\""'
```

Expected:
- `written`: close to 215,985 (minus any skipped_load_fail or oom_count)
- `oom_count`: ideally 0; if > 0, run mop-up pass (Step 4)
- `total_bytes`: within ±5% of Phase 0a dry-run projection × 215,985

- [ ] **Step 4: Mop up any OOM'd samples (if oom_count > 0)**

If `oom_count > 0`, run a second pass with `--batch-size 1`:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe scripts\build_encoder_cache.py --manifest src\data\manifests\token_manifest_stage3.jsonl --checkpoint checkpoints\full_radio_stage2_systems_v2\stage2-radio-systems-polyphonic_best.pt --cache-root data\cache\encoder --batch-size 1 --device cuda 2>&1 | Tee-Object -FilePath logs\encoder_cache_mopup_2026-05-08.log"'
```
The builder's resume logic (skip existing entries) means this only re-processes the OOM'd samples. Expected: `written` ≈ original `oom_count`; new `oom_count` = 0.

- [ ] **Step 5: Read and verify metadata.json**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\data\cache\encoder\<hash16>\metadata.json\""'
```
(Replace `<hash16>` with the actual hash from Task 6 Step 3.)

Expected: JSON with `sample_count` close to 215,985, `dtype: "bfloat16"`, `hidden_dim: 1280`.

- [ ] **Step 6: Verify total disk size matches projection ±5%**

Run:
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "(Get-ChildItem -Recurse \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\data\cache\encoder\<hash16>\" | Measure-Object -Property Length -Sum).Sum / 1GB"'
```

Expected: result in GB is within ±5% of `dry_run_per_sample_bytes × 215,985 / 1e9`.

> **Review:** Confirm `sample_count` in metadata.json is within 1% of 215,985. Confirm `oom_count` is 0 after mop-up. Confirm total disk size matches projection ±5% (Phase 0 gate criterion #2).

---

### Task 8: Tier-grouped sampler (TDD)

**Files:**
- Create: `src/train/tier_sampler.py`
- Create: `tests/train/test_tier_grouped_sampler.py`

**Why this task:** The sampler guarantees cached and live batches are never mixed, which is required because they use different batch sizes (`b_cached` vs. `b_live=2`) and different forward paths. Without this guarantee the collate_fn would receive mixed batches and fail.

- [ ] **Step 1: Write failing tests**

Create `tests/train/test_tier_grouped_sampler.py`:

```python
"""Tests for src/train/tier_sampler.py::build_tier_grouped_sampler."""
from __future__ import annotations

import random
from collections import Counter

import pytest
import torch


CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
LIVE_DATASETS = {"cameraprimus_systems"}


def _make_mock_entries(n_cached: int, n_live: int) -> list[dict]:
    entries = []
    for i in range(n_cached):
        ds = list(CACHED_DATASETS)[i % len(CACHED_DATASETS)]
        entries.append({"dataset": ds, "sample_id": f"cached_{i}"})
    for i in range(n_live):
        entries.append({"dataset": "cameraprimus_systems", "sample_id": f"live_{i}"})
    return entries


def test_all_batches_are_tier_pure() -> None:
    """Every batch returned by the sampler must contain samples from exactly one tier."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=900, n_live=100)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=200,
        b_cached=8,
        b_live=2,
        seed=42,
    )

    for batch_idx, batch in enumerate(batches):
        tiers = set()
        for sample_idx in batch:
            ds = entries[sample_idx]["dataset"]
            if ds in CACHED_DATASETS:
                tiers.add("cached")
            else:
                tiers.add("live")
        assert len(tiers) == 1, (
            f"Batch {batch_idx} mixes tiers: {tiers}. "
            f"sample datasets: {[entries[i]['dataset'] for i in batch]}"
        )


def test_cached_batch_size_is_b_cached() -> None:
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=900, n_live=100)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=200,
        b_cached=8,
        b_live=2,
        seed=0,
    )
    for batch in batches:
        ds = entries[batch[0]]["dataset"]
        tier = "cached" if ds in CACHED_DATASETS else "live"
        expected_bs = 8 if tier == "cached" else 2
        assert len(batch) == expected_bs, (
            f"Batch has {len(batch)} samples but expected {expected_bs} for tier={tier}"
        )


def test_ratio_approximately_90_10() -> None:
    """Long-run cached batch fraction should be 90% ±5%."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=9000, n_live=1000)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=1000,
        b_cached=8,
        b_live=2,
        seed=7,
    )
    n_cached_batches = sum(
        1 for b in batches if entries[b[0]]["dataset"] in CACHED_DATASETS
    )
    frac = n_cached_batches / len(batches)
    assert 0.85 <= frac <= 0.95, f"Cached batch fraction {frac:.3f} outside 85–95% band"


def test_returns_list_of_lists() -> None:
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=90, n_live=10)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=50,
        b_cached=4,
        b_live=2,
        seed=1,
    )
    assert isinstance(batches, list)
    assert all(isinstance(b, list) for b in batches)
    assert all(isinstance(idx, int) for b in batches for idx in b)


def test_indices_are_valid() -> None:
    """All returned indices must be in [0, len(entries))."""
    from src.train.tier_sampler import build_tier_grouped_sampler

    entries = _make_mock_entries(n_cached=100, n_live=20)
    batches = build_tier_grouped_sampler(
        entries=entries,
        cached_datasets=CACHED_DATASETS,
        live_datasets=LIVE_DATASETS,
        cached_ratio=0.90,
        total_batches=50,
        b_cached=4,
        b_live=2,
        seed=99,
    )
    n = len(entries)
    for batch in batches:
        for idx in batch:
            assert 0 <= idx < n, f"Index {idx} out of range [0, {n})"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/train/test_tier_grouped_sampler.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.train.tier_sampler'`.

- [ ] **Step 3: Implement `src/train/tier_sampler.py`**

Create `src/train/tier_sampler.py`:

```python
"""Tier-grouped batch sampler for Stage 3 two-tier dataloader.

Guarantees that each batch is 100% from one tier (cached or live). This is
required because cached batches use b_cached (8 or 16) while live batches
use b_live=2, and the model forward path dispatches on the tier key.

The sampler pre-computes a list of batched index lists, interleaved at the
specified cached_ratio. Indices are drawn with replacement within each tier.
"""
from __future__ import annotations

import random


def build_tier_grouped_sampler(
    entries: list[dict],
    cached_datasets: set[str],
    live_datasets: set[str],
    cached_ratio: float,
    total_batches: int,
    b_cached: int,
    b_live: int,
    seed: int = 0,
) -> list[list[int]]:
    """Build a list of tier-pure batched index lists.

    Args:
        entries: Full dataset entries list (same order as dataset.entries).
        cached_datasets: Set of dataset names that are in the cached tier.
        live_datasets: Set of dataset names that are in the live tier.
        cached_ratio: Fraction of batches that should be from the cached tier
            (e.g. 0.90 for 90% cached / 10% live).
        total_batches: Total number of batches to generate.
        b_cached: Batch size for cached-tier batches.
        b_live: Batch size for live-tier batches.
        seed: Random seed for reproducibility.

    Returns:
        A list of total_batches lists. Each inner list contains integer indices
        into `entries`. All indices in a given inner list come from the same tier.
    """
    rng = random.Random(seed)

    # Partition entry indices by tier
    cached_indices = [
        i for i, e in enumerate(entries)
        if e.get("dataset") in cached_datasets
    ]
    live_indices = [
        i for i, e in enumerate(entries)
        if e.get("dataset") in live_datasets
    ]

    if not cached_indices:
        raise ValueError(
            f"build_tier_grouped_sampler: no entries found for cached tier. "
            f"cached_datasets={cached_datasets}"
        )
    if not live_indices:
        raise ValueError(
            f"build_tier_grouped_sampler: no entries found for live tier. "
            f"live_datasets={live_datasets}"
        )

    # Determine how many cached vs live batches to generate
    n_cached_batches = round(total_batches * cached_ratio)
    n_live_batches = total_batches - n_cached_batches

    # Generate per-tier batches (with replacement)
    def _draw_batches(indices: list[int], batch_size: int, n_batches: int) -> list[list[int]]:
        batches = []
        for _ in range(n_batches):
            batch = [rng.choice(indices) for _ in range(batch_size)]
            batches.append(batch)
        return batches

    cached_batches = _draw_batches(cached_indices, b_cached, n_cached_batches)
    live_batches = _draw_batches(live_indices, b_live, n_live_batches)

    # Interleave cached and live batches in proportion (shuffle by tier assignment)
    # Deterministic shuffle: alternate with occasional live batch
    tier_sequence: list[str] = (["cached"] * n_cached_batches) + (["live"] * n_live_batches)
    rng.shuffle(tier_sequence)

    cached_iter = iter(cached_batches)
    live_iter = iter(live_batches)
    result: list[list[int]] = []
    for tier in tier_sequence:
        if tier == "cached":
            result.append(next(cached_iter))
        else:
            result.append(next(live_iter))

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/train/test_tier_grouped_sampler.py -v`
Expected: all 5 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/train/tier_sampler.py tests/train/test_tier_grouped_sampler.py
git commit -m "feat(train): tier-grouped batch sampler for Stage 3 two-tier dataloader"
```

> **Review:** Confirm every batch in test output contains only one tier (verified by `test_all_batches_are_tier_pure`). Confirm cached batch size is always exactly `b_cached` and live batch size always exactly `b_live`. Confirm ratio is 90±5% over 1000 batches.

---

### Task 9: Cached dataset extension + tier-aware collate_fn (TDD)

**Files:**
- Modify: `src/train/train.py` — extend `StageBDataset.__getitem__` and `StageBDataset.collate_fn`
- Create: `tests/train/test_cached_dataset.py`

**Why this task:** The dataset's `__getitem__` is what the dataloader calls at training time. Adding the cached path here completes the data pipeline so that cached batches don't load images or run the encoder.

The current `StageBDataset.__getitem__` is at `train.py:597`. The current `collate_fn` is at `train.py:676`.

**Implementer note:** If line numbers have shifted, grep for the anchor strings:
- `__getitem__`: grep for `def __getitem__(self, idx: int)`
- `collate_fn`: grep for `def collate_fn(samples:`

- [ ] **Step 1: Write failing tests**

Create `tests/train/test_cached_dataset.py`:

```python
"""Tests for the cached-path extension to StageBDataset."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/train/test_cached_dataset.py -v`
Expected: FAIL — `StageBDataset.__init__` does not yet accept `cache_root` / `cache_hash16` kwargs; `__getitem__` does not return `tier` key.

- [ ] **Step 3: Modify `StageBDataset` in `src/train/train.py`**

The `__init__` signature is at `train.py:547`. The `__getitem__` is at `train.py:597`. The `collate_fn` is at `train.py:676`.

Implementer: grep for the following anchor strings to locate lines if they've shifted:
- `__init__`: `def __init__(self, stage: "StageTrainingConfig"`
- `__getitem__`: `def __getitem__(self, idx: int) -> "Dict[str, torch.Tensor]":`
- `collate_fn`: `def collate_fn(samples: "List[Dict[str, torch.Tensor]]")`

**3a. Add `cache_root` and `cache_hash16` parameters to `__init__`:**

Find the `__init__` signature (currently ends with `rng_seed: Optional[int] = None,`) and add two new parameters:

```python
    def __init__(
        self,
        stage: "StageTrainingConfig",
        grouped_entries: "Dict[Tuple[str, str], List[Dict[str, object]]]",
        *,
        split: str = "train",
        project_root: "Path",
        image_height: int = 250,
        image_width: int = 2500,
        max_sequence_length: int = 512,
        vocab=None,
        augment: bool = True,
        rng_seed: Optional[int] = None,
        cache_root: "Optional[Path]" = None,
        cache_hash16: "Optional[str]" = None,
    ) -> None:
```

Add these two lines at the end of `__init__`'s body (after `self._rng = random.Random(rng_seed)`):
```python
        self.cache_root = Path(cache_root) if cache_root is not None else None
        self.cache_hash16 = cache_hash16
```

**3b. Extend `__getitem__` to handle the cached tier:**

Replace the current `__getitem__` method body. The full replacement (locate by `def __getitem__(self, idx: int)`):

```python
    def __getitem__(self, idx: int) -> "Dict[str, object]":
        import torch

        entry = self.entries[idx]
        vocab = self._vocab
        pad_id = vocab.token_to_id["<pad>"]
        bos_id = vocab.token_to_id["<bos>"]
        eos_id = vocab.token_to_id["<eos>"]
        measure_end_id = vocab.token_to_id.get("<measure_end>")

        sample_id = str(entry.get("sample_id", f"<idx:{idx}>"))
        dataset_name = str(entry.get("dataset", ""))

        # Determine tier
        _CACHED_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
        is_cached_tier = dataset_name in _CACHED_DATASETS

        # --- Cached path: load pre-computed encoder features from disk ---
        if is_cached_tier and self.cache_root is not None and self.cache_hash16 is not None:
            from src.data.encoder_cache import _sanitize_sample_key, read_cache_entry
            key = _sanitize_sample_key(sample_id)
            encoder_hidden, h16, w16 = read_cache_entry(
                self.cache_root, self.cache_hash16, dataset_name, key
            )
            # Token encode (same as live path)
            sequence = entry.get("token_sequence", [])
            if not isinstance(sequence, list) or not sequence:
                sequence = ["<bos>", "<eos>"]
            try:
                token_ids = vocab.encode(sequence, strict=True)
            except KeyError:
                token_ids = [bos_id, eos_id]
            if len(token_ids) < 2:
                token_ids = [bos_id, eos_id]
            if len(token_ids) > self.max_sequence_length:
                truncated = token_ids[: self.max_sequence_length - 1]
                if measure_end_id is not None:
                    last_me = -1
                    for _i in range(len(truncated) - 1, -1, -1):
                        if truncated[_i] == measure_end_id:
                            last_me = _i
                            break
                    if last_me > 0:
                        token_ids = truncated[: last_me + 1] + [eos_id]
                    else:
                        token_ids = truncated + [eos_id]
                else:
                    token_ids = truncated + [eos_id]
            contour_target = _derive_pitch_contour(sequence)
            seq_len = self.max_sequence_length - 1
            input_ids = token_ids[:-1]
            label_ids = token_ids[1:]
            if not input_ids:
                input_ids = [bos_id]
                label_ids = [eos_id]
            input_pad = [pad_id] * max(0, seq_len - len(input_ids))
            label_pad = [-100] * max(0, seq_len - len(label_ids))
            decoder_inputs = (input_ids + input_pad)[:seq_len]
            labels = (label_ids + label_pad)[:seq_len]
            return {
                "tier": "cached",
                "encoder_hidden": encoder_hidden,  # (seq_tokens, 1280) bf16
                "_h16": h16,
                "_w16": w16,
                "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "contour_targets": torch.tensor(contour_target, dtype=torch.long),
            }

        # --- Live path: full image load + augment pipeline ---
        try:
            image_tensor, content_width = _load_entry_image_tensor(
                entry,
                project_root=self.project_root,
                height=self.image_height,
                max_width=self.image_width,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            import torch as _torch
            print(f"[StageBDataset] skipping {sample_id}: {exc}", file=sys.stderr)
            image_tensor = _torch.zeros(1, self.image_height, self.image_width, dtype=_torch.float32)
            content_width = self.image_width

        sequence = entry.get("token_sequence", [])
        if not isinstance(sequence, list) or not sequence:
            sequence = ["<bos>", "<eos>"]
        try:
            token_ids = vocab.encode(sequence, strict=True)
        except KeyError:
            token_ids = [bos_id, eos_id]
        if len(token_ids) < 2:
            token_ids = [bos_id, eos_id]
        if len(token_ids) > self.max_sequence_length:
            truncated = token_ids[: self.max_sequence_length - 1]
            if measure_end_id is not None:
                last_me = -1
                for _i in range(len(truncated) - 1, -1, -1):
                    if truncated[_i] == measure_end_id:
                        last_me = _i
                        break
                if last_me > 0:
                    token_ids = truncated[: last_me + 1] + [eos_id]
                else:
                    token_ids = truncated + [eos_id]
            else:
                token_ids = truncated + [eos_id]

        if self.augment:
            image_tensor = _apply_online_augmentations(image_tensor.unsqueeze(0), self._rng).squeeze(0)

        contour_target = _derive_pitch_contour(sequence)
        input_ids = token_ids[:-1]
        label_ids = token_ids[1:]
        if not input_ids:
            input_ids = [bos_id]
            label_ids = [eos_id]
        seq_len = self.max_sequence_length - 1
        input_pad = [pad_id] * max(0, seq_len - len(input_ids))
        label_pad = [-100] * max(0, seq_len - len(label_ids))
        decoder_inputs = (input_ids + input_pad)[:seq_len]
        labels = (label_ids + label_pad)[:seq_len]

        return {
            "tier": "live",
            "images": image_tensor,
            "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "contour_targets": torch.tensor(contour_target, dtype=torch.long),
            "content_widths": torch.tensor(int(content_width), dtype=torch.long),
        }
```

**3c. Replace `collate_fn` to be tier-aware:**

Replace the current `collate_fn` body (locate by `def collate_fn(samples:`):

```python
    @staticmethod
    def collate_fn(samples: "List[Dict[str, object]]") -> "Dict[str, object]":
        """Stack a list of per-sample dicts into a batched dict.

        All samples must be from the same tier (cached or live). Mixed-tier
        batches raise ValueError — the tier-grouped sampler prevents them.
        """
        import torch

        tiers = {s["tier"] for s in samples}
        if len(tiers) > 1:
            raise ValueError(
                f"collate_fn received mixed tiers: {tiers}. "
                "The tier-grouped sampler must guarantee tier-pure batches."
            )
        tier = tiers.pop()

        decoder_inputs = torch.stack([s["decoder_inputs"] for s in samples], dim=0)
        labels = torch.stack([s["labels"] for s in samples], dim=0)
        contour_targets = torch.stack([s["contour_targets"] for s in samples], dim=0)

        if tier == "cached":
            encoder_hidden = torch.stack([s["encoder_hidden"] for s in samples], dim=0)
            h16 = samples[0]["_h16"]
            w16 = samples[0]["_w16"]
            return {
                "tier": "cached",
                "encoder_hidden": encoder_hidden,  # (B, seq_tokens, 1280)
                "_h16": h16,
                "_w16": w16,
                "decoder_inputs": decoder_inputs,
                "labels": labels,
                "contour_targets": contour_targets,
            }
        else:
            images = torch.stack([s["images"] for s in samples], dim=0)
            content_widths = torch.stack([s["content_widths"] for s in samples], dim=0)
            return {
                "tier": "live",
                "images": images,
                "decoder_inputs": decoder_inputs,
                "labels": labels,
                "contour_targets": contour_targets,
                "content_widths": content_widths,
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/train/test_cached_dataset.py -v`
Expected: all 6 tests PASS.

- [ ] **Step 5: Run full test suite**

Run: `cd /home/ari/work/Clarity-OMR-Train-RADIO && python -m pytest tests/data/ tests/models/ tests/train/ -q`
Expected: all green; no regressions in existing `tests/train/` tests.

- [ ] **Step 6: Commit**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add src/train/train.py tests/train/test_cached_dataset.py
git commit -m "feat(train): StageBDataset cached/live tier dispatch and tier-aware collate_fn"
```

> **Review:** Confirm existing callers of `StageBDataset(...)` still work (all new params have defaults). Confirm `collate_fn` raises on mixed tiers. Confirm live path still returns `"content_widths"` key (backward compat). Confirm cached path returns `"_h16"` and `"_w16"` which the model's `forward()` uses for reshape.

---

### Task 10: Correctness validation script (operational + TDD)

**Files:**
- Create: `scripts/validate_cache_correctness.py`

**Why this task:** This is the Phase 0d exit criterion #3. Must demonstrate ≤ 1e-3 max absolute diff between cached and live forward passes on 100 samples from the actual GPU-box cache.

- [ ] **Step 1: Create the validation script**

Create `scripts/validate_cache_correctness.py`:

```python
#!/usr/bin/env python3
"""Validate encoder cache correctness against live encoder forward.

For N random cached samples:
  1. Load the cached encoder features from disk.
  2. Run live encoder forward on the same image.
  3. Compare the two tensors element-wise.
  4. Report max absolute diff, mean absolute diff, and pass/fail.

Phase 0 exit criterion: max abs diff ≤ 1e-3 on 100 samples.

Usage:
    python scripts/validate_cache_correctness.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --hash16 <16-char-hash> \\
        --n-samples 100 \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import yaml

from src.data.encoder_cache import _sanitize_sample_key, read_cache_entry

CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}


def _load_image(image_path: Path, project_root: Path, image_height: int, image_width: int):
    import torchvision.transforms.functional as TF
    from PIL import Image

    full_path = project_root / image_path if not Path(image_path).is_absolute() else Path(image_path)
    img = Image.open(full_path).convert("L")
    scale = image_height / img.height
    new_w = min(int(img.width * scale), image_width)
    img = img.resize((new_w, image_height), Image.LANCZOS)
    canvas = Image.new("L", (image_width, image_height), color=255)
    canvas.paste(img, (0, 0))
    return TF.to_tensor(canvas)  # (1, H, W) float32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/data/manifests/token_manifest_stage3.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/full_radio_stage2_systems_v2/"
                                     "stage2-radio-systems-polyphonic_best.pt"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/encoder"))
    parser.add_argument("--hash16", type=str, required=True,
                        help="16-char cache hash from build_encoder_cache.py output")
    parser.add_argument("--preproc-cfg", type=Path, default=Path("configs/preproc_stage3.yaml"))
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    args = parser.parse_args()

    with args.preproc_cfg.open() as fh:
        preproc_cfg = yaml.safe_load(fh)
    image_height = preproc_cfg.get("image_height", 250)
    image_width = preproc_cfg.get("image_width", 2500)

    # Load manifest, filter to cached tier with valid image_path
    entries = []
    with args.manifest.open() as fh:
        for line in fh:
            e = json.loads(line.strip())
            if e.get("dataset") in CACHED_TIER_DATASETS and e.get("image_path"):
                entries.append(e)
    print(f"[validate] cached entries with images: {len(entries)}", flush=True)

    rng = random.Random(args.seed)
    sample_entries = rng.sample(entries, min(args.n_samples, len(entries)))
    print(f"[validate] sampling {len(sample_entries)} entries", flush=True)

    # Load model
    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model = RadioStageB(RadioStageBConfig())
    model.load_state_dict(state_dict, strict=False)
    model.encoder.eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    device = torch.device(args.device)
    model.encoder.to(device)

    # Run validation
    max_diffs = []
    mean_diffs = []
    failed = []

    for i, entry in enumerate(sample_entries):
        sid = entry["sample_id"]
        ds = entry["dataset"]
        key = _sanitize_sample_key(sid)

        # Load cached tensor
        try:
            cached_tensor, h16, w16 = read_cache_entry(args.cache_root, args.hash16, ds, key)
        except Exception as exc:
            print(f"[validate] MISS {sid}: {exc}", flush=True)
            failed.append({"sample_id": sid, "reason": "cache_miss"})
            continue

        # Run live encoder on the same image
        try:
            img = _load_image(Path(entry["image_path"]), ROOT, image_height, image_width)
        except Exception as exc:
            print(f"[validate] IMG_FAIL {sid}: {exc}", flush=True)
            failed.append({"sample_id": sid, "reason": "image_load_fail"})
            continue

        img_batch = img.unsqueeze(0).to(device)
        if img_batch.shape[1] == 1:
            img_batch = img_batch.repeat(1, 3, 1, 1)

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            live_feat = model.encoder(img_batch)  # (1, 1280, h16, w16)

        live_flat = live_feat[0].cpu().to(torch.bfloat16).flatten(1).transpose(0, 1)  # (seq, 1280)
        cached_cpu = cached_tensor.cpu()

        # Shape must match
        if live_flat.shape != cached_cpu.shape:
            failed.append({"sample_id": sid, "reason": f"shape_mismatch live={live_flat.shape} cached={cached_cpu.shape}"})
            continue

        diff = (live_flat.float() - cached_cpu.float()).abs()
        max_d = diff.max().item()
        mean_d = diff.mean().item()
        max_diffs.append(max_d)
        mean_diffs.append(mean_d)

        status = "PASS" if max_d <= args.tolerance else "FAIL"
        if status == "FAIL":
            failed.append({"sample_id": sid, "max_diff": max_d})
        if i % 10 == 0:
            print(f"[validate] {i+1}/{len(sample_entries)} {status} max_diff={max_d:.2e} mean_diff={mean_d:.2e}", flush=True)

    print(f"\n[validate] === RESULTS ===", flush=True)
    print(f"[validate] samples_checked:  {len(max_diffs)}", flush=True)
    print(f"[validate] samples_failed:   {len(failed)}", flush=True)
    if max_diffs:
        print(f"[validate] max_diff_overall: {max(max_diffs):.4e}", flush=True)
        print(f"[validate] mean_diff_mean:   {sum(mean_diffs)/len(mean_diffs):.4e}", flush=True)
        overall_pass = max(max_diffs) <= args.tolerance
        print(f"[validate] PHASE 0 GATE:     {'PASS' if overall_pass else 'FAIL'} (tolerance={args.tolerance:.0e})", flush=True)
        return 0 if overall_pass else 1
    else:
        print(f"[validate] ERROR: no samples validated", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit the script**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/validate_cache_correctness.py
git commit -m "feat(scripts): validate_cache_correctness.py for Phase 0d exit criterion"
```

- [ ] **Step 3: Push and run on GPU box (operational)**

Push:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; git pull origin feat/stage3-encoder-cache"'
```

Run (replace `<hash16>` with the actual hash from Task 6):
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe scripts\validate_cache_correctness.py --hash16 <hash16> --n-samples 100 --device cuda 2>&1 | Tee-Object -FilePath logs\cache_correctness_2026-05-08.log"'
```

Expected final output:
```
[validate] PHASE 0 GATE: PASS (tolerance=1e-03)
```

- [ ] **Step 4: Read results**

```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\cache_correctness_2026-05-08.log\" | Select-String -Pattern \"GATE|max_diff|mean_diff|samples\""'
```

Record `max_diff_overall` and `PHASE 0 GATE` result. If FAIL: halt and investigate (check encoder is in eval mode, autocast is bf16, image preprocessing matches exactly between builder and validator).

> **Review:** Confirm `PHASE 0 GATE: PASS`. Confirm `max_diff_overall ≤ 1e-3`. Confirm `samples_checked == 100`. If any `shape_mismatch` entries, those indicate a positional-embedding interpolation inconsistency — escalate.

---

### Task 11: Push branch and sync GPU box for final throughput testing

**Files:**
- No new files.

- [ ] **Step 1: Push all commits**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
```
Expected: all commits from Tasks 0–10 on remote.

- [ ] **Step 2: Pull on GPU box**

```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; git pull origin feat/stage3-encoder-cache"'
```

- [ ] **Step 3: Run the existing test suite on GPU box to catch any environment differences**

```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe -m pytest tests\data\test_encoder_cache.py tests\models\test_radio_stage_b_cached.py tests\train\test_tier_grouped_sampler.py tests\train\test_cached_dataset.py -v 2>&1 | Tee-Object -FilePath logs\encoder_cache_tests_gpubox_2026-05-08.log"'
```

Expected: all tests pass. If any fail on the GPU box but pass locally, investigate environment differences (Python version, torch version).

> **Review:** Confirm GPU-box test run is all green before proceeding to throughput sweep.

---

### Task 12: Throughput + VRAM sweep (operational)

**Files:**
- Create: `scripts/measure_encoder_cache_throughput.py`

**Why this task:** Produces `b_cached` for Phase 1. Without this measurement the training config (Plan C) cannot be finalized. This is Phase 0 exit criterion #4 (dataloader throughput) and #5 (VRAM sweep).

- [ ] **Step 1: Create the throughput measurement script**

Create `scripts/measure_encoder_cache_throughput.py`:

```python
#!/usr/bin/env python3
"""Throughput and VRAM sweep for Stage 3 two-tier training shape.

Runs the cached dataloader at batch sizes 4, 8, 16, 32 and measures:
  - GPU VRAM usage (peak)
  - Step time (forward + backward, no optimizer)
  - Samples/second

Also validates the live path at b_live=2 alongside the chosen b_cached to
ensure VRAM doesn't overflow when interleaved.

Usage:
    python scripts/measure_encoder_cache_throughput.py \\
        --manifest src/data/manifests/token_manifest_stage3.jsonl \\
        --checkpoint checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt \\
        --cache-root data/cache/encoder \\
        --hash16 <hash16> \\
        --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
import yaml


CACHED_TIER_DATASETS = {"synthetic_systems", "grandstaff_systems", "primus_systems"}
LIVE_TIER_DATASETS = {"cameraprimus_systems"}
BATCH_SIZES_TO_TEST = [4, 8, 16, 32]
N_WARMUP_STEPS = 5
N_MEASURE_STEPS = 20
MAX_SEQ_LEN = 512


def _load_cached_entries(manifest: Path) -> list[dict]:
    entries = []
    with manifest.open() as fh:
        for line in fh:
            e = json.loads(line.strip())
            if e.get("dataset") in CACHED_TIER_DATASETS and e.get("image_path"):
                entries.append(e)
    return entries


def _load_live_entries(manifest: Path) -> list[dict]:
    entries = []
    with manifest.open() as fh:
        for line in fh:
            e = json.loads(line.strip())
            if e.get("dataset") in LIVE_TIER_DATASETS and e.get("image_path"):
                entries.append(e)
    return entries


def _measure_cached_forward(model, cache_root, hash16, entries, batch_size, device, n_steps):
    """Run cached forward + backward for n_steps, return (avg_step_sec, peak_vram_gb)."""
    import random
    from src.data.encoder_cache import _sanitize_sample_key, read_cache_entry

    rng = random.Random(42)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)

    times = []
    for step in range(n_steps):
        batch_entries = rng.choices(entries, k=batch_size)
        tensors, h16s, w16s = [], [], []
        for e in batch_entries:
            key = _sanitize_sample_key(e["sample_id"])
            t, h16, w16 = read_cache_entry(cache_root, hash16, e["dataset"], key)
            tensors.append(t)
            h16s.append(h16)
            w16s.append(w16)
        encoder_hidden = torch.stack(tensors, dim=0).to(device)
        tgt = torch.zeros(batch_size, MAX_SEQ_LEN - 1, dtype=torch.long, device=device)

        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.forward(
                cached_features=encoder_hidden,
                tgt=tgt,
                _h16=h16s[0],
                _w16=w16s[0],
            )
            loss = F.cross_entropy(
                out["logits"].reshape(-1, out["logits"].shape[-1]),
                tgt.reshape(-1).clamp(0),
            )
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)

    peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
    avg_sec = sum(times[N_WARMUP_STEPS:]) / max(1, len(times) - N_WARMUP_STEPS)
    return avg_sec, peak_gb


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("src/data/manifests/token_manifest_stage3.jsonl"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/full_radio_stage2_systems_v2/"
                                     "stage2-radio-systems-polyphonic_best.pt"))
    parser.add_argument("--cache-root", type=Path, default=Path("data/cache/encoder"))
    parser.add_argument("--hash16", type=str, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    from src.models.radio_stage_b import RadioStageB, RadioStageBConfig
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = payload.get("model_state_dict", payload)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model = RadioStageB(RadioStageBConfig())
    model.load_state_dict(state_dict, strict=False)
    # Freeze encoder; only trainable surface is decoder+bridge
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    model.to(device)
    model.train()

    cached_entries = _load_cached_entries(args.manifest)
    print(f"[sweep] cached entries: {len(cached_entries)}", flush=True)

    results = []
    print(f"\n{'batch':>5} {'avg_step_sec':>14} {'samples/sec':>12} {'peak_vram_gb':>13} {'vram%':>7}", flush=True)
    print("-" * 60, flush=True)

    for bs in BATCH_SIZES_TO_TEST:
        try:
            avg_sec, peak_gb = _measure_cached_forward(
                model, args.cache_root, args.hash16, cached_entries,
                batch_size=bs, device=device,
                n_steps=N_WARMUP_STEPS + N_MEASURE_STEPS,
            )
            total_vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9
            vram_pct = peak_gb / total_vram_gb * 100
            samples_per_sec = bs / avg_sec
            status = "OK" if vram_pct <= 80 else "OOM_RISK"
            results.append({
                "batch_size": bs, "avg_step_sec": avg_sec, "samples_per_sec": samples_per_sec,
                "peak_vram_gb": peak_gb, "vram_pct": vram_pct, "status": status,
            })
            print(f"{bs:>5} {avg_sec:>14.3f} {samples_per_sec:>12.1f} {peak_gb:>13.2f} {vram_pct:>6.1f}%  {status}", flush=True)
        except torch.cuda.OutOfMemoryError:
            print(f"{bs:>5} {'OOM':>14}", flush=True)
            results.append({"batch_size": bs, "status": "OOM"})

    # Determine b_cached recommendation
    passing = [r for r in results if r.get("status") == "OK"]
    if passing:
        b_cached = max(r["batch_size"] for r in passing)
        print(f"\n[sweep] RECOMMENDATION: b_cached={b_cached} (largest batch with VRAM ≤ 80%)", flush=True)
    else:
        b_cached = None
        print(f"\n[sweep] WARNING: no batch size passes VRAM ≤ 80% constraint", flush=True)

    print(f"\n[sweep] Phase 0 exit criteria check:", flush=True)
    print(f"[sweep]   Throughput gate: dataloader not bottleneck", flush=True)
    print(f"[sweep]   (Verify: step time is dominated by backward pass, not data load)", flush=True)
    print(f"[sweep]   b_cached recommendation for Phase 1: {b_cached}", flush=True)
    print(f"[sweep]   Record b_cached in handoff doc (Task 14).", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Commit the script**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add scripts/measure_encoder_cache_throughput.py
git commit -m "feat(scripts): throughput and VRAM sweep for Phase 0d exit criteria"
```

- [ ] **Step 3: Push and run on GPU box**

Push:
```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; git pull origin feat/stage3-encoder-cache"'
```

Run (replace `<hash16>`):
```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "cd \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\"; venv-cu132\Scripts\python.exe scripts\measure_encoder_cache_throughput.py --hash16 <hash16> --device cuda 2>&1 | Tee-Object -FilePath logs\throughput_sweep_2026-05-08.log"'
```

Expected output table (approximate values for RTX 5090):
```
batch  avg_step_sec  samples/sec  peak_vram_gb   vram%
------------------------------------------------------------
    4         0.200         20.0          8.50   10.6%  OK
    8         0.240         33.3         14.20   17.8%  OK
   16         0.320         50.0         24.50   30.6%  OK
   32         0.600         53.3         46.00   57.5%  OK
```

- [ ] **Step 4: Record results and b_cached recommendation**

```bash
ssh 10.10.1.29 'powershell -NoProfile -Command "Get-Content \"C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\logs\throughput_sweep_2026-05-08.log\" | Select-String -Pattern \"RECOMMENDATION|batch|PASS|OOM\""'
```

Record:
- `b_cached` recommendation (largest passing batch size)
- VRAM% at `b_cached`
- avg_step_sec at `b_cached`
- grad_accum_cached = `16 / b_cached` (for Phase 1 training config)

> **Review:** Confirm `b_cached` is determined. Confirm VRAM at `b_cached` is ≤ 80%. Confirm step times are consistent with the spec's ~0.25–0.5s projection for decoder-only cached steps.

---

### Task 13: Phase 0d gate review (user checkpoint)

**Files:**
- No new code.

**Why this task:** All five Phase 0 exit criteria must be explicitly verified before Plan C (Phase 1 training) can begin. This is a mandatory user review gate.

- [ ] **Step 1: Verify all five Phase 0 exit criteria**

Review each criterion against the evidence from prior tasks:

| # | Criterion | Task | Evidence |
|---|---|---|---|
| 1 | Disk math: measured ≤ 2 TB, GPU box NVMe has ≥ 200 GB headroom | Task 6 | dry-run log `projected_total` and `free_disk` values |
| 2 | Cache built: total size matches measurement ±5% | Task 7 Step 6 | PowerShell `Measure-Object` output vs. dry-run projection |
| 3 | Correctness: ≤ 1e-3 max abs diff on 100 samples | Task 10 Step 4 | `logs\cache_correctness_2026-05-08.log` `PHASE 0 GATE: PASS` |
| 4 | Throughput: dataloader not the bottleneck | Task 12 Step 4 | step time is backward-dominated, not I/O-dominated |
| 5 | Memory: `b_cached` chosen with VRAM ≤ 80% | Task 12 Step 4 | throughput sweep table |

- [ ] **Step 2: Record gate results and b_cached for Phase 1**

Fill in this table (edit this step with actual numbers before marking done):

```
Phase 0 Gate Results (2026-05-08)
----------------------------------
Disk math:     projected=X.X TB, free=X GB, headroom=X GB  [PASS/FAIL]
Cache built:   sample_count=X, total_size=X GB ±X%          [PASS/FAIL]
Correctness:   max_diff=X.Xe-X on 100 samples              [PASS/FAIL]
Throughput:    avg_step_sec=X.Xs at b_cached=X              [PASS/FAIL]
Memory:        VRAM=X.X% at b_cached=X                     [PASS/FAIL]

b_cached for Phase 1: X
grad_accum_cached:    X  (= 16 / b_cached)
cache_hash16:         <hash>
```

- [ ] **Step 3: If any gate fails — halt and diagnose**

Do NOT proceed to Task 14 (handoff) if any gate fails. Investigate root cause:
- Disk math fail: check if actual crop shapes are larger than estimated; may need to drop primus from cache.
- Cache size mismatch > 5%: check for OOM'd samples that weren't mop'd up; rerun mop-up with `--batch-size 1`.
- Correctness fail: check that builder and validator use identical preprocessing (same image_height/width, same pad convention, same autocast dtype).
- Throughput fail: profile with `torch.profiler` to determine whether bottleneck is I/O or compute.
- Memory fail: try next smaller batch size.

- [ ] **Step 4: Present gate results to user for explicit go-ahead**

This step requires user review. Do not auto-proceed. Present the filled-in table from Step 2 and wait for explicit approval before beginning Plan C (Phase 1).

> **Review:** All five rows in the gate table must say PASS. `b_cached` must be recorded. User must have explicitly approved Phase 1 launch.

---

### Task 14: Memory update + handoff

**Files:**
- Modify: `/home/ari/.claude/projects/-home-ari/memory/project_radio_stage3_design.md`
- Modify: `/home/ari/.claude/projects/-home-ari/memory/MEMORY.md`
- Create: `docs/superpowers/handoffs/2026-05-08-radio-stage3-phase0-complete-handoff.md`

**Why this task:** Persists the key Phase 0 outputs (cache hash, b_cached, gate results) so Phase 1 (Plan C) starts with complete context. Memory entries ensure future sessions don't re-derive already-known facts.

- [ ] **Step 1: Update memory entry for Stage 3 design**

Edit `/home/ari/.claude/projects/-home-ari/memory/project_radio_stage3_design.md`:
- Add a "Phase 0 complete" section recording: cache_hash16, b_cached, grad_accum_cached, cache root path, total cache size, correctness max_diff, throughput avg_step_sec.
- Note that `configs/preproc_stage3.yaml` is the authoritative preprocessing config for the cache.
- Note that DoRA adapters are encoder-side AND decoder-side; the cache captures frozen encoder+adapter output.

- [ ] **Step 2: Update MEMORY.md**

Edit `/home/ari/.claude/projects/-home-ari/memory/MEMORY.md`:
- Update the `[Stage 3 design committed]` entry to `[Stage 3 Phase 0 complete]`.
- Add a one-liner: `- [Stage 3 Phase 0 complete](project_radio_stage3_design.md) — encoder cache built; b_cached=X; cache at data/cache/encoder/<hash16>; Phase 1 ready pending user go-ahead`.

- [ ] **Step 3: Create handoff document**

Create `docs/superpowers/handoffs/2026-05-08-radio-stage3-phase0-complete-handoff.md`:

```markdown
# Stage 3 Phase 0 Complete Handoff

**Date:** 2026-05-08
**Branch:** feat/stage3-encoder-cache
**Plan:** docs/superpowers/plans/2026-05-08-radio-stage3-phase0-encoder-cache.md

## What was done

Phase 0 encoder cache infrastructure is complete and validated.

### Artifacts produced

- `src/data/encoder_cache.py` — cache I/O library (hash, write, read, CacheMiss, sanitize)
- `src/models/radio_stage_b.py` — `cached_features` branch in `forward()`
- `src/train/train.py` — `StageBDataset` cached/live tier dispatch + tier-aware collate_fn
- `src/train/tier_sampler.py` — `build_tier_grouped_sampler`
- `scripts/build_encoder_cache.py` — offline cache builder (dry-run + full + resume)
- `scripts/validate_cache_correctness.py` — 100-sample correctness gate script
- `scripts/measure_encoder_cache_throughput.py` — VRAM sweep script
- `configs/preproc_stage3.yaml` — preprocessing config (hash component)

### Cache on GPU box

- Location: `data\cache\encoder\<FILL_HASH16>\`
- Total samples: ~215,985
- Total size: ~X GB
- Dtype: bfloat16
- metadata.json verified

### Phase 0 Gate Results

| Criterion | Result | Value |
|---|---|---|
| Disk math | PASS | projected=X TB, free=X GB |
| Cache built | PASS | size matches ±X% |
| Correctness | PASS | max_diff=X.Xe-X |
| Throughput | PASS | X.Xs/step at b_cached=X |
| Memory | PASS | X.X% VRAM at b_cached=X |

### Key values for Phase 1 (Plan C)

```
cache_hash16:       <FILL>
b_cached:           <FILL>
grad_accum_cached:  <FILL>   (= 16 / b_cached)
b_live:             2
grad_accum_live:    8
checkpoint_init:    checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt
preproc_cfg:        configs/preproc_stage3.yaml
```

## What is next (Phase 1 / Plan C)

1. Write `configs/train_stage3_radio_systems.yaml` using `b_cached` from above.
2. Extend training loop to dispatch on `batch["tier"]` key and use correct grad_accum per tier.
3. Wire `build_tier_grouped_sampler` into the training loop's DataLoader construction.
4. Launch Stage 3 training with explicit user go-ahead.

## Locked decisions carried forward

- DoRA adapters: encoder-side AND decoder-side. Encoder frozen for Stage 3. Cache captures frozen encoder+adapter output.
- Cache hash includes git HEAD SHA by default. Use `--ignore-git-sha` for CI.
- Per-sample `.pt` files store `(tensor, h16, w16)` tuple (no batch dim). Collate to `(B, seq_tokens, 1280)`.
- `deformable_attention` + `positional_bridge` are trainable in Stage 3 (not frozen). They run in both cached and live paths.
- Contour logits are NOT cached; they are recomputed from `positional_bridge` output at training time.
```

- [ ] **Step 4: Commit handoff**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git add docs/superpowers/handoffs/2026-05-08-radio-stage3-phase0-complete-handoff.md configs/preproc_stage3.yaml
git commit -m "docs(plan): Stage 3 Phase 0 complete handoff"
```

- [ ] **Step 5: Push branch**

```bash
cd /home/ari/work/Clarity-OMR-Train-RADIO && git push origin feat/stage3-encoder-cache
```

> **Review:** Confirm handoff doc has `b_cached`, `cache_hash16`, and `grad_accum_cached` filled in (not placeholder). Confirm memory files updated. Confirm branch pushed. Confirm user has explicitly approved Phase 1 before any training runs start.

---

## Wrap-up checklist

- [ ] Branch `feat/stage3-encoder-cache` created off `feat/system-level-rebuild`.
- [ ] `configs/preproc_stage3.yaml` committed with all preprocessing fields.
- [ ] `src/data/encoder_cache.py` complete: hash, sanitize, write, read, exists, metadata.
- [ ] `src/models/radio_stage_b.py` extended: `cached_features` branch in `forward()`.
- [ ] `src/train/train.py` extended: `StageBDataset` cached/live tier dispatch + tier-aware `collate_fn`.
- [ ] `src/train/tier_sampler.py` created: `build_tier_grouped_sampler`.
- [ ] All unit tests green: `tests/data/test_encoder_cache.py`, `tests/models/test_radio_stage_b_cached.py`, `tests/train/test_tier_grouped_sampler.py`, `tests/train/test_cached_dataset.py`.
- [ ] Phase 0a dry-run complete; sizing gate decision documented.
- [ ] Full cache built on GPU box; `metadata.json` verified.
- [ ] Correctness validation PASS on 100 samples (max_diff ≤ 1e-3).
- [ ] Throughput sweep complete; `b_cached` determined.
- [ ] All 5 Phase 0 gate criteria PASS.
- [ ] User has given explicit go-ahead for Phase 1.
- [ ] Handoff doc committed with all key values filled in.

