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
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
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
      2. Replace any remaining '/', ':', '\\' with token-safe escape sequences
         (_SLASH_, _COLON_, _BSLASH_) so the mapping is reversible and two
         distinct IDs can never collide after sanitization (unlike the previous
         '__' escape, where e.g. 'Abbott__p001__sys00' and
         'Abbott__p001/sys00' would both become 'Abbott__p001__sys00').

    Note: Legacy caches built before 2026-05-09 used a lossy single-'__' escape;
    rebuilds use this reversible escape. The old and new escapes produce different
    filenames only for IDs that contain '/', ':', or '\\'; sample IDs consisting
    purely of alphanumerics and '__' are unaffected.
    """
    if ":" in sample_id:
        sample_id = sample_id.split(":", 1)[1]
    sample_id = (
        sample_id
        .replace("/", "_SLASH_")
        .replace(":", "_COLON_")
        .replace("\\", "_BSLASH_")
    )
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


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------

def write_cache_entry(
    cache_root: Path,
    hash16: str,
    tier: str,
    sample_key: str,
    tensor: "torch.Tensor",
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
    import torch

    dest_dir = Path(cache_root) / hash16 / tier
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{sample_key}.pt"
    # `.clone()` materializes a fresh contiguous storage. Without it, a tensor
    # that's a slice of a larger batch (e.g. `feature_map[i]` from an encoder
    # forward over a batch of B samples) would serialize the entire batch's
    # underlying storage — bloating each .pt file by ~B× on disk.
    payload = (tensor.detach().cpu().to(torch.bfloat16).contiguous().clone(),
               int(h16), int(w16))
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
) -> Tuple["torch.Tensor", int, int]:
    """Load and return the cached bf16 tensor plus spatial shape.

    Returns:
        (tensor, h16, w16) where tensor has shape (seq_tokens, 1280) and
        h16 * w16 == seq_tokens.

    Raises:
        CacheMiss: If the .pt file does not exist.
    """
    import torch

    p = Path(cache_root) / hash16 / tier / f"{sample_key}.pt"
    if not p.exists():
        raise CacheMiss(
            f"Cache miss: no entry for tier={tier!r} key={sample_key!r} "
            f"under hash {hash16!r} in {cache_root}"
        )
    payload = torch.load(p, weights_only=True, map_location="cpu")
    tensor, h16, w16 = payload
    return tensor, int(h16), int(w16)
