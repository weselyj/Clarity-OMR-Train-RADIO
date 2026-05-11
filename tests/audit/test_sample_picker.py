"""Unit tests for the audit sample picker."""
from __future__ import annotations
import json
from pathlib import Path


def _write_manifest(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_picks_n_samples_per_corpus(tmp_path: Path):
    from scripts.audit._sample_picker import pick_audit_samples

    # Build fake manifest with 100 entries across 4 corpora
    manifest = tmp_path / "manifest.jsonl"
    entries = []
    for corpus in ("synthetic_v2", "grandstaff", "primus", "cameraprimus"):
        for i in range(25):
            entries.append({
                "sample_id": f"{corpus}_{i:03d}",
                "dataset": corpus,
                "split": "train" if i < 20 else "val",
                "image_path": f"/fake/{corpus}/{i:03d}.png",
                "token_sequence": ["<bos>", "clef-G2", "<eos>"],
            })
    _write_manifest(manifest, entries)

    samples = pick_audit_samples(manifest, n_per_corpus=5, seed=42)

    # 4 corpora x 5 samples = 20
    assert len(samples) == 20
    # 5 from each corpus
    from collections import Counter
    counts = Counter(s["dataset"] for s in samples)
    assert all(c == 5 for c in counts.values()), f"got {counts}"
    # Only train split
    assert all(s["split"] == "train" for s in samples)
    # Deterministic with same seed
    samples2 = pick_audit_samples(manifest, n_per_corpus=5, seed=42)
    assert [s["sample_id"] for s in samples] == [s["sample_id"] for s in samples2]


def test_handles_missing_corpus(tmp_path: Path):
    """If a corpus has 0 train entries, return what's available without crashing."""
    from scripts.audit._sample_picker import pick_audit_samples

    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {"sample_id": "a", "dataset": "synthetic_v2", "split": "train",
         "image_path": "/x.png", "token_sequence": []},
        {"sample_id": "b", "dataset": "synthetic_v2", "split": "train",
         "image_path": "/y.png", "token_sequence": []},
    ]
    _write_manifest(manifest, entries)

    samples = pick_audit_samples(manifest, n_per_corpus=5, seed=42)

    # Only 2 available, all from synthetic_v2
    assert len(samples) == 2
    assert all(s["dataset"] == "synthetic_v2" for s in samples)
