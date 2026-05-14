"""Test scanned_grandstaff_systems corpus builder."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.build_scanned_grandstaff_systems import build_corpus


def test_build_corpus_preserves_token_labels(tmp_path):
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    # Make a test image
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    # Make a source manifest with one entry
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"],
            "split": "train",
            "staff_count": 2,
            "token_count": 4,
        }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None)

    out_manifest = out_root / "manifests/synthetic_token_manifest.jsonl"
    assert out_manifest.exists()
    entries = [json.loads(line) for line in out_manifest.read_text().splitlines()]
    assert len(entries) == 1
    assert entries[0]["tokens"] == ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"]
    assert entries[0]["split"] == "train"


def test_build_corpus_writes_degraded_image(tmp_path):
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": [],
            "split": "train",
            "staff_count": 2,
            "token_count": 0,
        }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None)

    out_images = list((out_root / "images").glob("*.png"))
    assert len(out_images) == 1
    out_img = Image.open(out_images[0])
    assert out_img.size == img.size
    # Pixel content must differ from source
    assert not np.array_equal(np.asarray(img), np.asarray(out_img.convert("L")))


def test_build_corpus_idempotent(tmp_path):
    """Re-running with same inputs yields same outputs (deterministic seeds)."""
    src_root = tmp_path / "src"
    src_images = src_root / "images"
    src_manifests = src_root / "manifests"
    src_images.mkdir(parents=True)
    src_manifests.mkdir(parents=True)
    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    img.save(src_images / "test.png")
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "source_path": "test.png",
            "image_path": str(src_images / "test.png"),
            "tokens": [],
            "split": "train",
            "staff_count": 2,
            "token_count": 0,
        }) + "\n")

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    build_corpus(src_root=src_root, out_root=out_a, limit=None)
    build_corpus(src_root=src_root, out_root=out_b, limit=None)

    img_a = np.asarray(Image.open(next((out_a / "images").glob("*.png"))))
    img_b = np.asarray(Image.open(next((out_b / "images").glob("*.png"))))
    assert np.array_equal(img_a, img_b)
