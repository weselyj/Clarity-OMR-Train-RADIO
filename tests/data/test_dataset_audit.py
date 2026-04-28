"""Unit test for src/data/dataset_audit.py."""
from pathlib import Path

import pytest


def make_fake_split(root: Path, n: int, prefix: str, width: int = 800, height: int = 1000):
    """Create n image+label pairs at the given dimensions."""
    from PIL import Image
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        img = Image.new("RGB", (width, height), "white")
        img.save(images / f"{prefix}_{i:03d}.png")
        # Two staff bboxes per image
        (labels / f"{prefix}_{i:03d}.txt").write_text(
            "0 0.5 0.2 0.8 0.05\n"
            "0 0.5 0.5 0.8 0.05\n"
        )


def test_audit_returns_split_summary(tmp_path: Path):
    from src.data.dataset_audit import audit_dataset

    # Build a minimal dataset structure: train/ and val/ with images + labels
    make_fake_split(tmp_path / "train", 5, "p")
    make_fake_split(tmp_path / "val", 2, "p")

    report = audit_dataset(tmp_path)

    assert "train" in report
    assert "val" in report
    assert report["train"]["n_images"] == 5
    assert report["val"]["n_images"] == 2


def test_audit_counts_bboxes(tmp_path: Path):
    from src.data.dataset_audit import audit_dataset

    make_fake_split(tmp_path / "train", 3, "p")  # 3 images x 2 bboxes = 6 total
    (tmp_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

    report = audit_dataset(tmp_path)
    assert report["train"]["bbox_count_mean"] == 2.0
    assert report["train"]["class_distribution"][0] == 6


def test_audit_handles_orphan_image(tmp_path: Path):
    """An image with no .txt label counts as 0 bboxes (not an error)."""
    from src.data.dataset_audit import audit_dataset
    from PIL import Image

    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    Image.new("RGB", (100, 100), "white").save(tmp_path / "train" / "images" / "lonely.png")
    (tmp_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

    report = audit_dataset(tmp_path)
    assert report["train"]["n_images"] == 1
    assert report["train"]["bbox_count_mean"] == 0


def test_audit_size_histogram(tmp_path: Path):
    """Image-size histogram bins by 500-pixel buckets."""
    from src.data.dataset_audit import audit_dataset

    make_fake_split(tmp_path / "train", 2, "small", width=800, height=1000)   # bucket 500x1000
    make_fake_split(tmp_path / "train", 3, "big", width=2500, height=3300)    # bucket 2500x3000
    (tmp_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "val" / "labels").mkdir(parents=True, exist_ok=True)

    report = audit_dataset(tmp_path)
    hist = report["train"]["size_histogram"]
    assert hist["500x1000"] == 2
    assert hist["2500x3000"] == 3


def test_audit_main_writes_json(tmp_path: Path, monkeypatch):
    """main() respects --out path and writes JSON."""
    from src.data import dataset_audit

    make_fake_split(tmp_path / "train", 2, "p")
    (tmp_path / "val" / "images").mkdir(parents=True, exist_ok=True)
    (tmp_path / "val" / "labels").mkdir(parents=True, exist_ok=True)
    out_json = tmp_path / "audit.json"

    monkeypatch.setattr(
        "sys.argv",
        ["dataset_audit.py", str(tmp_path), "--out", str(out_json)],
    )
    dataset_audit.main()

    import json
    payload = json.loads(out_json.read_text())
    assert "train" in payload
