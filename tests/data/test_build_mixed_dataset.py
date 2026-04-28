"""Unit test for src/data/build_mixed_dataset.py."""
from pathlib import Path

import pytest


def make_fake_split(root: Path, n_images: int, prefix: str) -> Path:
    """Create n fake image+label pairs under root/{images,labels}/."""
    images = root / "images"
    labels = root / "labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    # Minimal valid PNG header + IHDR (8 + 25 bytes is enough for it to be openable)
    png_stub = bytes.fromhex(
        "89504e470d0a1a0a"  # PNG signature
        "0000000d"          # IHDR length
        "49484452"          # IHDR
        "0000000100000001"  # 1x1
        "0802000000"        # bit depth + color type + ...
        "907753de"          # CRC
        "0000000a49444154789c63000100000005000170"  # IDAT
        "00000000"          # CRC (placeholder)
        "0000000049454e44ae426082"  # IEND
    )
    for i in range(n_images):
        (images / f"{prefix}_{i:03d}.png").write_bytes(png_stub)
        (labels / f"{prefix}_{i:03d}.txt").write_text("0 0.5 0.5 0.8 0.1\n")
    return root


def test_assembles_all_sources(tmp_path: Path):
    from src.data.build_mixed_dataset import build_mixed_dataset

    syn = make_fake_split(tmp_path / "syn", 100, "syn")
    real = make_fake_split(tmp_path / "real", 50, "real")
    aug = make_fake_split(tmp_path / "aug", 30, "aug")
    out = tmp_path / "mixed"

    yaml_path = build_mixed_dataset(
        sources={"synthetic": syn, "real": real, "augment": aug},
        out_dir=out,
        val_ratio=0.2,
        seed=42,
    )

    assert yaml_path.exists()
    train_imgs = list((out / "train" / "images").glob("*.png"))
    val_imgs = list((out / "val" / "images").glob("*.png"))
    assert len(train_imgs) + len(val_imgs) == 180


def test_val_split_is_stratified(tmp_path: Path):
    """Each source corpus should be represented in val proportionally."""
    from src.data.build_mixed_dataset import build_mixed_dataset

    syn = make_fake_split(tmp_path / "syn", 100, "syn")
    real = make_fake_split(tmp_path / "real", 50, "real")
    out = tmp_path / "mixed"

    build_mixed_dataset(
        sources={"synthetic": syn, "real": real},
        out_dir=out,
        val_ratio=0.2,
        seed=42,
    )

    val_imgs = [p.name for p in (out / "val" / "images").glob("*.png")]
    syn_in_val = sum(1 for n in val_imgs if n.startswith("syn"))
    real_in_val = sum(1 for n in val_imgs if n.startswith("real"))
    assert 18 <= syn_in_val <= 22  # ~20 of 100
    assert 8 <= real_in_val <= 12  # ~10 of 50


def test_data_yaml_format(tmp_path: Path):
    from src.data.build_mixed_dataset import build_mixed_dataset

    syn = make_fake_split(tmp_path / "syn", 10, "syn")
    out = tmp_path / "mixed"
    yaml_path = build_mixed_dataset({"synthetic": syn}, out_dir=out, val_ratio=0.2, seed=0)

    import yaml
    cfg = yaml.safe_load(yaml_path.read_text())
    assert cfg["nc"] == 1
    assert cfg["names"] == ["staff"]
    assert "train" in cfg
    assert "val" in cfg


def test_skips_images_without_labels(tmp_path: Path):
    """An image without a matching label file must NOT appear in the output."""
    from src.data.build_mixed_dataset import build_mixed_dataset

    syn = make_fake_split(tmp_path / "syn", 5, "syn")
    # Add an orphan image (no matching .txt)
    png_stub = (syn / "images" / "syn_000.png").read_bytes()
    (syn / "images" / "orphan.png").write_bytes(png_stub)
    out = tmp_path / "mixed"

    build_mixed_dataset({"synthetic": syn}, out_dir=out, val_ratio=0.2, seed=0)

    all_imgs = list((out / "train" / "images").glob("*.png")) + list((out / "val" / "images").glob("*.png"))
    assert not any(p.name.startswith("orphan") for p in all_imgs)


def test_filename_collision_handling(tmp_path: Path):
    """Same filename in two sources must NOT clobber each other in the output."""
    from src.data.build_mixed_dataset import build_mixed_dataset

    src_a = tmp_path / "a"
    src_b = tmp_path / "b"
    make_fake_split(src_a, 1, "page")  # produces page_000.png
    make_fake_split(src_b, 1, "page")  # ALSO produces page_000.png
    out = tmp_path / "mixed"

    build_mixed_dataset({"a": src_a, "b": src_b}, out_dir=out, val_ratio=0.0, seed=0)

    train_imgs = list((out / "train" / "images").glob("*.png"))
    train_lbls = list((out / "train" / "labels").glob("*.txt"))
    assert len(train_imgs) == 2, f"Expected 2 distinct images after collision-handling; got {len(train_imgs)}: {[p.name for p in train_imgs]}"
    assert len(train_lbls) == 2
