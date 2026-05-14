"""Test scanned_grandstaff_systems corpus builder.

Fixtures mirror the real grandstaff_systems manifest schema (sample_id,
dataset, variant, token_sequence, staves_in_system, image_path, krn_path)
so the tests catch the same field-name issues that would surface on real data.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.build_scanned_grandstaff_systems import build_corpus


def _write_fixture(tmp_path, image_path_str, sample_id="grandstaff_systems:demo/piece:m-0-5"):
    """Build a 1-entry source manifest + matching image file under tmp_path.

    image_path_str may be absolute (test mode) or relative (production mode);
    the actual image is always written to its absolute location under tmp_path.
    """
    src_root = tmp_path / "src"
    src_manifests = src_root / "manifests"
    src_manifests.mkdir(parents=True)

    img = Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L")
    p = Path(image_path_str)
    img_abs = p if p.is_absolute() else tmp_path / p
    img_abs.parent.mkdir(parents=True, exist_ok=True)
    img.save(img_abs)

    entry = {
        "sample_id": sample_id,
        "dataset": "grandstaff_systems",
        "group_id": "demo/piece",
        "modality": "image+notation",
        "variant": "clean",
        "split": "train",
        "image_path": str(image_path_str),
        "krn_path": "data/grandstaff/demo/piece/m-0-5.krn",
        "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "clef-G2", "<staff_end>", "<eos>"],
        "staves_in_system": 2,
    }
    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps(entry) + "\n")
    return src_root, img


def test_build_corpus_preserves_token_labels(tmp_path):
    abs_img = tmp_path / "src/images/test.png"
    src_root, _ = _write_fixture(tmp_path, str(abs_img))

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None, repo_root=tmp_path)

    out_manifest = out_root / "manifests/synthetic_token_manifest.jsonl"
    assert out_manifest.exists()
    entries = [json.loads(line) for line in out_manifest.read_text().splitlines()]
    assert len(entries) == 1
    e = entries[0]
    assert e["token_sequence"] == ["<bos>", "<staff_start>", "<staff_idx_0>", "clef-G2", "<staff_end>", "<eos>"]
    assert e["staves_in_system"] == 2
    assert e["split"] == "train"
    assert e["dataset"] == "scanned_grandstaff_systems"
    assert e["variant"] == "scanned"
    assert e["sample_id"] == "scanned_grandstaff_systems:demo/piece:m-0-5"
    assert e["group_id"] == "demo/piece"
    assert e["krn_path"] == "data/grandstaff/demo/piece/m-0-5.krn"
    assert "original_image_path" in e


def test_build_corpus_writes_degraded_image(tmp_path):
    abs_img = tmp_path / "src/images/test.png"
    src_root, source_img = _write_fixture(tmp_path, str(abs_img))

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None, repo_root=tmp_path)

    out_images = list((out_root / "images").rglob("*.png"))
    assert len(out_images) == 1
    out_img = Image.open(out_images[0])
    assert out_img.size == source_img.size
    assert not np.array_equal(np.asarray(source_img), np.asarray(out_img.convert("L")))


def test_build_corpus_idempotent(tmp_path):
    """Re-running with same inputs yields same outputs (deterministic seeds)."""
    abs_img = tmp_path / "src/images/test.png"
    src_root, _ = _write_fixture(tmp_path, str(abs_img))

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"
    build_corpus(src_root=src_root, out_root=out_a, limit=None, repo_root=tmp_path)
    build_corpus(src_root=src_root, out_root=out_b, limit=None, repo_root=tmp_path)

    img_a = np.asarray(Image.open(next((out_a / "images").rglob("*.png"))))
    img_b = np.asarray(Image.open(next((out_b / "images").rglob("*.png"))))
    assert np.array_equal(img_a, img_b)


def test_build_corpus_mirrors_relative_image_path_structure(tmp_path):
    """Relative image_paths (production case) must preserve subdir structure to avoid stem collisions."""
    src_root = tmp_path / "src"
    src_manifests = src_root / "manifests"
    src_manifests.mkdir(parents=True)

    repo_root = tmp_path
    for piece in ["sonata01", "sonata02"]:
        img_rel = f"data/grandstaff/beethoven/{piece}/m-0-5.jpg"
        img_abs = repo_root / img_rel
        img_abs.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L").save(img_abs)

    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        for piece in ["sonata01", "sonata02"]:
            f.write(json.dumps({
                "sample_id": f"grandstaff_systems:beethoven/{piece}/m-0-5:m-0-5",
                "dataset": "grandstaff_systems",
                "group_id": f"beethoven/{piece}",
                "variant": "clean",
                "split": "train",
                "image_path": f"data/grandstaff/beethoven/{piece}/m-0-5.jpg",
                "krn_path": f"data/grandstaff/beethoven/{piece}/m-0-5.krn",
                "token_sequence": ["<bos>", "<eos>"],
                "staves_in_system": 2,
            }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None, repo_root=repo_root)

    out_images = sorted((out_root / "images").rglob("*.png"))
    assert len(out_images) == 2
    rel_paths = [str(p.relative_to(out_root / "images")) for p in out_images]
    assert any("sonata01" in p for p in rel_paths)
    assert any("sonata02" in p for p in rel_paths)

    entries = [json.loads(line) for line in (out_root / "manifests/synthetic_token_manifest.jsonl").read_text().splitlines()]
    image_paths = [e["image_path"] for e in entries]
    assert len(set(image_paths)) == 2


def test_build_corpus_seed_uses_sample_id_when_available(tmp_path):
    """Identical source images with different sample_ids must produce different degradations."""
    src_root = tmp_path / "src"
    src_manifests = src_root / "manifests"
    src_manifests.mkdir(parents=True)

    img_rel_a = "data/grandstaff/demo/piece/m-0-5.jpg"
    img_rel_b = "data/grandstaff/demo/piece/m-0-5-copy.jpg"
    for r in (img_rel_a, img_rel_b):
        p = tmp_path / r
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.full((200, 800), 255, dtype=np.uint8), mode="L").save(p)

    with (src_manifests / "synthetic_token_manifest.jsonl").open("w") as f:
        f.write(json.dumps({
            "sample_id": "grandstaff_systems:alpha", "dataset": "grandstaff_systems",
            "image_path": img_rel_a, "token_sequence": [], "staves_in_system": 2, "variant": "clean",
        }) + "\n")
        f.write(json.dumps({
            "sample_id": "grandstaff_systems:beta", "dataset": "grandstaff_systems",
            "image_path": img_rel_b, "token_sequence": [], "staves_in_system": 2, "variant": "clean",
        }) + "\n")

    out_root = tmp_path / "out"
    build_corpus(src_root=src_root, out_root=out_root, limit=None, repo_root=tmp_path)

    out_pngs = sorted((out_root / "images").rglob("*.png"))
    assert len(out_pngs) == 2
    a = np.asarray(Image.open(out_pngs[0]))
    b = np.asarray(Image.open(out_pngs[1]))
    assert not np.array_equal(a, b), "Different sample_ids must produce different degradations"
