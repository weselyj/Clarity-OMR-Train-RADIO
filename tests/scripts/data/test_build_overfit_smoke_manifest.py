"""Smoke-manifest builder must produce 20 entries deterministically."""
from pathlib import Path
import json
import tempfile

from scripts.data.build_overfit_smoke_manifest import build_manifest


def test_build_manifest_deterministic(tmp_path):
    """Builder is deterministic and produces 20 entries."""
    # Setup: fake source manifest with 100 candidate entries
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(100):
            f.write(
                json.dumps({
                    "source_path": f"data/processed/grandstaff_systems/page_{i:03d}_sys000.png",
                    "image_path": f"data/processed/grandstaff_systems/images/page_{i:03d}_sys000.png",
                    "tokens": ["<staff_idx_0>", "clef-G2", "<staff_idx_1>", "clef-F4"],
                    "staff_count": 2,
                    "split": "train",
                    "token_count": 100,
                })
                + "\n"
            )

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    build_manifest(src_manifest=src, output=out_a, n=20, max_token_count=256)
    build_manifest(src_manifest=src, output=out_b, n=20, max_token_count=256)

    a_lines = out_a.read_text().splitlines()
    b_lines = out_b.read_text().splitlines()
    assert len(a_lines) == 20
    assert a_lines == b_lines  # deterministic


def test_build_manifest_respects_max_tokens(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(50):
            tok_count = 100 if i < 25 else 500
            f.write(
                json.dumps({
                    "source_path": f"src_{i}.png",
                    "image_path": f"img_{i}.png",
                    "tokens": ["<staff_idx_0>", "clef-G2"],
                    "staff_count": 2,
                    "split": "train",
                    "token_count": tok_count,
                })
                + "\n"
            )
    out = tmp_path / "out.jsonl"
    build_manifest(src_manifest=src, output=out, n=20, max_token_count=256)
    entries = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(e["token_count"] <= 256 for e in entries)


def test_build_manifest_two_staff_only(tmp_path):
    src = tmp_path / "src.jsonl"
    with src.open("w") as f:
        for i in range(50):
            staff = 2 if i % 2 == 0 else 3
            f.write(
                json.dumps({
                    "source_path": f"src_{i}.png",
                    "image_path": f"img_{i}.png",
                    "tokens": [],
                    "staff_count": staff,
                    "split": "train",
                    "token_count": 100,
                })
                + "\n"
            )
    out = tmp_path / "out.jsonl"
    build_manifest(src_manifest=src, output=out, n=20, max_token_count=256)
    entries = [json.loads(line) for line in out.read_text().splitlines()]
    assert all(e["staff_count"] == 2 for e in entries)
