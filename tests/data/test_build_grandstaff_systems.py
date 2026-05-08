"""Tests for scripts/build_grandstaff_systems.py — the GrandStaff manifest builder."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_grandstaff_systems import (
    build_manifest_entries,
    BuilderStats,
)


def _write_kern(p: Path, content: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")


def _write_jpg(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\xff\xd8\xff\xe0fake-jpg")


def test_two_spine_kern_with_clean_image_produces_one_entry(tmp_path: Path) -> None:
    grandstaff_root = tmp_path / "data" / "grandstaff"
    krn = grandstaff_root / "composer" / "piece" / "movement" / "original_m-0-5.krn"
    _write_kern(
        krn,
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4C\t4c\n*-\t*-\n",
    )
    _write_jpg(krn.with_suffix(".jpg"))

    entries, stats = build_manifest_entries(
        grandstaff_root,
        max_sequence_length=768,
        project_root=tmp_path,
    )

    assert len(entries) == 1
    e = entries[0]
    assert e["dataset"] == "grandstaff_systems"
    assert e["staves_in_system"] == 2
    assert e["variant"] == "clean"
    assert e["split"] in {"train", "val", "test"}
    assert e["image_path"].endswith("original_m-0-5.jpg")
    assert e["krn_path"].endswith("original_m-0-5.krn")
    seq = e["token_sequence"]
    assert seq[0] == "<bos>" and seq[-1] == "<eos>"
    assert seq.count("<staff_idx_0>") == 1
    assert seq.count("<staff_idx_1>") == 1
    assert stats.entries_written == 1
    assert stats.spine_count_histogram[2] == 1


def test_clean_and_distorted_variants_share_split(tmp_path: Path) -> None:
    """Both image variants of one .krn must land in the same split (no leakage)."""
    grandstaff_root = tmp_path / "data" / "grandstaff"
    krn = grandstaff_root / "c" / "p" / "m" / "original_m-0-5.krn"
    _write_kern(
        krn,
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4C\t4c\n*-\t*-\n",
    )
    _write_jpg(krn.with_suffix(".jpg"))
    _write_jpg(krn.with_name("original_m-0-5_distorted.jpg"))

    entries, _ = build_manifest_entries(
        grandstaff_root, max_sequence_length=768, project_root=tmp_path
    )
    splits = {e["split"] for e in entries}
    assert len(splits) == 1, f"variants assigned to different splits: {splits}"


def test_distorted_variant_emits_separate_entry(tmp_path: Path) -> None:
    grandstaff_root = tmp_path / "data" / "grandstaff"
    krn = grandstaff_root / "c" / "p" / "m" / "original_m-0-5.krn"
    _write_kern(
        krn,
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4C\t4c\n*-\t*-\n",
    )
    _write_jpg(krn.with_suffix(".jpg"))
    _write_jpg(krn.with_name("original_m-0-5_distorted.jpg"))

    entries, _ = build_manifest_entries(
        grandstaff_root, max_sequence_length=768, project_root=tmp_path
    )
    variants = sorted(e["variant"] for e in entries)
    assert variants == ["clean", "distorted"]


def test_oversized_sequence_is_skipped_and_logged(tmp_path: Path) -> None:
    grandstaff_root = tmp_path / "data" / "grandstaff"
    krn = grandstaff_root / "c" / "p" / "m" / "long.krn"
    # Create a kern with many measures (will exceed token budget).
    measures = "\n".join(f"=*\t=*\n4C\t4c" for _ in range(400))
    _write_kern(
        krn,
        "**kern\t**kern\n*clefF4\t*clefG2\n" + measures + "\n*-\t*-\n",
    )
    _write_jpg(krn.with_suffix(".jpg"))

    entries, stats = build_manifest_entries(
        grandstaff_root, max_sequence_length=50, project_root=tmp_path
    )
    assert entries == []
    assert stats.skipped_oversized == 1


def test_empty_root_produces_no_entries(tmp_path: Path) -> None:
    entries, stats = build_manifest_entries(
        tmp_path / "data" / "grandstaff", max_sequence_length=768, project_root=tmp_path
    )
    assert entries == []
    assert stats.entries_written == 0
