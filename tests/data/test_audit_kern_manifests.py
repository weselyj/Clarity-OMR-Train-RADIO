"""Tests for scripts/audit_kern_manifests.py."""
from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_kern_manifests import (
    audit_manifest,
    classify_diff,
)


def test_classify_diff_unchanged() -> None:
    new = ["<bos>", "<staff_start>", "note-C4", "<staff_end>", "<eos>"]
    old = list(new)
    assert classify_diff(new, old) == "unchanged"


def test_classify_diff_minor() -> None:
    # Edit distance 1 of length-10 -> 10% but spec uses len ratio; pick something <=5%.
    new = ["a"] * 100
    old = ["a"] * 99 + ["b"]
    assert classify_diff(new, old) == "changed_minor"


def test_classify_diff_major() -> None:
    new = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
    old = ["x"] * 10
    assert classify_diff(new, old) == "changed_major"


def test_audit_manifest_with_unchanged_and_changed(tmp_path: Path) -> None:
    # A manifest with one unchanged entry (single-spine kern) and one changed entry (2-spine).
    grandstaff_root = tmp_path / "data" / "grandstaff"
    single = grandstaff_root / "single.krn"
    single.parent.mkdir(parents=True, exist_ok=True)
    single.write_text("**kern\n*clefG2\n*k[]\n*M4/4\n=1\n4c\n*-\n", encoding="utf-8")

    multi = grandstaff_root / "multi.krn"
    multi.write_text(
        "**kern\t**kern\n*clefF4\t*clefG2\n*k[]\t*k[]\n*M4/4\t*M4/4\n=1\t=1\n4C\t4c\n*-\t*-\n",
        encoding="utf-8",
    )

    manifest_path = tmp_path / "manifests" / "test.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Stored sequences match the OLD (single-staff-with-voices) tokenization.
    manifest_path.write_text(
        json.dumps(
            {
                "sample_id": "single",
                "krn_path": str(single.relative_to(tmp_path)),
                "token_sequence": ["<bos>", "<staff_start>", "clef-G2", "timeSignature-4/4", "<measure_start>", "note-C4", "_quarter", "<measure_end>", "<staff_end>", "<eos>"],
            }
        )
        + "\n"
        + json.dumps(
            {
                "sample_id": "multi",
                "krn_path": str(multi.relative_to(tmp_path)),
                "token_sequence": ["<bos>", "<staff_start>", "clef-F4", "<voice_1>", "note-C2", "_quarter", "<voice_2>", "clef-G2", "note-C4", "_quarter", "<staff_end>", "<eos>"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = audit_manifest(manifest_path, project_root=tmp_path)
    assert report["entries_with_krn_source"] == 2
    assert report["spine_count_histogram"]["1"] == 1
    assert report["spine_count_histogram"]["2"] == 1
    # 'multi' should be tagged as changed (major or minor depending on diff).
    assert report["tag_counts"]["unchanged"] + report["tag_counts"]["changed_minor"] + report["tag_counts"]["changed_major"] == 2
    assert report["tag_counts"]["changed_major"] >= 1
