import json
import subprocess
from pathlib import Path


def _write_manifest(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry) + "\n")


def test_audit_clean_manifest_reports_no_drift(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": 0, "token_sequence": ["<bos>"]},
        {"page_id": "p001", "staff_index": 1, "token_sequence": ["<bos>"]},
        {"page_id": "p001", "staff_index": 2, "token_sequence": ["<bos>"]},
    ])
    out = tmp_path / "audit.json"
    result = subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest), "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert data["pages_total"] == 1
    assert data["pages_with_index_gap"] == 0
    assert data["pages_with_non_contiguous_indices"] == 0


def test_audit_drift_manifest_reports_gap(tmp_path: Path) -> None:
    """A page with manifest indices 0, 1, 2, 3, 5, 6 (gap at 4) should be flagged."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": idx, "token_sequence": ["<bos>"]}
        for idx in [0, 1, 2, 3, 5, 6]
    ])
    out = tmp_path / "audit.json"
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest), "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert data["pages_with_index_gap"] == 1
    assert data["pages_with_non_contiguous_indices"] == 1
    assert data["sample_drifted_pages"][0]["page_id"] == "p001"
    assert data["sample_drifted_pages"][0]["staff_indices"] == [0, 1, 2, 3, 5, 6]


def test_audit_post_filter_contiguous_manifest_not_flagged_without_labels(tmp_path: Path) -> None:
    """If indices ARE contiguous (0..K-1) and labels-root is not provided,
    we cannot detect post-filter renumbering; just report counts."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": idx, "token_sequence": ["<bos>"]}
        for idx in [0, 1, 2, 3, 4, 5, 6]
    ])
    out = tmp_path / "audit.json"
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest), "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert data["pages_with_non_contiguous_indices"] == 0
    assert data["entries_per_page"] == {"7": 1}  # 1 page with 7 entries


def test_audit_with_labels_root_detects_post_filter_renumber(tmp_path: Path) -> None:
    """When labels-root is provided, audit compares manifest entry count
    to label line count per page — exposing post-filter renumbering."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": idx, "token_sequence": ["<bos>"]}
        for idx in [0, 1, 2, 3, 4, 5, 6]
    ])
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    (labels_dir / "p001.txt").write_text("\n".join(["0 0.5 0.1 0.9 0.05"] * 9) + "\n")
    out = tmp_path / "audit.json"
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest),
         "--labels-root", str(labels_dir),
         "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert data["pages_with_label_count_mismatch"] == 1
    assert data["sample_label_mismatch_pages"][0]["manifest_entry_count"] == 7
    assert data["sample_label_mismatch_pages"][0]["manifest_unique_count"] == 7
    assert data["sample_label_mismatch_pages"][0]["label_count"] == 9


def test_audit_duplicate_indices_flagged(tmp_path: Path) -> None:
    """A page with duplicate staff_index values (e.g., [0, 1, 1, 2])
    should be flagged as having duplicates, separate from contiguity."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": idx, "token_sequence": ["<bos>"]}
        for idx in [0, 1, 1, 2]
    ])
    out = tmp_path / "audit.json"
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest), "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    assert data["pages_with_duplicate_indices"] == 1
    assert data["pages_with_non_contiguous_indices"] == 0


def test_audit_filters_by_dataset_when_flag_set(tmp_path: Path) -> None:
    """Manifest has two dataset variants per (page, staff_index) — piano-style layout.

    A page with 3 physical staves produces 6 entries with indices [0, 0, 1, 1, 2, 2]
    when both 'synthetic_fullpage' and 'synthetic_polyphonic' datasets are present.

    With --dataset synthetic_fullpage only the 3 matching entries are loaded (indices
    [0, 1, 2]), so no duplicate-index false positive is raised.

    Without the flag, all 6 entries are loaded and the duplicate-index check fires.
    """
    manifest = tmp_path / "manifest.jsonl"
    entries = []
    for staff_idx in range(3):
        for ds in ("synthetic_fullpage", "synthetic_polyphonic"):
            entries.append({
                "page_id": "p001",
                "staff_index": staff_idx,
                "dataset": ds,
                "token_sequence": ["<bos>"],
            })
    _write_manifest(manifest, entries)
    out_filtered = tmp_path / "audit_filtered.json"
    out_unfiltered = tmp_path / "audit_unfiltered.json"

    # With --dataset filter: expect no duplicates
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest),
         "--dataset", "synthetic_fullpage",
         "--output", str(out_filtered)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    filtered = json.loads(out_filtered.read_text())
    assert filtered["dataset_filter"] == "synthetic_fullpage"
    assert filtered["pages_total"] == 1
    assert filtered["pages_with_duplicate_indices"] == 0
    assert filtered["pages_with_non_contiguous_indices"] == 0

    # Without --dataset filter: 6 entries per page → indices [0,0,1,1,2,2] → duplicates fired
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest),
         "--output", str(out_unfiltered)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    unfiltered = json.loads(out_unfiltered.read_text())
    assert unfiltered["dataset_filter"] is None
    assert unfiltered["pages_total"] == 1
    assert unfiltered["pages_with_duplicate_indices"] == 1


def test_audit_dataset_filter_preserves_old_no_filter_behavior(tmp_path: Path) -> None:
    """Regression guard: on a single-dataset manifest, omitting --dataset produces
    the same output as before the flag was introduced."""
    manifest = tmp_path / "manifest.jsonl"
    _write_manifest(manifest, [
        {"page_id": "p001", "staff_index": 0, "dataset": "only_dataset", "token_sequence": ["<bos>"]},
        {"page_id": "p001", "staff_index": 1, "dataset": "only_dataset", "token_sequence": ["<bos>"]},
        {"page_id": "p002", "staff_index": 0, "dataset": "only_dataset", "token_sequence": ["<bos>"]},
    ])
    out = tmp_path / "audit.json"
    subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest),
         "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=True,
    )
    data = json.loads(out.read_text())
    # No filter applied
    assert data["dataset_filter"] is None
    # Two pages, both clean
    assert data["pages_total"] == 2
    assert data["pages_with_duplicate_indices"] == 0
    assert data["pages_with_index_gap"] == 0
    assert data["pages_with_non_contiguous_indices"] == 0
    # p001 has 2 entries, p002 has 1 entry
    assert data["entries_per_page"] == {"1": 1, "2": 1}


def test_audit_raises_on_malformed_manifest_line(tmp_path: Path) -> None:
    """A malformed JSON line should raise with line number context."""
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        '{"page_id": "p001", "staff_index": 0, "token_sequence": ["<bos>"]}\n'
        '{not json\n'  # malformed
    )
    out = tmp_path / "audit.json"
    result = subprocess.run(
        ["python3", "scripts/audit_per_staff_alignment.py",
         "--manifest", str(manifest), "--output", str(out)],
        cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
        capture_output=True, text=True, check=False,
    )
    assert result.returncode != 0
    assert "Bad manifest line 2" in result.stderr
