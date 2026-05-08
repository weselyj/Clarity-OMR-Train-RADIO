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
    assert data["sample_label_mismatch_pages"][0]["manifest_count"] == 7
    assert data["sample_label_mismatch_pages"][0]["label_count"] == 9
