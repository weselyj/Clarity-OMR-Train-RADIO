# tests/data/test_build_stage3_combined_manifest.py
import json
import subprocess
import sys
from pathlib import Path

# Repo root resolved relative to this test file (tests/data/<file>.py -> repo root).
REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Canonical field set guaranteed by the normalization step in
# build_stage3_combined_manifest.py.  All entries in the combined manifest
# must carry these keys (values may be None / empty for corpora that don't
# have a meaningful value for a field).
# ---------------------------------------------------------------------------
CANONICAL_FIELDS = {
    # Core identity / routing
    "sample_id",
    "dataset",
    "split",
    # Image reference (at least one of image_path / source_path must be set,
    # but both are normalised to a non-None value in the combined manifest)
    "image_path",
    "source_path",
    # Token payload
    "token_sequence",
    "token_count",
    "staves_in_system",
    # Provenance / metadata — synthetic_systems–style fields
    "page_id",
    "system_index",
    "staff_indices",
    "style_id",
    "page_number",
    "source_format",
    "score_type",
    # GrandStaff–style fields
    "group_id",
    "modality",
    "variant",
    "krn_path",
}


def _make_synthetic_entries(n: int) -> list[dict]:
    """Entries that look like synthetic_systems output (rich field set)."""
    return [
        {
            "sample_id": f"synthetic_systems:page{i}__sys00",
            "dataset": "synthetic_systems",
            "split": "train",
            "image_path": f"data/synthetic_systems/{i}.png",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "x", "<staff_end>", "<eos>"],
            "token_count": 6,
            "staves_in_system": 2,
            "page_id": f"page{i}",
            "source_path": f"data/synthetic/{i}.mxl",
            "style_id": "bravura-compact",
            "page_number": 1,
            "system_index": 0,
            "staff_indices": [0, 1],
            "source_format": "musicxml",
            "score_type": "piano",
        }
        for i in range(n)
    ]


def _make_grandstaff_entries(n: int) -> list[dict]:
    """Entries that look like grandstaff_systems output (sparse field set)."""
    return [
        {
            "sample_id": f"grandstaff_systems:dir/piece{i}:piece{i}",
            "dataset": "grandstaff_systems",
            "group_id": f"dir/piece{i}",
            "modality": "image+notation",
            "variant": "clean",
            "split": "train",
            "image_path": f"data/grandstaff/{i}.jpg",
            "krn_path": f"data/grandstaff/{i}.krn",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "y", "<staff_end>", "<eos>"],
            "staves_in_system": 2,
        }
        for i in range(n)
    ]


def _make_primus_entries(n: int) -> list[dict]:
    """Entries that look like retokenize_with_staff_markers output for primus."""
    return [
        {
            "sample_id": f"primus_systems:{i}",
            "dataset": "primus_systems",
            "split": "train",
            "image_path": f"data/primus/{i}.png",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "z", "<staff_end>", "<eos>"],
            "token_count": 6,
            "staves_in_system": 1,
        }
        for i in range(n)
    ]


def _make_cameraprimus_entries(n: int) -> list[dict]:
    return [
        {
            "sample_id": f"cameraprimus_systems:{i}",
            "dataset": "cameraprimus_systems",
            "split": "train",
            "image_path": f"data/cameraprimus/{i}.png",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "w", "<staff_end>", "<eos>"],
            "token_count": 6,
            "staves_in_system": 1,
        }
        for i in range(n)
    ]


def _make_entries(dataset: str, n: int) -> list[dict]:
    """Legacy helper retained for the existing concatenation test."""
    return [
        {
            "sample_id": f"{dataset}:{i}",
            "dataset": dataset,
            "split": "train",
            "image_path": f"data/{dataset}/{i}.png",
            "token_sequence": ["<bos>", "<staff_start>", "<staff_idx_0>", "x", "<staff_end>", "<eos>"],
            "token_count": 6,
            "staves_in_system": 1,
        }
        for i in range(n)
    ]


def _write_manifest(path: Path, entries: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_build_combined_manifest_concatenates_four_sources(tmp_path: Path):
    syn = tmp_path / "synth.jsonl"
    gs = tmp_path / "gs.jsonl"
    pr = tmp_path / "primus.jsonl"
    cp = tmp_path / "cp.jsonl"
    _write_manifest(syn, _make_entries("synthetic_systems", 5))
    _write_manifest(gs, _make_entries("grandstaff_systems", 3))
    _write_manifest(pr, _make_entries("primus_systems", 2))
    _write_manifest(cp, _make_entries("cameraprimus_systems", 2))

    out = tmp_path / "combined.jsonl"
    audit = tmp_path / "combined_audit.json"
    cmd = [
        sys.executable, "scripts/build_stage3_combined_manifest.py",
        "--synthetic-systems-manifest", str(syn),
        "--grandstaff-systems-manifest", str(gs),
        "--primus-systems-manifest", str(pr),
        "--cameraprimus-systems-manifest", str(cp),
        "--output-manifest", str(out),
        "--audit-output", str(audit),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    lines = [json.loads(L) for L in out.read_text().splitlines()]
    assert len(lines) == 12
    # All four datasets represented
    datasets = {e["dataset"] for e in lines}
    assert datasets == {"synthetic_systems", "grandstaff_systems",
                        "primus_systems", "cameraprimus_systems"}

    audit_data = json.loads(audit.read_text())
    assert audit_data["total_entries"] == 12
    assert audit_data["per_dataset"]["synthetic_systems"] == 5
    assert audit_data["per_dataset"]["grandstaff_systems"] == 3
    assert audit_data["per_dataset"]["primus_systems"] == 2
    assert audit_data["per_dataset"]["cameraprimus_systems"] == 2


def _run_combined_manifest(tmp_path: Path, syn_entries, gs_entries, pr_entries, cp_entries):
    """Helper: write four source manifests, run the script, return parsed output lines."""
    syn = tmp_path / "synth.jsonl"
    gs = tmp_path / "gs.jsonl"
    pr = tmp_path / "primus.jsonl"
    cp = tmp_path / "cp.jsonl"
    _write_manifest(syn, syn_entries)
    _write_manifest(gs, gs_entries)
    _write_manifest(pr, pr_entries)
    _write_manifest(cp, cp_entries)

    out = tmp_path / "combined.jsonl"
    audit = tmp_path / "combined_audit.json"
    cmd = [
        sys.executable, "scripts/build_stage3_combined_manifest.py",
        "--synthetic-systems-manifest", str(syn),
        "--grandstaff-systems-manifest", str(gs),
        "--primus-systems-manifest", str(pr),
        "--cameraprimus-systems-manifest", str(cp),
        "--output-manifest", str(out),
        "--audit-output", str(audit),
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return [json.loads(L) for L in out.read_text().splitlines()]


def test_combined_manifest_all_entries_have_canonical_fields(tmp_path: Path):
    """Every entry in the combined manifest must carry the full canonical field set.

    This catches schema heterogeneity between builders: grandstaff_systems omits
    page_id / source_path / style_id / ... while synthetic_systems omits
    group_id / modality / variant / krn_path.  The combiner normalises missing
    fields to sensible defaults (None / "" / 0 / []) so downstream tools that
    access fields without .get() don't KeyError.
    """
    lines = _run_combined_manifest(
        tmp_path,
        _make_synthetic_entries(3),
        _make_grandstaff_entries(2),
        _make_primus_entries(2),
        _make_cameraprimus_entries(2),
    )

    assert len(lines) == 9
    for entry in lines:
        missing = CANONICAL_FIELDS - set(entry.keys())
        assert not missing, (
            f"Entry {entry.get('sample_id')} from dataset "
            f"'{entry.get('dataset')}' is missing canonical fields: {sorted(missing)}"
        )


def test_grandstaff_entry_gets_synthetic_field_defaults(tmp_path: Path):
    """grandstaff_systems entries must have synthetic-style fields populated with defaults."""
    lines = _run_combined_manifest(
        tmp_path,
        _make_synthetic_entries(1),
        _make_grandstaff_entries(1),
        _make_primus_entries(1),
        _make_cameraprimus_entries(1),
    )

    gs_entry = next(e for e in lines if e["dataset"] == "grandstaff_systems")
    # Fields that grandstaff builder doesn't set — normalised defaults expected
    assert gs_entry["page_id"] is None
    assert gs_entry["source_path"] is None
    assert gs_entry["style_id"] is None
    assert gs_entry["page_number"] is None
    assert gs_entry["system_index"] is None
    assert gs_entry["staff_indices"] is None
    assert gs_entry["source_format"] is None
    assert gs_entry["score_type"] is None
    # token_count must be filled from token_sequence length when absent
    assert gs_entry["token_count"] == len(gs_entry["token_sequence"])


def test_synthetic_entry_gets_grandstaff_field_defaults(tmp_path: Path):
    """synthetic_systems entries must have grandstaff-style fields populated with defaults."""
    lines = _run_combined_manifest(
        tmp_path,
        _make_synthetic_entries(1),
        _make_grandstaff_entries(1),
        _make_primus_entries(1),
        _make_cameraprimus_entries(1),
    )

    syn_entry = next(e for e in lines if e["dataset"] == "synthetic_systems")
    # Fields that synthetic builder doesn't set — normalised defaults expected
    assert syn_entry["group_id"] is None
    assert syn_entry["modality"] is None
    assert syn_entry["variant"] is None
    assert syn_entry["krn_path"] is None
