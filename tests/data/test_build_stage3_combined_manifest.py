# tests/data/test_build_stage3_combined_manifest.py
import json
import subprocess
import sys
from pathlib import Path


def _make_entries(dataset: str, n: int) -> list[dict]:
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
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
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
