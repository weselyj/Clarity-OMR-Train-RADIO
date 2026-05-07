# tests/data/test_retokenize_with_staff_markers.py
import json
from pathlib import Path
import subprocess
import sys


def _make_entry(dataset: str, sample_id: str, content_tokens: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "dataset": dataset,
        "split": "train",
        "image_path": f"data/{dataset}/x.png",
        "source_path": f"data/{dataset}/x.semantic",
        "source_format": "semantic",
        "token_sequence": ["<bos>", "<staff_start>"] + content_tokens + ["<staff_end>", "<eos>"],
        "token_count": 4 + len(content_tokens),
    }


def test_retokenize_adds_staff_idx_0_marker(tmp_path: Path):
    src = tmp_path / "input.jsonl"
    src.write_text(
        json.dumps(_make_entry("primus", "primus:001", ["clef-G2", "note-C4"])) + "\n"
        + json.dumps(_make_entry("primus", "primus:002", ["clef-G2", "rest"])) + "\n"
        + json.dumps(_make_entry("cameraprimus", "cameraprimus:001", ["clef-G2", "note-D4"])) + "\n"
    )

    out = tmp_path / "output.jsonl"
    cmd = [
        sys.executable,
        "scripts/retokenize_with_staff_markers.py",
        "--input-manifest", str(src),
        "--source-dataset", "primus",
        "--target-dataset", "primus_systems",
        "--output-manifest", str(out),
    ]
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    lines = [json.loads(L) for L in out.read_text().splitlines()]
    # Only primus entries selected (not cameraprimus)
    assert len(lines) == 2
    for e in lines:
        assert e["dataset"] == "primus_systems"
        assert e["staves_in_system"] == 1
        assert e["token_sequence"][0] == "<bos>"
        assert e["token_sequence"][1] == "<staff_start>"
        assert e["token_sequence"][2] == "<staff_idx_0>"
        assert e["token_sequence"][-2] == "<staff_end>"
        assert e["token_sequence"][-1] == "<eos>"
        # token_count updated to reflect inserted marker
        assert e["token_count"] == len(e["token_sequence"])


def test_retokenize_rejects_malformed_input(tmp_path: Path):
    src = tmp_path / "bad.jsonl"
    src.write_text(json.dumps({
        "sample_id": "x",
        "dataset": "primus",
        "split": "train",
        "image_path": "x.png",
        "source_path": "x",
        "source_format": "semantic",
        "token_sequence": ["clef-G2"],  # missing wrappers entirely
        "token_count": 1,
    }) + "\n")

    out = tmp_path / "out.jsonl"
    cmd = [
        sys.executable,
        "scripts/retokenize_with_staff_markers.py",
        "--input-manifest", str(src),
        "--source-dataset", "primus",
        "--target-dataset", "primus_systems",
        "--output-manifest", str(out),
    ]
    result = subprocess.run(cmd, cwd="/home/ari/work/Clarity-OMR-Train-RADIO",
                            capture_output=True, text=True)
    # Should exit non-zero (malformed input is a builder bug, not silent skip)
    assert result.returncode != 0
