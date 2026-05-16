"""Unit test for eval/score_stage_a_only.py."""
import json
from pathlib import Path

import pytest


def _write_manifest(out_dir: Path, piece: str, page0_boxes: int) -> None:
    """Write a fake manifest with `page0_boxes` page-0 detections + 1 page-1 box."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{piece}_stage_a.jsonl"
    with out_path.open("w") as f:
        for i in range(page0_boxes):
            f.write(json.dumps({
                "piece": piece, "page": 0,
                "bbox_xyxy": [10, 10 + i*100, 90, 90 + i*100],
                "conf": 0.9,
                "page_width": 100, "page_height": 1000,
            }) + "\n")
        f.write(json.dumps({
            "piece": piece, "page": 1,
            "bbox_xyxy": [10, 10, 90, 90],
            "conf": 0.9,
            "page_width": 100, "page_height": 100,
        }) + "\n")


def _write_fake_score(scores_dir: Path, piece: str, n_parts: int) -> None:
    """Write a minimal MusicXML with n_parts parts."""
    import music21
    scores_dir.mkdir(parents=True, exist_ok=True)
    s = music21.stream.Score()
    for i in range(n_parts):
        p = music21.stream.Part()
        p.partName = f"Part{i}"
        m = music21.stream.Measure()
        m.append(music21.note.Note("C4", quarterLength=4))
        p.append(m)
        s.append(p)
    s.write("musicxml", fp=str(scores_dir / f"{piece}.mxl"))


def test_scores_each_piece(tmp_path: Path):
    """For one manifest with 3 page-0 boxes vs an MXL with 3 parts → missing=0."""
    from eval.score_stage_a_only import score_run

    manifests = tmp_path / "manifests"
    scores = tmp_path / "scores"
    out_csv = tmp_path / "recall.csv"

    _write_manifest(manifests, "piece_a", page0_boxes=3)
    _write_fake_score(scores, "piece_a", n_parts=3)

    score_run(manifest_dir=manifests, scores_dir=scores, out_csv=out_csv)

    rows = out_csv.read_text().strip().splitlines()
    assert len(rows) == 2  # header + one piece
    header = rows[0].split(",")
    assert header == ["piece", "expected_p1_staves", "detected_p1_staves", "missing_count"]
    data = rows[1].split(",")
    assert data[0] == "piece_a"
    assert int(data[1]) == 3
    assert int(data[2]) == 3
    assert int(data[3]) == 0


def test_missing_count_when_underdetected(tmp_path: Path):
    """1 page-0 box vs 3-part score → missing_count=2."""
    from eval.score_stage_a_only import score_run

    manifests = tmp_path / "m"
    scores = tmp_path / "s"
    out_csv = tmp_path / "r.csv"

    _write_manifest(manifests, "p", page0_boxes=1)
    _write_fake_score(scores, "p", n_parts=3)

    score_run(manifest_dir=manifests, scores_dir=scores, out_csv=out_csv)
    rows = out_csv.read_text().strip().splitlines()
    data = rows[1].split(",")
    assert int(data[3]) == 2


def test_skips_pieces_without_score(tmp_path: Path):
    """A manifest with no matching MXL is skipped (not an error)."""
    from eval.score_stage_a_only import score_run

    manifests = tmp_path / "m"
    scores = tmp_path / "s"
    scores.mkdir()
    out_csv = tmp_path / "r.csv"

    _write_manifest(manifests, "no_score", page0_boxes=2)

    score_run(manifest_dir=manifests, scores_dir=scores, out_csv=out_csv)
    rows = out_csv.read_text().strip().splitlines()
    assert len(rows) == 1  # header only


def test_overdetected_is_zero_missing(tmp_path: Path):
    """5 page-0 boxes vs 3-part score → missing_count=0 (not negative)."""
    from eval.score_stage_a_only import score_run

    manifests = tmp_path / "m"
    scores = tmp_path / "s"
    out_csv = tmp_path / "r.csv"

    _write_manifest(manifests, "p", page0_boxes=5)
    _write_fake_score(scores, "p", n_parts=3)

    score_run(manifest_dir=manifests, scores_dir=scores, out_csv=out_csv)
    rows = out_csv.read_text().strip().splitlines()
    data = rows[1].split(",")
    assert int(data[3]) == 0


def _write_fake_score_nested(scores_dir: Path, composer: str, title: str, piece: str, n_parts: int) -> None:
    """Write a minimal MusicXML nested under scores_dir/<Composer>/_/<Title>/<piece>.mxl."""
    import music21
    nested = scores_dir / composer / "_" / title
    nested.mkdir(parents=True, exist_ok=True)
    s = music21.stream.Score()
    for i in range(n_parts):
        p = music21.stream.Part()
        p.partName = f"Part{i}"
        m = music21.stream.Measure()
        m.append(music21.note.Note("C4", quarterLength=4))
        p.append(m)
        s.append(p)
    s.write("musicxml", fp=str(nested / f"{piece}.mxl"))


def test_nested_openscore_layout_is_found(tmp_path: Path):
    """Pieces nested like <Composer>/_/<Title>/<piece>.mxl must be found and scored.

    This is the real openscore_lieder directory structure; flat lookup silently
    skips every piece (the bug this test covers).
    """
    from eval.score_stage_a_only import score_run

    manifests = tmp_path / "manifests"
    scores = tmp_path / "scores"
    out_csv = tmp_path / "recall.csv"

    piece = "lc28688206"
    _write_manifest(manifests, piece, page0_boxes=2)
    _write_fake_score_nested(scores, "Brahms", "LiederOp49", piece, n_parts=2)

    score_run(manifest_dir=manifests, scores_dir=scores, out_csv=out_csv)

    rows = out_csv.read_text().strip().splitlines()
    # Must have header + one data row (not just header-only, which is the bug)
    assert len(rows) == 2, f"Expected header + 1 data row, got {len(rows)} rows: {rows}"
    data = rows[1].split(",")
    assert data[0] == piece
    assert int(data[1]) == 2
    assert int(data[2]) == 2
    assert int(data[3]) == 0


def test_main_cli_writes_csv(tmp_path: Path, monkeypatch):
    from eval import score_stage_a_only

    manifests = tmp_path / "m"
    scores = tmp_path / "s"
    out_csv = tmp_path / "r.csv"

    _write_manifest(manifests, "x", page0_boxes=2)
    _write_fake_score(scores, "x", n_parts=2)

    monkeypatch.setattr(
        "sys.argv",
        [
            "score_stage_a_only.py",
            "--manifest-dir", str(manifests),
            "--scores-dir", str(scores),
            "--out-csv", str(out_csv),
        ],
    )
    score_stage_a_only.main()

    assert out_csv.exists()
    text = out_csv.read_text()
    assert "x,2,2,0" in text
