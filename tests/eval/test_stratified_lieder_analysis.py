# tests/eval/test_stratified_lieder_analysis.py
"""Stratified onset_f1 analysis: group lieder eval CSV by staves_in_system,
emit per-bucket means, run lc6548281 sanity check.
"""
from __future__ import annotations
import csv
from pathlib import Path
import pytest


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_stratified_groups_by_staves_in_system(tmp_path):
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
        {"piece": "p2", "staves_in_system": "1", "onset_f1": "0.6"},
        {"piece": "p3", "staves_in_system": "2", "onset_f1": "0.3"},
        {"piece": "p4", "staves_in_system": "2", "onset_f1": "0.4"},
        {"piece": "p5", "staves_in_system": "3", "onset_f1": "0.2"},
    ])

    result = analyze(csv_path)

    assert result.per_bucket == {
        "1": pytest.approx(0.55),
        "2": pytest.approx(0.35),
        "3": pytest.approx(0.20),
    }
    assert result.bucket_counts == {"1": 2, "2": 2, "3": 1}
    assert result.overall_mean == pytest.approx(0.4)


def test_stratified_lc6548281_sanity_check(tmp_path):
    """The architectural sanity check from spec line 241: lc6548281 should
    improve from DaViT's 0.05 to >= 0.10."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "lc6548281", "staves_in_system": "2", "onset_f1": "0.18"},
        {"piece": "other", "staves_in_system": "1", "onset_f1": "0.5"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 == pytest.approx(0.18)
    assert result.lc6548281_passes_sanity is True


def test_stratified_lc6548281_fails_sanity_when_below_threshold(tmp_path):
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "lc6548281", "staves_in_system": "2", "onset_f1": "0.08"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 == pytest.approx(0.08)
    assert result.lc6548281_passes_sanity is False


def test_stratified_lc6548281_missing_returns_none(tmp_path):
    """If lc6548281 isn't in the eval set (e.g. a smoke run), the field is
    None, not raising — the decision-gate consumer treats this as 'not
    evaluated' rather than a failure."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
    ])

    result = analyze(csv_path)

    assert result.lc6548281_onset_f1 is None
    assert result.lc6548281_passes_sanity is None


def test_stratified_skips_rows_with_missing_onset_f1(tmp_path):
    """Pieces that errored during scoring have onset_f1 empty/None.
    They're excluded from the buckets so the means aren't poisoned."""
    from eval.stratified_lieder_analysis import analyze

    csv_path = tmp_path / "lieder.csv"
    _write_csv(csv_path, [
        {"piece": "p1", "staves_in_system": "1", "onset_f1": "0.5"},
        {"piece": "p2", "staves_in_system": "1", "onset_f1": ""},
        {"piece": "p3", "staves_in_system": "1", "onset_f1": "0.7"},
    ])

    result = analyze(csv_path)

    assert result.per_bucket["1"] == pytest.approx(0.6)
    assert result.bucket_counts["1"] == 2
