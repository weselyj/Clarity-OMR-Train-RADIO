"""Tests for src/data/kern_validation.py."""
from __future__ import annotations

from src.data.kern_validation import CanonicalEvent, Divergence, CompareResult


def test_canonical_event_dataclass() -> None:
    e = CanonicalEvent(offset_ql=0.0, kind="note", payload=("C4", 1.0), staff_idx=0)
    assert e.kind == "note"
    assert e.payload == ("C4", 1.0)


def test_divergence_dataclass() -> None:
    d = Divergence(
        kind="tie_open",
        offset_ql=2.0,
        staff_idx=0,
        ref_value=True,
        our_value=None,
        note="missing tie_open",
    )
    assert d.kind == "tie_open"
    assert d.our_value is None


def test_compare_result_passed_when_no_divergences() -> None:
    r = CompareResult(
        kern_path=__import__("pathlib").Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[],
    )
    assert r.passed is True


def test_compare_result_not_passed_when_divergences_exist() -> None:
    from pathlib import Path
    r = CompareResult(
        kern_path=Path("/x.krn"),
        ref_canonical=[],
        our_canonical=[],
        divergences=[
            Divergence(kind="note", offset_ql=0.0, staff_idx=0, ref_value="C4", our_value="C5", note="")
        ],
    )
    assert r.passed is False
