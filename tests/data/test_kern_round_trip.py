"""Round-trip parity test for the kern converter against music21's humdrum parser.

Runs compare_via_music21 on a small set of curated .krn fixtures that exercise the
common kern features. Asserts zero divergences for in-vocab features.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.data.kern_validation import compare_via_music21


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "kern"


def _list_fixtures():
    if not FIXTURES_DIR.exists():
        return []
    return sorted(FIXTURES_DIR.glob("*.krn"))


@pytest.mark.parametrize("kern_path", _list_fixtures(), ids=lambda p: p.name)
def test_round_trip_parity(kern_path: Path) -> None:
    result = compare_via_music21(kern_path)
    if not result.passed:
        msgs = [
            f"  staff={d.staff_idx} offset={d.offset_ql} kind={d.kind} ref={d.ref_value!r} our={d.our_value!r} ({d.note})"
            for d in result.divergences[:20]
        ]
        pytest.fail(
            f"{kern_path.name}: {len(result.divergences)} divergences:\n" + "\n".join(msgs)
        )
