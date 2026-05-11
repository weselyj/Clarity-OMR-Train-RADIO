"""Regression tests for _enforce_global_key_time first-emission-wins semantics.

The time-signature glyph is typically printed only on the first staff of the
first system. The decoder reads it correctly there but has to hallucinate on
later staves with no glyph. The old majority-vote drowned the informed first
emission in N-1 uninformed guesses; first-emission-wins keeps the signal.
"""
from __future__ import annotations


def _staff(sample_id: str, key: str, time: str, *extra: str):
    from src.pipeline.assemble_score import StaffLocation, StaffRecognitionResult

    return StaffRecognitionResult(
        sample_id=sample_id,
        tokens=[f"keySignature-{key}", f"timeSignature-{time}", *extra],
        location=StaffLocation(0, 0.0, 1.0, 0.0, 1.0),
    )


def test_first_staff_overrides_majority_for_time_signature():
    """Staff 0 emits 9/8 (informed by visible glyph); 4 later staves emit 4/4
    (hallucinated prior). First-emission-wins keeps 9/8 globally even though
    4/4 is the 4:1 majority."""
    from src.pipeline.assemble_score import _enforce_global_key_time

    staves = [
        _staff("s0", "DbM", "9/8"),
        _staff("s1", "DbM", "4/4"),
        _staff("s2", "DbM", "4/4"),
        _staff("s3", "DbM", "4/4"),
        _staff("s4", "DbM", "4/4"),
    ]

    fixed = _enforce_global_key_time(staves)

    for f in fixed:
        assert "timeSignature-9/8" in f.tokens, f"{f.sample_id} missing 9/8"
        assert "timeSignature-4/4" not in f.tokens, f"{f.sample_id} still has 4/4"


def test_first_staff_overrides_majority_for_key_signature():
    from src.pipeline.assemble_score import _enforce_global_key_time

    staves = [
        _staff("s0", "EM", "4/4"),
        _staff("s1", "CM", "4/4"),
        _staff("s2", "CM", "4/4"),
        _staff("s3", "CM", "4/4"),
    ]

    fixed = _enforce_global_key_time(staves)

    for f in fixed:
        assert "keySignature-EM" in f.tokens
        assert "keySignature-CM" not in f.tokens


def test_falls_through_when_first_staff_emits_no_signature():
    """If staff 0 didn't emit a time/key token, use the first staff that did."""
    from src.pipeline.assemble_score import (
        StaffLocation,
        StaffRecognitionResult,
        _enforce_global_key_time,
    )

    staves = [
        StaffRecognitionResult(
            sample_id="s0",
            tokens=["clef-G2"],  # no key, no time
            location=StaffLocation(0, 0.0, 1.0, 0.0, 1.0),
        ),
        _staff("s1", "EbM", "3/4"),
        _staff("s2", "EbM", "4/4"),
    ]

    fixed = _enforce_global_key_time(staves)

    # s2's 4/4 should be rewritten to s1's 3/4
    assert "timeSignature-3/4" in fixed[2].tokens
    assert "timeSignature-4/4" not in fixed[2].tokens


def test_no_signatures_anywhere_is_a_noop():
    from src.pipeline.assemble_score import (
        StaffLocation,
        StaffRecognitionResult,
        _enforce_global_key_time,
    )

    staves = [
        StaffRecognitionResult(
            sample_id=f"s{i}",
            tokens=["clef-G2", "<measure_start>", "rest", "_whole", "<measure_end>"],
            location=StaffLocation(0, 0.0, 1.0, 0.0, 1.0),
        )
        for i in range(3)
    ]

    fixed = _enforce_global_key_time(staves)

    # Tokens unchanged (no key/time emissions to rewrite)
    for original, after in zip(staves, fixed):
        assert original.tokens == after.tokens
