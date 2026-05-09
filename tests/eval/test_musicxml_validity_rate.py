"""Tests for MusicXML validity rate metric (Phase 2 metric, enabled in Plan C / Task 10).

Spec ref: docs/superpowers/plans/2026-05-09-radio-stage3-phase1-training.md §"3. MusicXML
validity rate" (line 264). Stage 2 v2 left this metric at None because the eval driver
never decoded predicted tokens to MusicXML; Stage 3 enables the codepath so Plan D
(Phase 2 evaluation) can read the rate out of the standard summary JSON.

The validity check is intentionally light: write the music21-rendered score to a
MusicXML byte string and verify it parses with ``xml.etree.ElementTree.fromstring``
and has a MusicXML-shaped root tag (``score-partwise`` or ``score-timewise``). No
schema validation, no music21 round-trip on the produced XML — that would be a
different metric (``musicxml_musical_similarity``) and uses different inputs.
"""

from __future__ import annotations

from typing import List, Sequence

import pytest


SIMPLE_VALID_TOKENS: List[str] = [
    "<staff_start>",
    "<measure_start>",
    "note-C4",
    "_quarter",
    "note-D4",
    "_quarter",
    "note-E4",
    "_quarter",
    "note-F4",
    "_quarter",
    "<measure_end>",
    "<staff_end>",
]

# Empty token sequence – append_tokens_to_part produces an empty Part. The
# resulting MusicXML still parses (music21 always emits a <score-partwise> root)
# so this is a *valid* but musically-empty case.
EMPTY_TOKENS: List[str] = []


def test_musicxml_validity_from_tokens_all_valid() -> None:
    """All sequences yield parseable MusicXML → rate is 1.0."""
    from src.eval.metrics import musicxml_validity_from_tokens

    rate = musicxml_validity_from_tokens([SIMPLE_VALID_TOKENS, SIMPLE_VALID_TOKENS])
    assert rate == 1.0


def test_musicxml_validity_from_tokens_empty_list_returns_none() -> None:
    """No token sequences provided → metric is undefined; return None
    (matches the existing ``musicxml_validity`` semantics in evaluate_rows)."""
    from src.eval.metrics import musicxml_validity_from_tokens

    rate = musicxml_validity_from_tokens([])
    assert rate is None


def test_musicxml_validity_from_tokens_invalid_drops_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """When token-to-MusicXML conversion fails for some sequences, the rate
    drops proportionally. We simulate failure by patching the exporter to
    raise on the second call.

    This guards the metric against a regression where a crash in the decoder
    is silently treated as a *successful* eval (rate stays 1.0)."""
    from src.eval import metrics as metrics_mod

    call_count = {"n": 0}
    real = metrics_mod._render_tokens_to_musicxml_bytes  # type: ignore[attr-defined]

    def flaky(tokens: Sequence[str]):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated decoder failure")
        return real(tokens)

    monkeypatch.setattr(metrics_mod, "_render_tokens_to_musicxml_bytes", flaky)

    rate = metrics_mod.musicxml_validity_from_tokens(
        [SIMPLE_VALID_TOKENS, SIMPLE_VALID_TOKENS, SIMPLE_VALID_TOKENS]
    )
    assert rate == pytest.approx(2.0 / 3.0)


def test_musicxml_validity_from_tokens_bad_xml_drops_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the rendered bytes parse but lack a MusicXML root tag, the
    sequence is invalid. This catches regressions where the renderer emits
    a non-MusicXML XML document (e.g. an error snippet)."""
    from src.eval import metrics as metrics_mod

    call_count = {"n": 0}
    real = metrics_mod._render_tokens_to_musicxml_bytes  # type: ignore[attr-defined]

    def half_bad(tokens: Sequence[str]):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return b"<?xml version='1.0'?><not-musicxml>nope</not-musicxml>"
        return real(tokens)

    monkeypatch.setattr(metrics_mod, "_render_tokens_to_musicxml_bytes", half_bad)

    rate = metrics_mod.musicxml_validity_from_tokens([SIMPLE_VALID_TOKENS, SIMPLE_VALID_TOKENS])
    assert rate == 0.5


def test_evaluate_rows_emits_musicxml_validity_rate_from_pred_tokens() -> None:
    """End-to-end: evaluate_rows must populate musicxml_validity_rate from
    pred_tokens when no pred_musicxml_path is provided. This is the actual
    spec deliverable — Plan D reads ``musicxml_validity_rate`` out of the
    summary JSON.
    """
    from src.eval.run_eval import evaluate_rows

    rows = [
        {
            "pred_tokens": SIMPLE_VALID_TOKENS,
            "gt_tokens": SIMPLE_VALID_TOKENS,
            "dataset": "synthetic_systems",
        },
        {
            "pred_tokens": SIMPLE_VALID_TOKENS,
            "gt_tokens": SIMPLE_VALID_TOKENS,
            "dataset": "synthetic_systems",
        },
    ]
    summary = evaluate_rows(rows)
    assert "musicxml_validity_rate" in summary
    assert summary["musicxml_validity_rate"] is not None
    assert summary["musicxml_validity_rate"] == 1.0
