# tests/eval/test_decision_gate.py
"""Decision gate: aggregates lieder + per-dataset + validity into a verdict."""
from __future__ import annotations
import pytest

from dataclasses import asdict


def _mk_inputs(**overrides):
    """Build a complete, all-passing input dict; override specific fields per test."""
    base = {
        "lieder_mean_onset_f1": 0.32,
        "musicxml_validity_rate": 0.95,
        "per_dataset": {
            "synthetic_systems": 92.0,
            "grandstaff_systems": 96.0,
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 76.0,
            "cameraprimus": 76.0,
        },
        "cameraprimus_systems_baseline": 75.2,
        "cameraprimus_baseline": 75.2,
        "lc6548281_onset_f1": 0.15,
    }
    base.update(overrides)
    return base


def test_all_pass_strong_lieder_yields_ship():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs())

    assert result.verdict == Verdict.SHIP
    assert result.lieder_outcome == "Strong"
    assert all(v.passed for v in result.per_dataset_results.values())


def test_all_floors_pass_mixed_lieder_yields_investigate():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.27))

    assert result.verdict == Verdict.INVESTIGATE
    assert result.lieder_outcome == "Mixed"


def test_all_floors_pass_flat_lieder_yields_pivot():
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.20))

    assert result.verdict == Verdict.PIVOT
    assert result.lieder_outcome == "Flat"


def test_lieder_at_strong_threshold_is_strong():
    """Boundary check: 0.30 exactly is Strong (>= 0.30 per spec line 235)."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.30))

    assert result.lieder_outcome == "Strong"
    assert result.verdict == Verdict.SHIP


def test_lieder_at_mixed_threshold_is_mixed():
    """Boundary: 0.241 is Mixed (>= 0.241 per spec line 236)."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.2410))

    assert result.lieder_outcome == "Mixed"


def test_lieder_below_mixed_is_flat():
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(lieder_mean_onset_f1=0.2409))

    assert result.lieder_outcome == "Flat"


def test_per_dataset_floor_fail_yields_diagnose_regardless_of_lieder():
    """A regression on grandstaff_systems means Stage 3 broke something
    Stage 2 v2 had. Don't draw lieder conclusions over a broken baseline.
    Spec §"Decision flow" line 285."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(
        lieder_mean_onset_f1=0.40,                          # Strong, but...
        per_dataset={
            "synthetic_systems": 92.0,
            "grandstaff_systems": 80.0,                     # FAILS floor 95
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 76.0,
            "cameraprimus": 76.0,
        },
    ))

    assert result.verdict == Verdict.DIAGNOSE
    assert result.per_dataset_results["grandstaff_systems"].passed is False
    assert result.per_dataset_results["grandstaff_systems"].floor == 95.0


def test_cameraprimus_floor_lifts_when_baseline_re_eval_higher():
    """Spec §"Phase 2 §2" line 261: 'cameraprimus ≥ 75 if 200-sample eval
    confirms baseline at 75.2; raise floor accordingly if eval shows higher.'
    Implementation: floor = max(75, baseline - 5). Same dynamic for cameraprimus_systems."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        per_dataset={
            "synthetic_systems": 92.0,
            "grandstaff_systems": 96.0,
            "grandstaff": 92.0,
            "primus_systems": 82.0,
            "primus": 82.0,
            "cameraprimus_systems": 79.0,
            "cameraprimus": 79.0,
        },
        cameraprimus_systems_baseline=85.0,
        cameraprimus_baseline=85.0,
    ))

    # Floor = max(75, 85.0 - 5) = 80; both cameraprimus variants at 79 fail.
    assert result.per_dataset_results["cameraprimus"].floor == 80.0
    assert result.per_dataset_results["cameraprimus"].passed is False
    assert result.per_dataset_results["cameraprimus_systems"].floor == 80.0
    assert result.per_dataset_results["cameraprimus_systems"].passed is False


def test_cameraprimus_floor_holds_at_75_when_baseline_at_or_below_baseline():
    """When the re-eval comes in at or below 80 (75 + 5), floor stays at 75."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        cameraprimus_systems_baseline=78.0,
        cameraprimus_baseline=78.0,
    ))

    assert result.per_dataset_results["cameraprimus"].floor == 75.0
    assert result.per_dataset_results["cameraprimus_systems"].floor == 75.0


def test_cameraprimus_variants_can_have_different_dynamic_floors():
    """The two cameraprimus variants are evaluated on different manifests
    (token_manifest_full.jsonl single-staff vs token_manifest_stage3.jsonl
    _systems). Stage 2 v2 baselines may differ — each variant's floor uses
    its own baseline."""
    from eval.decision_gate import evaluate

    result = evaluate(**_mk_inputs(
        cameraprimus_systems_baseline=82.0,   # higher → floor 77
        cameraprimus_baseline=78.0,           # lower → floor 75
    ))

    assert result.per_dataset_results["cameraprimus_systems"].floor == 77.0
    assert result.per_dataset_results["cameraprimus"].floor == 75.0


def test_musicxml_validity_is_recorded_but_not_gated():
    """Spec line 266: corroborating signal, not gated. A low validity rate
    annotates the report but doesn't change the verdict."""
    from eval.decision_gate import evaluate, Verdict

    result = evaluate(**_mk_inputs(musicxml_validity_rate=0.20))  # very low

    assert result.musicxml_validity_rate == 0.20
    # All other checks pass + lieder Strong → Ship, despite low validity.
    assert result.verdict == Verdict.SHIP


def test_render_report_contains_verdict_and_evidence():
    from eval.decision_gate import evaluate, render_report

    result = evaluate(**_mk_inputs())
    report = render_report(result)

    assert "SHIP" in report
    assert "0.32" in report or "0.3200" in report
    assert "synthetic_systems" in report
    assert "grandstaff_systems" in report
