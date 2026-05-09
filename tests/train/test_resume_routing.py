"""Resume routing: _compute_resume_position helper.

Covers the legacy-checkpoint detection that lets v1/v2 Stage 3 checkpoints
saved before commit 715a89b (when `stage_step` field changed from
micro-batch units to opt-step units) be safely resumed without silently
skipping the entire stage via the `stage_start_step > stage_total_steps`
guard at train.py:2370.
"""
from __future__ import annotations
from types import SimpleNamespace


def _make_stages_and_runtime():
    """Three-stage fixture matching production layout (Stage 1 / 2 / 3)."""
    stages = [
        SimpleNamespace(stage_name="stage1"),
        SimpleNamespace(stage_name="stage2"),
        SimpleNamespace(stage_name="stage3-radio-systems-frozen-encoder"),
    ]
    stage_runtime = [
        {"stage_total_steps": 1000},
        {"stage_total_steps": 4000},
        {"stage_total_steps": 6000},
    ]
    return stages, stage_runtime


def test_compute_resume_position_no_checkpoint_returns_zero_zero():
    """Without a checkpoint payload, the helper returns (0, 0) — start of stage 0."""
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    idx, completed = _compute_resume_position(
        checkpoint_payload=None,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="",
        global_step=0,
    )
    assert idx == 0
    assert completed == 0


def test_compute_resume_position_new_convention_uses_stage_step():
    """Post-715a89b checkpoints store `stage_step` in opt-step units.
    When stage_name matches and stage_step ≤ stage_steps_total, the helper
    trusts the field directly — supports the spec's incremental extension
    protocol (raise YAML target 4500 → 6000 → 7500 and resume mid-stage).
    """
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    payload = {
        "stage_name": "stage3-radio-systems-frozen-encoder",
        "stage_step": 4500,            # opt-step units (post-715a89b)
        "stage_steps_total": 6000,     # 4500 ≤ 6000 → not inflated
        "global_step": 5000 + 4500,
    }
    idx, completed = _compute_resume_position(
        checkpoint_payload=payload,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="stage3-radio-systems-frozen-encoder",
        global_step=5000 + 4500,
    )
    assert idx == 2
    assert completed == 4500


def test_compute_resume_position_legacy_checkpoint_falls_back_to_walk(capsys):
    """Pre-715a89b checkpoints have `stage_step` in micro-batch units (e.g. 7650
    for a 4500-opt-step run with grad_accum_live=8 and 90/10 cached/live mix).

    `stage_step > stage_steps_total` is impossible in the new opt-step convention,
    so the helper treats it as a legacy-units signal and falls back to the
    legacy global_step walk. Without this fallback, resume_stage_completed_steps
    would be 7650 → stage_start_step=7651 > stage_total_steps=6000 → the stage
    is silently skipped via the resume-skip path at train.py:2370.
    """
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    payload = {
        "stage_name": "stage3-radio-systems-frozen-encoder",
        "stage_step": 7650,            # MICRO-batch units (pre-715a89b)
        "stage_steps_total": 4500,     # 7650 > 4500 → legacy signal
        "global_step": 5000 + 4500,
    }
    idx, completed = _compute_resume_position(
        checkpoint_payload=payload,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="stage3-radio-systems-frozen-encoder",
        global_step=5000 + 4500,
    )
    # Walk: global_step=9500; stage1 consumes 1000, stage2 consumes 4000,
    # remaining=4500 lands in stage3 with completed=4500.
    assert idx == 2
    assert completed == 4500

    # Helper prints a detection notice so the operator knows why the
    # routing fell back instead of trusting stage_step.
    captured = capsys.readouterr()
    assert "legacy" in captured.err.lower() or "micro-batch" in captured.err.lower()


def test_compute_resume_position_no_stage_name_uses_walk():
    """Checkpoints lacking a stage_name fall back to the global_step walk.
    This preserves backwards compatibility with even older multi-stage
    checkpoints that pre-date the stage_name field."""
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    payload = {
        "stage_step": 4500,
        "stage_steps_total": 6000,
        "global_step": 1000 + 4000 + 2500,
    }
    idx, completed = _compute_resume_position(
        checkpoint_payload=payload,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="",
        global_step=1000 + 4000 + 2500,
    )
    # Walk: stage1 consumes 1000, stage2 consumes 4000, remaining=2500 in stage3.
    assert idx == 2
    assert completed == 2500


def test_compute_resume_position_missing_stage_steps_total_trusts_stage_step():
    """A checkpoint with stage_step but no stage_steps_total cannot be unit-
    checked (the legacy detection signal `stage_step > stage_steps_total`
    requires the latter). The helper trusts stage_step as opt-step units in
    this case — practical for any post-63f9c4e checkpoint, where the field
    is always written. Pre-63f9c4e checkpoints predate Stage 3 entirely
    and would not match a Stage 3 stage_name, so they fall through to the
    legacy walk regardless.

    This test pins down the documented behavior so a future refactor of
    the unit-detection logic doesn't accidentally invert this branch.
    """
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    payload = {
        "stage_name": "stage3-radio-systems-frozen-encoder",
        "stage_step": 4500,
        # stage_steps_total intentionally absent
        "global_step": 5000 + 4500,
    }
    idx, completed = _compute_resume_position(
        checkpoint_payload=payload,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="stage3-radio-systems-frozen-encoder",
        global_step=5000 + 4500,
    )
    assert idx == 2
    assert completed == 4500


def test_compute_resume_position_unknown_stage_name_uses_walk(capsys):
    """An unknown stage_name (not present in the YAML) falls back to walk."""
    from src.train.train import _compute_resume_position

    stages, stage_runtime = _make_stages_and_runtime()
    payload = {
        "stage_name": "stage-removed-from-yaml",
        "stage_step": 4500,
        "stage_steps_total": 6000,
        "global_step": 1000 + 4000 + 1000,
    }
    idx, completed = _compute_resume_position(
        checkpoint_payload=payload,
        stages=stages,
        stage_runtime=stage_runtime,
        resume_stage_name="stage-removed-from-yaml",
        global_step=1000 + 4000 + 1000,
    )
    assert idx == 2
    assert completed == 1000
