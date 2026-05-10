# eval/decision_gate.py
"""Phase 2 decision gate for RADIO Stage 3.

Aggregates the three eval surfaces from spec §"Phase 2" — lieder onset_f1,
per-dataset quality regression-check, MusicXML validity rate (corroborating)
— plus the cameraprimus baseline re-eval, into one of four verdicts:

    Ship       — all per-dataset floors PASS + lieder Strong (≥ 0.30)
    Investigate — all floors PASS + lieder Mixed (0.241 ≤ x < 0.30)
    Pivot      — all floors PASS + lieder Flat (< 0.241)
    Diagnose   — any floor FAILS (regardless of lieder)

Thresholds are spec constants (locked here; a spec revision is a one-file
edit). Inputs come from Tasks 1-6 outputs; the script aggregates and writes
a markdown report at eval/results/decision_gate_<name>.md.
"""
from __future__ import annotations
import argparse
import enum
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


# Spec §"Phase 2 §1" line 235-237.
STRONG_THRESHOLD = 0.30
MIXED_THRESHOLD = 0.2410

# Spec §"Phase 2 §2" line 254-262. Per-dataset Stage 3 quality floors (revised
# 2026-05-10 per Plan D Decision #4 revision — single-staff variants of
# grandstaff and primus are confirming context, not gating, since Stage 3 was
# trained on _systems only and the single-staff regression is by design;
# cameraprimus single-staff retained because it doubles as a regression
# tripwire for the cameraprimus baseline.).
PER_DATASET_FLOORS = {
    "synthetic_systems": 90.0,
    "grandstaff_systems": 95.0,
    "primus_systems": 80.0,
    # cameraprimus + cameraprimus_systems are dynamic — see _resolve_cameraprimus_floor().
}
CAMERAPRIMUS_FLOOR_BASE = 75.0
CAMERAPRIMUS_REGRESSION_TOLERANCE = 5.0


class Verdict(str, enum.Enum):
    SHIP = "Ship"
    INVESTIGATE = "Investigate"
    PIVOT = "Pivot"
    DIAGNOSE = "Diagnose"


@dataclass(frozen=True)
class FloorResult:
    dataset: str
    measured: float
    floor: float
    passed: bool


@dataclass(frozen=True)
class GateResult:
    verdict: Verdict
    lieder_mean_onset_f1: float
    lieder_outcome: str  # "Strong" | "Mixed" | "Flat"
    musicxml_validity_rate: float
    per_dataset_results: Dict[str, FloorResult]
    cameraprimus_systems_baseline: float
    cameraprimus_baseline: float
    lc6548281_onset_f1: Optional[float]


def _resolve_cameraprimus_floor(baseline_re_eval: float) -> float:
    """Spec line 261: floor = 75 if re-eval confirms 75.2; raise to
    re_eval - 5 if re-eval is higher. Used for both cameraprimus
    (single-staff) and cameraprimus_systems variants — each gets its
    own baseline + dynamic floor."""
    return max(CAMERAPRIMUS_FLOOR_BASE, baseline_re_eval - CAMERAPRIMUS_REGRESSION_TOLERANCE)


def _classify_lieder(mean_onset_f1: float) -> str:
    if mean_onset_f1 >= STRONG_THRESHOLD:
        return "Strong"
    if mean_onset_f1 >= MIXED_THRESHOLD:
        return "Mixed"
    return "Flat"


def evaluate(
    *,
    lieder_mean_onset_f1: float,
    musicxml_validity_rate: float,
    per_dataset: Dict[str, float],
    cameraprimus_systems_baseline: float,
    cameraprimus_baseline: float,
    lc6548281_onset_f1: Optional[float] = None,
) -> GateResult:
    """Aggregate the inputs into a verdict.

    Inputs:
        lieder_mean_onset_f1: from eval/results/lieder_<name>.csv
        musicxml_validity_rate: from same CSV (corroborating only)
        per_dataset: dict {dataset_name: composite quality_score, 0-100 scale};
            5 gated keys (synthetic_systems, grandstaff_systems, primus_systems,
            cameraprimus_systems, cameraprimus). Single-staff grandstaff/primus
            may be present in the dict but are not gated (Plan D Decision #4
            revision — confirming context only).
        cameraprimus_systems_baseline: Stage 2 v2 quality on token_manifest_stage3.jsonl
        cameraprimus_baseline: Stage 2 v2 quality on token_manifest_full.jsonl
        lc6548281_onset_f1: optional sanity-check value (None if not evaluated)
    """
    floors = dict(PER_DATASET_FLOORS)
    floors["cameraprimus_systems"] = _resolve_cameraprimus_floor(cameraprimus_systems_baseline)
    floors["cameraprimus"] = _resolve_cameraprimus_floor(cameraprimus_baseline)

    per_dataset_results = {}
    any_floor_fail = False
    for dataset, floor in floors.items():
        measured = per_dataset.get(dataset)
        if measured is None:
            # Treat missing-input as floor-fail (decision-gate is invoked
            # only after Task 2 produces all 7; missing means upstream broke).
            per_dataset_results[dataset] = FloorResult(dataset=dataset, measured=float("nan"), floor=floor, passed=False)
            any_floor_fail = True
            continue
        passed = measured >= floor
        per_dataset_results[dataset] = FloorResult(dataset=dataset, measured=measured, floor=floor, passed=passed)
        if not passed:
            any_floor_fail = True

    lieder_outcome = _classify_lieder(lieder_mean_onset_f1)

    if any_floor_fail:
        verdict = Verdict.DIAGNOSE
    elif lieder_outcome == "Strong":
        verdict = Verdict.SHIP
    elif lieder_outcome == "Mixed":
        verdict = Verdict.INVESTIGATE
    else:
        verdict = Verdict.PIVOT

    return GateResult(
        verdict=verdict,
        lieder_mean_onset_f1=lieder_mean_onset_f1,
        lieder_outcome=lieder_outcome,
        musicxml_validity_rate=musicxml_validity_rate,
        per_dataset_results=per_dataset_results,
        cameraprimus_systems_baseline=cameraprimus_systems_baseline,
        cameraprimus_baseline=cameraprimus_baseline,
        lc6548281_onset_f1=lc6548281_onset_f1,
    )


def render_report(result: GateResult, *, name: str = "stage3_v2") -> str:
    lines = []
    lines.append(f"# Decision Gate Report — {name}")
    lines.append("")
    lines.append(f"## Verdict: **{result.verdict.value.upper()}**")
    lines.append("")
    lines.append(f"- Lieder mean onset_f1: **{result.lieder_mean_onset_f1:.4f}** ({result.lieder_outcome})")
    if result.lc6548281_onset_f1 is not None:
        lines.append(f"- lc6548281 sanity-check onset_f1: {result.lc6548281_onset_f1:.4f} (threshold ≥ 0.10)")
    lines.append(f"- MusicXML validity rate (corroborating): {result.musicxml_validity_rate:.3f}")
    lines.append("")
    lines.append("## Per-dataset quality regression-check")
    lines.append("")
    lines.append("| Dataset | Measured | Floor | Status |")
    lines.append("|---|---|---|---|")
    for ds in ["synthetic_systems", "grandstaff_systems", "primus_systems", "cameraprimus_systems", "cameraprimus"]:
        r = result.per_dataset_results.get(ds)
        if r is None:
            lines.append(f"| {ds} | — | — | MISSING |")
            continue
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"| {ds} | {r.measured:.2f} | {r.floor:.2f} | {status} |")
    lines.append("")
    cps_floor = result.per_dataset_results["cameraprimus_systems"].floor
    cp_floor = result.per_dataset_results["cameraprimus"].floor
    lines.append(f"_cameraprimus_systems floor = max(75, {result.cameraprimus_systems_baseline:.2f} - 5) = {cps_floor:.2f}; cameraprimus floor = max(75, {result.cameraprimus_baseline:.2f} - 5) = {cp_floor:.2f}_")
    lines.append("")
    lines.append("## Decision flow (spec §Phase 2)")
    lines.append("")
    if result.verdict == Verdict.SHIP:
        lines.append("All regression-checks PASS + lieder Strong → **Ship**: open PR, set up follow-ups.")
    elif result.verdict == Verdict.INVESTIGATE:
        lines.append("All regression-checks PASS + lieder Mixed → **Investigate**: residual error mode analysis.")
    elif result.verdict == Verdict.PIVOT:
        lines.append("All regression-checks PASS + lieder Flat → **Pivot**: Phase 0 / Audiveris alternative.")
    else:
        lines.append("One or more per-dataset floors FAIL → **Diagnose** before any pivot decision.")
        lines.append("Don't draw lieder conclusions over a broken baseline.")
    return "\n".join(lines) + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2 decision gate for Stage 3.")
    p.add_argument("--lieder-onset-f1", type=float, required=True, help="Mean onset_f1 from eval/results/lieder_<name>.csv")
    p.add_argument("--musicxml-validity", type=float, required=True, help="Aggregate musicxml_validity_rate from the same CSV")
    p.add_argument("--per-dataset-json", type=str, required=True, help='JSON: {"dataset": composite_quality, ...} for 7 datasets')
    p.add_argument("--cameraprimus-systems-baseline", type=float, required=True, help="Stage 2 v2 cameraprimus_systems quality from token_manifest_stage3.jsonl eval")
    p.add_argument("--cameraprimus-baseline", type=float, required=True, help="Stage 2 v2 cameraprimus quality from token_manifest_full.jsonl eval")
    p.add_argument("--lc6548281-onset-f1", type=float, default=None, help="Optional sanity-check value")
    p.add_argument("--name", type=str, default="stage3_v2", help="Report identifier")
    p.add_argument("--output", type=Path, default=None, help="Optional path to write the markdown report")
    args = p.parse_args()

    per_dataset = json.loads(args.per_dataset_json)
    result = evaluate(
        lieder_mean_onset_f1=args.lieder_onset_f1,
        musicxml_validity_rate=args.musicxml_validity,
        per_dataset=per_dataset,
        cameraprimus_systems_baseline=args.cameraprimus_systems_baseline,
        cameraprimus_baseline=args.cameraprimus_baseline,
        lc6548281_onset_f1=args.lc6548281_onset_f1,
    )
    report = render_report(result, name=args.name)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
    else:
        print(report)
    return 0 if result.verdict in (Verdict.SHIP, Verdict.INVESTIGATE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
