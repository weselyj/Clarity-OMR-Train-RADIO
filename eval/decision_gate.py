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

# Per-dataset Stage 3 quality floors (revised 2026-05-10 per product rule:
# the model handles systems by default and naturally-single-staff scores as
# 1-staff systems. Single-staff datasets gate IFF their source is naturally
# single-staff — primus and cameraprimus qualify; grandstaff (single-staff)
# does NOT, because that data is artificially split from a 2-staff source).
#
# Static floors (absolute thresholds — system-level surfaces):
PER_DATASET_FLOORS = {
    "synthetic_systems": 90.0,
    "grandstaff_systems": 95.0,
    "primus_systems": 80.0,
}
#
# Dynamic floors (regression tripwires — naturally-single-staff surfaces and
# the cameraprimus_systems baseline). Each uses max(75, baseline - 5) so a
# small slip is tolerated, a catastrophic loss fires DIAGNOSE.
DYNAMIC_FLOOR_BASE = 75.0
DYNAMIC_FLOOR_REGRESSION_TOLERANCE = 5.0


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
    primus_baseline: float
    lc6548281_onset_f1: Optional[float]


def _resolve_dynamic_floor(baseline_re_eval: float) -> float:
    """Regression-tripwire floor: floor = max(75, baseline - 5).

    Used for naturally-single-staff datasets (primus, cameraprimus) and the
    cameraprimus_systems variant. Tolerates small slip, fires DIAGNOSE on
    catastrophic loss."""
    return max(DYNAMIC_FLOOR_BASE, baseline_re_eval - DYNAMIC_FLOOR_REGRESSION_TOLERANCE)


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
    primus_baseline: float,
    lc6548281_onset_f1: Optional[float] = None,
) -> GateResult:
    """Aggregate the inputs into a verdict.

    Inputs:
        lieder_mean_onset_f1: from eval/results/lieder_<name>.csv
        musicxml_validity_rate: from same CSV (corroborating only)
        per_dataset: dict {dataset_name: composite quality_score, 0-100 scale};
            6 gated keys (synthetic_systems, grandstaff_systems, primus_systems,
            cameraprimus_systems, primus, cameraprimus). The single-staff
            `grandstaff` key may be present but is NOT gated — that data is
            artificially split from a 2-staff source and is not a valid
            product input per the per-system inference design.
        cameraprimus_systems_baseline: Stage 2 v2 quality on token_manifest_stage3.jsonl
        cameraprimus_baseline: Stage 2 v2 quality on token_manifest_full.jsonl, cameraprimus rows
        primus_baseline: Stage 2 v2 quality on token_manifest_full.jsonl, primus rows
        lc6548281_onset_f1: optional sanity-check value (None if not evaluated)
    """
    floors = dict(PER_DATASET_FLOORS)
    floors["cameraprimus_systems"] = _resolve_dynamic_floor(cameraprimus_systems_baseline)
    floors["cameraprimus"] = _resolve_dynamic_floor(cameraprimus_baseline)
    floors["primus"] = _resolve_dynamic_floor(primus_baseline)

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
        primus_baseline=primus_baseline,
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
    for ds in ["synthetic_systems", "grandstaff_systems", "primus_systems", "cameraprimus_systems", "primus", "cameraprimus"]:
        r = result.per_dataset_results.get(ds)
        if r is None:
            lines.append(f"| {ds} | — | — | MISSING |")
            continue
        status = "✅ PASS" if r.passed else "❌ FAIL"
        lines.append(f"| {ds} | {r.measured:.2f} | {r.floor:.2f} | {status} |")
    lines.append("")
    cps_floor = result.per_dataset_results["cameraprimus_systems"].floor
    cp_floor = result.per_dataset_results["cameraprimus"].floor
    p_floor = result.per_dataset_results["primus"].floor
    lines.append(
        f"_Dynamic floors (max(75, baseline-5)): "
        f"cameraprimus_systems = max(75, {result.cameraprimus_systems_baseline:.2f} - 5) = {cps_floor:.2f}; "
        f"cameraprimus = max(75, {result.cameraprimus_baseline:.2f} - 5) = {cp_floor:.2f}; "
        f"primus = max(75, {result.primus_baseline:.2f} - 5) = {p_floor:.2f}_"
    )
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
    p.add_argument("--primus-baseline", type=float, required=True, help="Stage 2 v2 primus quality from token_manifest_full.jsonl eval")
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
        primus_baseline=args.primus_baseline,
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
