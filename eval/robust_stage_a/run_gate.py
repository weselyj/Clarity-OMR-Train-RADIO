"""Run the strict Stage-A gate for a given YOLO checkpoint.

Inference (GPU) + the existing lieder scorer are orchestrated here; the
verdict logic is the pure eval.robust_stage_a.gate module. Intended to run
on seder (GPU). The pure gate is unit-tested separately on CPU.

Example (on seder, repo root):
  venv-cu132/Scripts/python.exe -u -m eval.robust_stage_a.run_gate \
    --manifest eval/robust_stage_a/heldout/manifest.json \
    --yolo-weights runs/detect/runs/yolo26m_systems_faintink/weights/best.pt \
    --lieder-csv eval/results/stagea_faintink.csv \
    --lieder-baseline-csv eval/results/stagea_baseline_pre_faintink.csv \
    --lyric-baseline-out eval/robust_stage_a/lyric_baseline.json \
    --report-out eval/robust_stage_a/gate_report.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from eval.robust_stage_a.gate import (  # noqa: E402
    Pred,
    combined_gate,
    lyric_system_recall,
    recall_from_stagea_csv,
    score_scenario,
    verdict_to_report,
)
from eval.robust_stage_a.manifest import load_manifest  # noqa: E402


def _infer(image_path: str, model, conf: float) -> list[Pred]:
    """Run the Stage-A YOLO detector on one image with a preloaded model.
    Uses Ultralytics directly (same engine scripts/audit/dump_system_crops.py
    uses)."""
    res = model.predict(image_path, conf=conf, verbose=False)[0]
    out: list[Pred] = []
    for b in res.boxes:
        x1, y1, x2, y2 = (float(v) for v in b.xyxy[0].tolist())
        out.append(Pred(box=(x1, y1, x2, y2), conf=float(b.conf[0])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--yolo-weights", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--image-root", type=Path, default=None,
                    help="prefix for relative manifest image paths")
    ap.add_argument("--lieder-csv", type=Path, required=True,
                    help="score_stage_a_only.py CSV for the NEW checkpoint")
    ap.add_argument("--lieder-baseline-csv", type=Path,
                    default=Path("eval/results/stagea_baseline_pre_faintink.csv"))
    ap.add_argument("--lyric-baseline", type=Path, default=None,
                    help="JSON with prior lyric_system_recall to compare against")
    ap.add_argument("--lyric-baseline-out", type=Path, default=None,
                    help="write the computed lyric_system_recall here (snapshot)")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    scenarios = load_manifest(args.manifest)
    from ultralytics import YOLO  # deferred: keeps the module CPU-importable

    model = YOLO(str(args.yolo_weights))
    preds_by_scenario: dict[str, list[Pred]] = {}
    results = []
    for sc in scenarios:
        img = sc.image
        if args.image_root is not None and not Path(img).is_absolute():
            img = str(args.image_root / img)
        preds = _infer(img, model, args.conf)
        preds_by_scenario[sc.scenario_id] = preds
        results.append(score_scenario(sc, preds))

    lyric_recall = lyric_system_recall(scenarios, preds_by_scenario)
    if args.lyric_baseline_out is not None:
        args.lyric_baseline_out.write_text(
            json.dumps({"lyric_system_recall": lyric_recall}), encoding="utf-8")
    if args.lyric_baseline is not None:
        lyric_baseline = json.loads(
            args.lyric_baseline.read_text(encoding="utf-8"))["lyric_system_recall"]
    else:
        lyric_baseline = lyric_recall  # first run: self-baseline (no regression)

    lieder_recall = recall_from_stagea_csv(args.lieder_csv)
    lieder_baseline = recall_from_stagea_csv(args.lieder_baseline_csv)

    verdict = combined_gate(results, lyric_recall, lyric_baseline,
                            lieder_recall, lieder_baseline)
    report = verdict_to_report(verdict, results)
    print(report)
    if args.report_out is not None:
        args.report_out.write_text(report + "\n", encoding="utf-8")
    return 0 if verdict.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
