"""Score a predicted MusicXML against a ground-truth MusicXML.

Three metrics (see spec 2026-05-16-bethlehem-clean-transcription-design):
  measure_recall  - fraction of GT measures present in the prediction
                    (Defect 1 signal: Stage A missed systems)
  clef_accuracy   - per-clef-occurrence correctness (every clef declaration
                    counts; a single flipped system is visible, not masked by
                    majority); part 0 expects treble G2, part 1 expects bass F4
                    (Defect 2 signal)
  note_onset_f1   - per-part onset F1 by (measure, offset, midi); tracked
                    for regression visibility, not a gate

CPU-only, deterministic, no pipeline dependency.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import music21  # noqa: E402

EXPECTED_CLEF_BY_PART = {0: "G2", 1: "F4"}


def _clef_tokens(part) -> list[str]:
    return [c.sign + str(c.line)
            for c in part.flatten().getElementsByClass(music21.clef.Clef)
            if c.sign is not None and c.line is not None]


def _measure_count(part) -> int:
    return len(part.getElementsByClass(music21.stream.Measure))


def _onsets(part) -> set:
    out = set()
    for m in part.getElementsByClass(music21.stream.Measure):
        for n in m.flatten().notes:
            for p in (n.pitches if n.isChord else [n.pitch]):
                out.add((m.number, round(float(n.offset), 2), p.midi))
    return out


def score(pred_path: str, gt_path: str) -> dict:
    pred = music21.converter.parse(pred_path)
    gt = music21.converter.parse(gt_path)
    pred_parts = list(pred.parts)
    gt_parts = list(gt.parts)

    recalls = []
    for i, gp in enumerate(gt_parts):
        gt_m = _measure_count(gp)
        if gt_m == 0:
            continue
        pred_m = _measure_count(pred_parts[i]) if i < len(pred_parts) else 0
        recalls.append(min(1.0, pred_m / gt_m))
    measure_recall = min(recalls) if recalls else 0.0

    correct = 0
    total = 0
    for i, pp in enumerate(pred_parts):
        expected = EXPECTED_CLEF_BY_PART.get(i)
        if expected is None:
            continue
        toks = _clef_tokens(pp) or ["<none>"]
        for t in toks:
            total += 1
            if t == expected:
                correct += 1
    clef_accuracy = (correct / total) if total else 0.0

    tp = fp = fn = 0
    for i, gp in enumerate(gt_parts):
        g = _onsets(gp)
        p = _onsets(pred_parts[i]) if i < len(pred_parts) else set()
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    denom = (2 * tp + fp + fn)
    note_onset_f1 = (2 * tp / denom) if denom else 1.0

    return {
        "measure_recall": measure_recall,
        "clef_accuracy": clef_accuracy,
        "note_onset_f1": note_onset_f1,
        "detail": {
            "gt_parts": len(gt_parts),
            "pred_parts": len(pred_parts),
            "gt_measures": [_measure_count(p) for p in gt_parts],
            "pred_measures": [_measure_count(p) for p in pred_parts],
            "pred_clefs": [_clef_tokens(p) for p in pred_parts],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", type=Path)
    ap.add_argument("gt", type=Path)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    r = score(str(args.pred), str(args.gt))
    print(json.dumps(r, indent=2))
    if args.json_out:
        args.json_out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
