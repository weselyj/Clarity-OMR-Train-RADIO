#!/usr/bin/env python3
"""Evaluation runner for OMR prediction manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import (
    aggregate_metrics,
    default_ablation_matrix,
    musicxml_musical_similarity,
    musicxml_validity,
    musicxml_validity_from_tokens,
)


def _coerce_tokens(value) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [token for token in value.replace("\t", " ").split(" ") if token]
    raise ValueError(f"Unsupported token container: {type(value)}")


def load_rows(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if "pred_tokens" not in row or "gt_tokens" not in row:
                raise ValueError(f"Row at {path}:{line_no} missing pred_tokens/gt_tokens.")
            row["pred_tokens"] = _coerce_tokens(row["pred_tokens"])
            row["gt_tokens"] = _coerce_tokens(row["gt_tokens"])
            rows.append(row)
    return rows


def _group_by_dataset(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for row in rows:
        dataset = str(row.get("dataset", "unknown")).lower()
        grouped.setdefault(dataset, []).append(row)
    return grouped


def evaluate_rows(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    overall_pairs = [(row["pred_tokens"], row["gt_tokens"]) for row in rows]
    overall = aggregate_metrics(overall_pairs)

    by_dataset = {}
    for dataset, dataset_rows in _group_by_dataset(rows).items():
        pairs = [(row["pred_tokens"], row["gt_tokens"]) for row in dataset_rows]
        by_dataset[dataset] = aggregate_metrics(pairs)

    musicxml_paths = [str(row["pred_musicxml_path"]) for row in rows if row.get("pred_musicxml_path")]
    if musicxml_paths:
        # Some rows already carry rendered MusicXML files — validate them via
        # music21 (heavy but most accurate; matches Stage-B production output).
        musicxml_rate = musicxml_validity(musicxml_paths)
    else:
        # Stage 3 / Plan C path: rows only carry pred_tokens. Decode tokens to
        # MusicXML in-memory and validate via etree. Spec §3 (line 264) — the
        # eval driver was leaving this metric at None because no path was
        # available; this branch enables it always-on so Phase 2 / Plan D can
        # gate on the value without further changes.
        token_lists = [row["pred_tokens"] for row in rows if row.get("pred_tokens")]
        musicxml_rate = musicxml_validity_from_tokens(token_lists)
    roundtrip_pairs: List[Tuple[str, str]] = []
    for row in rows:
        pred_musicxml = row.get("pred_musicxml_path")
        if not pred_musicxml:
            continue
        reference_musicxml = None
        for key in ("reference_musicxml_path", "gt_musicxml_path", "ref_musicxml_path"):
            if row.get(key):
                reference_musicxml = str(row[key])
                break
        if reference_musicxml:
            roundtrip_pairs.append((str(pred_musicxml), reference_musicxml))
    roundtrip = musicxml_musical_similarity(roundtrip_pairs) if roundtrip_pairs else {
        "musical_samples": 0,
        "musical_precision": None,
        "musical_recall": None,
        "musical_f1": None,
        "musical_overlap": None,
        "musical_onset_precision": None,
        "musical_onset_recall": None,
        "musical_onset_f1": None,
    }

    return {
        "sample_count": len(rows),
        "overall": overall,
        "overall_quality": overall.get("quality", {}),
        "by_dataset": by_dataset,
        "musicxml_validity_rate": musicxml_rate,
        **roundtrip,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Evaluate OMR token predictions against ground truth.")
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="JSONL with rows containing pred_tokens and gt_tokens.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=project_root / "src" / "eval" / "evaluation_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--output-ablation-template",
        type=Path,
        default=project_root / "src" / "eval" / "ablation_template.json",
        help="Output ablation template JSON path.",
    )
    parser.add_argument(
        "--skip-ablation-template",
        action="store_true",
        help="Do not write ablation template file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(args.predictions)
    summary = evaluate_rows(rows)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.skip_ablation_template:
        args.output_ablation_template.parent.mkdir(parents=True, exist_ok=True)
        args.output_ablation_template.write_text(
            json.dumps(default_ablation_matrix(), indent=2),
            encoding="utf-8",
        )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
