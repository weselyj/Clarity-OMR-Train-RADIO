#!/usr/bin/env python3
"""Summarize Stage-B failure patterns from raw prediction JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.eval.metrics import BASE_DURATION_BEATS, DURATION_MODIFIERS, STRUCTURAL_TOKENS, VOICE_TOKENS, evaluate_pair


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return rows


def _coerce_tokens(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str):
        return [token for token in value.replace("\t", " ").split(" ") if token]
    return []


def _token_family(token: str) -> str:
    if token.startswith("note-"):
        return "note_pitch"
    if token.startswith("gracenote-"):
        return "gracenote_pitch"
    if token == "rest":
        return "rest"
    if token in BASE_DURATION_BEATS:
        return "duration"
    if token in DURATION_MODIFIERS or token.startswith("<tuplet_"):
        return "rhythm_modifier"
    if token.startswith("tempo-"):
        return "tempo"
    if token in {"tie_start", "tie_end"}:
        return "tie"
    if token in {"slur_start", "slur_end"}:
        return "slur"
    if token in {"<chord_start>", "<chord_end>"}:
        return "chord_structure"
    if token.startswith("timeSignature-"):
        return "time_signature"
    if token.startswith("keySignature-"):
        return "key_signature"
    if token.startswith("clef-"):
        return "clef"
    if token.startswith("dynamic-"):
        return "dynamic"
    if token.startswith("expr-"):
        return "expression"
    if token in VOICE_TOKENS:
        return "voice"
    if token in STRUCTURAL_TOKENS:
        return "structure"
    if token in {"<bos>", "<eos>", "<staff_start>", "<staff_end>"}:
        return "framing"
    return "other"


def _first_token_with_prefix(tokens: Sequence[str], prefix: str) -> str | None:
    for token in tokens:
        if token.startswith(prefix):
            return token
    return None


@dataclass(frozen=True)
class HardSample:
    sample_id: str
    dataset: str
    note_event_f1: float
    pitch_accuracy: float
    rhythm_accuracy: float
    ser: float
    key_time_accuracy: float


def _rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _counter_to_top(counter: Counter[str], limit: int) -> List[Dict[str, object]]:
    return [{"token": token, "count": count} for token, count in counter.most_common(max(1, limit))]


def _pair_counter_to_top(counter: Counter[Tuple[str, str]], limit: int) -> List[Dict[str, object]]:
    top: List[Dict[str, object]] = []
    for (gt_token, pred_token), count in counter.most_common(max(1, limit)):
        top.append({"gt": gt_token, "pred": pred_token, "count": count})
    return top


def _build_training_focus(summary: Dict[str, object]) -> List[str]:
    focus: List[str] = []
    family_rows = summary.get("family_error_rates", [])
    family_map = {str(row["family"]): row for row in family_rows if isinstance(row, dict)}

    note_row = family_map.get("note_pitch", {})
    note_missing_rate = float(note_row.get("missing_rate", 0.0))
    if note_missing_rate >= 0.08:
        focus.append("Boost recall on note heads: oversample dense-note crops and hard negatives where notes disappear.")

    duration_row = family_map.get("duration", {})
    if float(duration_row.get("missing_rate", 0.0)) >= 0.08:
        focus.append("Target rhythm duration tokens: run a duration-heavy fine-tune subset (_eighth/_sixteenth/_dot cases).")

    chord_row = family_map.get("chord_structure", {})
    if float(chord_row.get("missing_rate", 0.0)) >= 0.05:
        focus.append("Increase chord-structure supervision: emphasize <chord_start>/<chord_end> consistency examples.")

    tie_row = family_map.get("tie", {})
    if float(tie_row.get("missing_rate", 0.0)) >= 0.05:
        focus.append("Add tie-focused curriculum (tie_start/tie_end) with phrase-boundary and cross-measure examples.")

    tempo_summary = summary.get("tempo", {})
    if isinstance(tempo_summary, dict) and float(tempo_summary.get("mismatch_rate", 0.0)) >= 0.10:
        focus.append("Include tempo-token controlled training (tempo-* with expr-a_tempo/expr-rit contexts).")

    if not focus:
        focus.append("No single dominant failure family; use hard-sample replay from the worst note_event_f1 samples.")
    return focus


def summarize_failures(rows: Sequence[Dict[str, object]], *, top_k: int) -> Dict[str, object]:
    gt_family_total: Counter[str] = Counter()
    missing_family: Counter[str] = Counter()
    extra_family: Counter[str] = Counter()
    missing_tokens: Counter[str] = Counter()
    extra_tokens: Counter[str] = Counter()
    note_missing: Counter[str] = Counter()
    note_substitutions: Counter[Tuple[str, str]] = Counter()
    tempo_pair_mismatches: Counter[Tuple[str, str]] = Counter()
    hard_samples: List[HardSample] = []
    total_replace_spans = 0
    total_insert_tokens = 0
    total_delete_tokens = 0

    tempo_gt_present = 0
    tempo_pred_present = 0
    tempo_mismatch = 0

    for row in rows:
        gt_tokens = _coerce_tokens(row.get("gt_tokens"))
        pred_tokens = _coerce_tokens(row.get("tokens", row.get("pred_tokens")))
        if not gt_tokens:
            continue

        for token in gt_tokens:
            gt_family_total[_token_family(token)] += 1

        matcher = SequenceMatcher(a=gt_tokens, b=pred_tokens, autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "equal":
                continue
            if tag in {"delete", "replace"}:
                deleted = gt_tokens[i1:i2]
                total_delete_tokens += len(deleted)
                for token in deleted:
                    missing_tokens[token] += 1
                    missing_family[_token_family(token)] += 1
                    if token.startswith("note-"):
                        note_missing[token] += 1
            if tag in {"insert", "replace"}:
                inserted = pred_tokens[j1:j2]
                total_insert_tokens += len(inserted)
                for token in inserted:
                    extra_tokens[token] += 1
                    extra_family[_token_family(token)] += 1
            if tag == "replace":
                total_replace_spans += 1
                left = gt_tokens[i1:i2]
                right = pred_tokens[j1:j2]
                for gt_token, pred_token in zip(left, right):
                    if gt_token.startswith("note-") and pred_token.startswith("note-") and gt_token != pred_token:
                        note_substitutions[(gt_token, pred_token)] += 1

        gt_tempo = _first_token_with_prefix(gt_tokens, "tempo-")
        pred_tempo = _first_token_with_prefix(pred_tokens, "tempo-")
        if gt_tempo is not None:
            tempo_gt_present += 1
        if pred_tempo is not None:
            tempo_pred_present += 1
        if gt_tempo is not None and pred_tempo is not None and gt_tempo != pred_tempo:
            tempo_mismatch += 1
            tempo_pair_mismatches[(gt_tempo, pred_tempo)] += 1

        metrics = evaluate_pair(pred_tokens=pred_tokens, gt_tokens=gt_tokens)
        hard_samples.append(
            HardSample(
                sample_id=str(row.get("sample_id", "")),
                dataset=str(row.get("dataset", "unknown")),
                note_event_f1=float(metrics.note_event_f1),
                pitch_accuracy=float(metrics.pitch_accuracy),
                rhythm_accuracy=float(metrics.rhythm_accuracy),
                ser=float(metrics.ser),
                key_time_accuracy=float(metrics.key_time_accuracy),
            )
        )

    families = sorted(set(gt_family_total) | set(missing_family) | set(extra_family))
    family_rows: List[Dict[str, object]] = []
    for family in families:
        gt_count = int(gt_family_total.get(family, 0))
        miss_count = int(missing_family.get(family, 0))
        extra_count = int(extra_family.get(family, 0))
        family_rows.append(
            {
                "family": family,
                "gt_count": gt_count,
                "missing_count": miss_count,
                "extra_count": extra_count,
                "missing_rate": _rate(miss_count, gt_count),
                "extra_rate": _rate(extra_count, gt_count),
            }
        )
    family_rows.sort(key=lambda row: float(row["missing_rate"]), reverse=True)

    hard_samples.sort(key=lambda item: (item.note_event_f1, item.pitch_accuracy, -item.ser))
    hardest = [asdict(item) for item in hard_samples[: max(1, top_k)]]

    summary: Dict[str, object] = {
        "sample_count": len(hard_samples),
        "edit_ops": {
            "delete_tokens": total_delete_tokens,
            "insert_tokens": total_insert_tokens,
            "replace_spans": total_replace_spans,
        },
        "family_error_rates": family_rows,
        "top_missing_tokens": _counter_to_top(missing_tokens, top_k),
        "top_extra_tokens": _counter_to_top(extra_tokens, top_k),
        "top_missing_notes": _counter_to_top(note_missing, top_k),
        "top_note_substitutions": _pair_counter_to_top(note_substitutions, top_k),
        "tempo": {
            "gt_with_tempo_count": tempo_gt_present,
            "pred_with_tempo_count": tempo_pred_present,
            "tempo_mismatch_count": tempo_mismatch,
            "mismatch_rate": _rate(tempo_mismatch, tempo_gt_present),
            "top_mismatches": _pair_counter_to_top(tempo_pair_mismatches, top_k),
        },
        "hardest_samples": hardest,
    }
    summary["training_focus"] = _build_training_focus(summary)
    return summary


def _print_console_summary(summary: Dict[str, object], *, top_k: int) -> None:
    print(f"samples={summary.get('sample_count', 0)}")
    edit_ops = summary.get("edit_ops", {})
    if isinstance(edit_ops, dict):
        print(
            "ops:"
            f" delete={int(edit_ops.get('delete_tokens', 0))}"
            f" insert={int(edit_ops.get('insert_tokens', 0))}"
            f" replace_spans={int(edit_ops.get('replace_spans', 0))}"
        )

    family_rows = summary.get("family_error_rates", [])
    if isinstance(family_rows, list):
        print("\nTop families by missing_rate:")
        for row in family_rows[: max(1, top_k)]:
            if not isinstance(row, dict):
                continue
            print(
                f"  {row.get('family')}:"
                f" missing_rate={float(row.get('missing_rate', 0.0)):.3f}"
                f" (missing={int(row.get('missing_count', 0))} / gt={int(row.get('gt_count', 0))})"
            )

    top_notes = summary.get("top_missing_notes", [])
    if isinstance(top_notes, list) and top_notes:
        print("\nTop missing note pitches:")
        for item in top_notes[: max(1, top_k)]:
            if isinstance(item, dict):
                print(f"  {item.get('token')}: {int(item.get('count', 0))}")

    tempo_summary = summary.get("tempo", {})
    if isinstance(tempo_summary, dict):
        print(
            "\nTempo:"
            f" mismatch_rate={float(tempo_summary.get('mismatch_rate', 0.0)):.3f}"
            f" ({int(tempo_summary.get('tempo_mismatch_count', 0))}"
            f"/{int(tempo_summary.get('gt_with_tempo_count', 0))})"
        )

    focus = summary.get("training_focus", [])
    if isinstance(focus, list):
        print("\nTraining focus:")
        for item in focus:
            print(f"  - {item}")


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Summarize Stage-B error patterns from raw prediction JSONL.")
    parser.add_argument(
        "--predictions-jsonl",
        type=Path,
        default=project_root / "src" / "eval" / "checkpoint_eval" / "stageb_eval_predictions_raw.jsonl",
        help="Path to stageb_eval_predictions_raw.jsonl.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=15,
        help="Number of top tokens/samples to report.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON file to write full summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.predictions_jsonl.resolve()
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions JSONL not found: {predictions_path}")
    rows = _read_jsonl(predictions_path)
    summary = summarize_failures(rows, top_k=max(3, int(args.top_k)))
    _print_console_summary(summary, top_k=max(3, int(args.top_k)))

    if args.output_json:
        output_path = args.output_json.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
