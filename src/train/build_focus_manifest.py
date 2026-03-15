#!/usr/bin/env python3
"""Build a focused boost manifest for controlled Stage-B retraining."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


RHYTHM_TOKENS = {
    "_eighth",
    "_sixteenth",
    "_thirty_second",
    "_sixty_fourth",
    "_dot",
    "_double_dot",
    "<tuplet_3>",
    "<tuplet_5>",
    "<tuplet_6>",
    "<tuplet_7>",
}


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


def _write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _coerce_tokens(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str):
        return [token for token in value.replace("\t", " ").split(" ") if token]
    return []


def _parse_csv_tokens(raw: str) -> List[str]:
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def _is_distorted_row(row: Dict[str, object]) -> bool:
    variant = str(row.get("variant", "")).strip().lower()
    if variant == "distorted":
        return True
    sample_id = str(row.get("sample_id", ""))
    if "_distorted" in sample_id:
        return True
    image_path = str(row.get("image_path", ""))
    if "_distorted" in image_path:
        return True
    dataset = str(row.get("dataset", "")).strip().lower()
    # CameraPrimus is the distorted Primus capture set.
    if dataset == "cameraprimus":
        return True
    return False


def _score_row(
    tokens: Sequence[str],
    *,
    target_notes: set[str],
    target_tokens: set[str],
    target_prefixes: Sequence[str],
    include_chords: bool,
    include_rhythm: bool,
) -> Tuple[int, Dict[str, int]]:
    note_hits = 0
    token_hits = 0
    prefix_hits = 0
    chord_hits = 0
    rhythm_hits = 0

    for token in tokens:
        if token in target_notes:
            note_hits += 1
        if token in target_tokens:
            token_hits += 1
        if target_prefixes and any(token.startswith(prefix) for prefix in target_prefixes):
            prefix_hits += 1
        if include_chords and token in {"<chord_start>", "<chord_end>"}:
            chord_hits += 1
        if include_rhythm and token in RHYTHM_TOKENS:
            rhythm_hits += 1

    score = (3 * note_hits) + (2 * token_hits) + (2 * prefix_hits) + (1 * chord_hits) + (1 * rhythm_hits)
    return score, {
        "note_hits": note_hits,
        "token_hits": token_hits,
        "prefix_hits": prefix_hits,
        "chord_hits": chord_hits,
        "rhythm_hits": rhythm_hits,
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build a focused boost token manifest for controlled retraining.")
    parser.add_argument(
        "--input-manifests",
        type=str,
        default=",".join(
            [
                str(project_root / "src" / "data" / "manifests" / "token_manifest.jsonl"),
                str(project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"),
            ]
        ),
        help="Comma-separated token manifest JSONL paths.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Output boost manifest JSONL path.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Manifest split to select (default: train).",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated dataset allowlist (e.g. grandstaff,synthetic_fullpage,synthetic_polyphonic).",
    )
    parser.add_argument(
        "--exclude-distorted",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exclude distorted variants (variant=distorted, *_distorted sample/image, and cameraprimus).",
    )
    parser.add_argument(
        "--target-notes",
        type=str,
        default="",
        help="Comma-separated note tokens to upweight (e.g. note-C2,note-G4,note-B2).",
    )
    parser.add_argument(
        "--target-tokens",
        type=str,
        default="",
        help="Optional comma-separated exact tokens to upweight (e.g. clef-G2,keySignature-CM,timeSignature-4/4).",
    )
    parser.add_argument(
        "--target-prefixes",
        type=str,
        default="",
        help="Optional comma-separated token prefixes to upweight (e.g. clef-,keySignature-,timeSignature-).",
    )
    parser.add_argument(
        "--include-chords",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include chord-structure tokens in scoring.",
    )
    parser.add_argument(
        "--include-rhythm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include rhythm tokens in scoring.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=3,
        help="Minimum score for a row to be selected.",
    )
    parser.add_argument(
        "--max-selected",
        type=int,
        default=60000,
        help="Cap on selected source rows before duplication.",
    )
    parser.add_argument(
        "--max-repeat",
        type=int,
        default=4,
        help="Maximum duplicate repeat per selected row.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON summary output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_target = str(args.split).strip().lower()
    allowed_datasets = {item.lower() for item in _parse_csv_tokens(args.datasets)}
    target_notes = set(_parse_csv_tokens(args.target_notes))
    target_tokens = set(_parse_csv_tokens(args.target_tokens))
    target_prefixes = tuple(_parse_csv_tokens(args.target_prefixes))
    if not target_notes and not target_tokens and not target_prefixes and not bool(args.include_chords) and not bool(args.include_rhythm):
        raise ValueError("Provide at least one target: --target-notes, --target-tokens, --target-prefixes, --include-chords, or --include-rhythm.")

    manifest_paths: List[Path] = []
    for raw_path in _parse_csv_tokens(args.input_manifests):
        path = Path(raw_path)
        if not path.is_absolute():
            path = (Path.cwd() / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input manifest not found: {path}")
        manifest_paths.append(path)

    all_rows: List[Dict[str, object]] = []
    for path in manifest_paths:
        all_rows.extend(_read_jsonl(path))

    scored: List[Tuple[int, Dict[str, int], Dict[str, object]]] = []
    skipped_wrong_split = 0
    skipped_dataset = 0
    skipped_distorted = 0
    skipped_empty_tokens = 0
    skipped_low_score = 0
    for row in all_rows:
        if str(row.get("split", "train")).strip().lower() != split_target:
            skipped_wrong_split += 1
            continue
        dataset = str(row.get("dataset", "")).strip().lower()
        if allowed_datasets and dataset not in allowed_datasets:
            skipped_dataset += 1
            continue
        if bool(args.exclude_distorted) and _is_distorted_row(row):
            skipped_distorted += 1
            continue
        tokens = _coerce_tokens(row.get("token_sequence"))
        if not tokens:
            skipped_empty_tokens += 1
            continue
        score, detail = _score_row(
            tokens,
            target_notes=target_notes,
            target_tokens=target_tokens,
            target_prefixes=target_prefixes,
            include_chords=bool(args.include_chords),
            include_rhythm=bool(args.include_rhythm),
        )
        if score < int(args.min_score):
            skipped_low_score += 1
            continue
        scored.append((score, detail, row))

    scored.sort(key=lambda item: item[0], reverse=True)
    if int(args.max_selected) > 0:
        scored = scored[: int(args.max_selected)]

    boosted_rows: List[Dict[str, object]] = []
    total_repeats = 0
    for index, (score, detail, row) in enumerate(scored):
        base_id = str(row.get("sample_id", f"sample_{index}"))
        repeats = min(int(args.max_repeat), 1 + min(3, max(0, score // max(1, int(args.min_score)))))
        repeats = max(1, repeats)
        total_repeats += repeats
        for rep in range(repeats):
            copy_row = dict(row)
            copy_row["sample_id"] = f"{base_id}__focus{rep+1}"
            copy_row["focus_boost"] = {
                "score": score,
                "note_hits": int(detail["note_hits"]),
                "token_hits": int(detail["token_hits"]),
                "prefix_hits": int(detail["prefix_hits"]),
                "chord_hits": int(detail["chord_hits"]),
                "rhythm_hits": int(detail["rhythm_hits"]),
            }
            boosted_rows.append(copy_row)

    output_path = args.output_manifest.resolve()
    _write_jsonl(output_path, boosted_rows)

    report = {
        "input_manifests": [str(path) for path in manifest_paths],
        "split": split_target,
        "dataset_allowlist": sorted(allowed_datasets),
        "exclude_distorted": bool(args.exclude_distorted),
        "target_notes": sorted(target_notes),
        "target_tokens": sorted(target_tokens),
        "target_prefixes": list(target_prefixes),
        "rows_seen": len(all_rows),
        "rows_skipped": {
            "wrong_split": skipped_wrong_split,
            "dataset_filtered": skipped_dataset,
            "distorted_filtered": skipped_distorted,
            "empty_tokens": skipped_empty_tokens,
            "low_score": skipped_low_score,
        },
        "selected_rows": len(scored),
        "boosted_rows": len(boosted_rows),
        "avg_repeat": (float(total_repeats) / float(max(1, len(scored)))),
        "output_manifest": str(output_path),
    }
    print(json.dumps(report, indent=2))

    if args.report_json:
        report_path = args.report_json.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
