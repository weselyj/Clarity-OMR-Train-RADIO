#!/usr/bin/env python3
"""Compare candidate MusicXML files against a ground-truth MusicXML.

Usage:
    python src/eval/compare_musicxml.py ground_truth.musicxml candidate1.musicxml candidate2.musicxml ...

Outputs a side-by-side comparison table showing which candidate is closest
to the ground truth using mir_eval note-level transcription metrics.

Comparison approach:
  - Notes extracted as (onset, offset, pitch_hz) via music21.
  - Tied notes merged (stripTies) so C4-half == C4-quarter + C4-quarter-tied.
  - mir_eval.transcription computes precision/recall/F1 with onset tolerance
    and pitch tolerance (50 cents = enharmonic equivalence).
  - Two modes: onset+offset (full note accuracy) and onset-only (ignores
    duration errors).
  - Structural checks (parts, measures, key/time signatures) reported
    alongside the musical similarity scores.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _import_music21():
    import music21
    return music21


@dataclass
class ScoreInfo:
    """Extracted musical content from a MusicXML file."""
    path: str
    score: object = None  # music21 Score object (kept for mir_eval extraction)
    num_parts: int = 0
    num_measures: int = 0
    total_notes: int = 0
    total_rests: int = 0
    key_signatures: List[str] = field(default_factory=list)
    time_signatures: List[str] = field(default_factory=list)
    error: Optional[str] = None


def extract_score_info(path: Path) -> ScoreInfo:
    """Parse a MusicXML file and extract structural metadata.

    The music21 Score object is retained for mir_eval note extraction.
    Tied notes are merged via stripTies() so that the same sounding note
    is always represented as a single event regardless of notation.
    """
    m21 = _import_music21()
    info = ScoreInfo(path=str(path))

    try:
        score = m21.converter.parse(str(path))
    except Exception as exc:
        info.error = str(exc)
        return info

    try:
        score = score.stripTies()
    except Exception:
        pass

    info.score = score
    parts = score.parts
    info.num_parts = len(parts)

    for part_idx, part in enumerate(parts):
        measures = part.getElementsByClass(m21.stream.Measure)
        if part_idx == 0:
            info.num_measures = len(measures)

        for measure in measures:
            for ks in measure.getElementsByClass(m21.key.KeySignature):
                try:
                    name = ks.asKey().name if hasattr(ks, "asKey") else str(ks)
                except Exception:
                    name = str(ks)
                if name not in info.key_signatures:
                    info.key_signatures.append(name)
            for ts in measure.getElementsByClass(m21.meter.TimeSignature):
                ts_str = ts.ratioString
                if ts_str not in info.time_signatures:
                    info.time_signatures.append(ts_str)

        for elem in part.recurse().notesAndRests:
            if elem.isRest:
                info.total_rests += 1
            elif hasattr(elem, "pitches") and elem.pitches:
                info.total_notes += len(elem.pitches)

    return info


# ---------------------------------------------------------------------------
# mir_eval note extraction
# ---------------------------------------------------------------------------

def _get_bpm(score) -> float:
    """Return the first MetronomeMark BPM from a score, or 120.0 as default."""
    m21 = _import_music21()
    for elem in score.recurse():
        if isinstance(elem, m21.tempo.MetronomeMark):
            bpm = elem.getQuarterBPM()
            if bpm and bpm > 0:
                return float(bpm)
    return 120.0


def _extract_note_events(score, bpm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (intervals, pitches_hz) from a music21 score for mir_eval.

    *intervals* is Nx2 (onset, offset) in **seconds**, converted from
    quarter-note offsets using the given *bpm*.
    *pitches_hz* is N-array of frequencies in Hz.
    Tied notes should already be merged via stripTies().
    """
    m21 = _import_music21()
    qn_to_sec = 60.0 / bpm  # seconds per quarter-note

    onsets: List[float] = []
    offsets: List[float] = []
    pitches: List[float] = []

    for part in score.parts:
        for elem in part.recurse().notesAndRests:
            if elem.isRest:
                continue
            abs_offset = float(elem.getOffsetInHierarchy(score)) * qn_to_sec
            dur = float(elem.quarterLength) * qn_to_sec
            if dur <= 0:
                dur = 0.001  # grace notes
            if isinstance(elem, m21.note.Note):
                onsets.append(abs_offset)
                offsets.append(abs_offset + dur)
                pitches.append(float(elem.pitch.frequency))
            elif isinstance(elem, m21.chord.Chord):
                for n in elem.notes:
                    onsets.append(abs_offset)
                    offsets.append(abs_offset + dur)
                    pitches.append(float(n.pitch.frequency))

    if not onsets:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    return (
        np.column_stack([onsets, offsets]).astype(np.float64),
        np.array(pitches, dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

@dataclass
class ComparisonResult:
    """Result of comparing a candidate against ground truth."""
    candidate_path: str
    # Structure
    parts_match: bool = False
    measures_match: bool = False
    measures_diff: int = 0
    key_sig_match: bool = False
    time_sig_match: bool = False
    # Note counts
    note_count_diff: int = 0
    rest_count_diff: int = 0
    # mir_eval onset+offset metrics
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    overlap: float = 0.0
    # mir_eval onset-only metrics
    onset_precision: float = 0.0
    onset_recall: float = 0.0
    onset_f1: float = 0.0
    # Composite
    quality_score: float = 0.0
    error: Optional[str] = None


def compare(gt: ScoreInfo, candidate: ScoreInfo) -> ComparisonResult:
    """Compare a candidate ScoreInfo against ground truth using mir_eval."""
    import mir_eval

    result = ComparisonResult(candidate_path=candidate.path)

    if candidate.error:
        result.error = candidate.error
        return result

    # Structure
    result.parts_match = gt.num_parts == candidate.num_parts
    result.measures_match = gt.num_measures == candidate.num_measures
    result.measures_diff = candidate.num_measures - gt.num_measures
    result.key_sig_match = set(gt.key_signatures) == set(candidate.key_signatures)
    result.time_sig_match = set(gt.time_signatures) == set(candidate.time_signatures)

    # Counts
    result.note_count_diff = candidate.total_notes - gt.total_notes
    result.rest_count_diff = candidate.total_rests - gt.total_rests

    # Use GT's BPM to convert both scores to seconds
    gt_bpm = _get_bpm(gt.score)
    ref_intervals, ref_pitches = _extract_note_events(gt.score, bpm=gt_bpm)
    est_intervals, est_pitches = _extract_note_events(candidate.score, bpm=gt_bpm)

    if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
        return result

    # Onset + offset matching (full note accuracy)
    result.precision, result.recall, result.f1, result.overlap = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,  # 50 ms (mir_eval standard)
            pitch_tolerance=50.0,  # cents (enharmonic equivalence)
        )
    )

    # Onset-only matching (ignores duration differences)
    result.onset_precision, result.onset_recall, result.onset_f1, _ = (
        mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches,
            est_intervals, est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=None,
        )
    )

    # Composite quality score (0-100)
    #
    # Weights:
    #   onset_f1  (30): right pitch at right time (ignoring duration)
    #   f1        (30): full note accuracy (onset + offset + pitch)
    #   overlap   (15): temporal overlap quality of matched notes
    #   key/time  (10): structural correctness
    #   measures   (5): measure count match
    #   parts     (10): part count match
    result.quality_score = (
        30.0 * result.onset_f1
        + 30.0 * result.f1
        + 15.0 * result.overlap
        + 5.0 * (1.0 if result.key_sig_match else 0.0)
        + 5.0 * (1.0 if result.time_sig_match else 0.0)
        + 5.0 * (1.0 if result.measures_match else 0.0)
        + 10.0 * (1.0 if result.parts_match else 0.0)
    )

    return result


def print_comparison_table(gt: ScoreInfo, results: List[ComparisonResult]) -> None:
    """Print a formatted comparison table."""
    names = [Path(r.candidate_path).stem for r in results]
    max_name = max(len(n) for n in names) if names else 10
    col_w = max(max_name + 2, 18)

    def row(label: str, values: List[str]) -> str:
        cols = "".join(v.rjust(col_w) for v in values)
        return f"  {label:<24}{cols}"

    header = "".join(n.rjust(col_w) for n in names)
    sep = "-" * (26 + col_w * len(names))

    gt_bpm = _get_bpm(gt.score)
    print(f"\n  Ground truth: {gt.path}")
    print(f"  Parts: {gt.num_parts} | Measures: {gt.num_measures} | "
          f"Notes: {gt.total_notes} | Rests: {gt.total_rests}")
    print(f"  Key: {', '.join(gt.key_signatures) or '--'} | "
          f"Time: {', '.join(gt.time_signatures) or '--'} | "
          f"BPM: {gt_bpm:.0f} (used for all candidates)")
    print()
    print(f"  {'METRIC':<24}{header}")
    print(f"  {sep}")

    # Structure
    print(row("Parts match", ["YES" if r.parts_match else "NO" for r in results]))
    print(row("Measures match", [
        "YES" if r.measures_match else f"NO ({r.measures_diff:+d})" for r in results
    ]))
    print(row("Key sig match", ["YES" if r.key_sig_match else "NO" for r in results]))
    print(row("Time sig match", ["YES" if r.time_sig_match else "NO" for r in results]))
    print()

    # Counts
    print(row("Note count diff", [f"{r.note_count_diff:+d}" for r in results]))
    print(row("Rest count diff", [f"{r.rest_count_diff:+d}" for r in results]))
    print()

    # mir_eval onset+offset
    print(row("Note F1", [f"{r.f1:.3f}" for r in results]))
    print(row("  precision", [f"{r.precision:.3f}" for r in results]))
    print(row("  recall", [f"{r.recall:.3f}" for r in results]))
    print(row("  avg overlap", [f"{r.overlap:.3f}" for r in results]))
    print()

    # mir_eval onset-only
    print(row("Onset F1", [f"{r.onset_f1:.3f}" for r in results]))
    print(row("  precision", [f"{r.onset_precision:.3f}" for r in results]))
    print(row("  recall", [f"{r.onset_recall:.3f}" for r in results]))
    print()

    # Quality
    print(f"  {sep}")
    print(row("QUALITY SCORE", [f"{r.quality_score:.1f}/100" for r in results]))

    # Winner
    if len(results) > 1:
        best_idx = max(range(len(results)), key=lambda i: results[i].quality_score)
        print(f"\n  BEST: {names[best_idx]} ({results[best_idx].quality_score:.1f}/100)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare candidate MusicXML files against ground truth."
    )
    parser.add_argument("ground_truth", type=Path, help="Ground truth MusicXML file.")
    parser.add_argument("candidates", type=Path, nargs="+", help="Candidate MusicXML files to compare.")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if not args.ground_truth.exists():
        print(f"Error: ground truth not found: {args.ground_truth}", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing ground truth: {args.ground_truth.name} ...", file=sys.stderr)
    gt = extract_score_info(args.ground_truth)
    if gt.error:
        print(f"Error parsing ground truth: {gt.error}", file=sys.stderr)
        sys.exit(1)

    results: List[ComparisonResult] = []
    for cand_path in args.candidates:
        if not cand_path.exists():
            print(f"Warning: candidate not found, skipping: {cand_path}", file=sys.stderr)
            results.append(ComparisonResult(candidate_path=str(cand_path), error="file not found"))
            continue
        print(f"Parsing candidate: {cand_path.name} ...", file=sys.stderr)
        cand = extract_score_info(cand_path)
        results.append(compare(gt, cand))

    print_comparison_table(gt, results)

    if args.json:
        import dataclasses
        out = {
            "ground_truth": gt.path,
            "gt_notes": gt.total_notes,
            "gt_measures": gt.num_measures,
            "candidates": [dataclasses.asdict(r) for r in results],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"JSON saved to {args.json}", file=sys.stderr)


if __name__ == "__main__":
    main()
