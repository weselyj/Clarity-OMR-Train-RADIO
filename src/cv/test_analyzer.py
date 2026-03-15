#!/usr/bin/env python3
"""Test the staff analyzer on sample crops and visualize results.

Usage:
    # Single image:
    python src/cv/test_analyzer.py --image path/to/crop.png

    # Batch from eval crops manifest:
    python src/cv/test_analyzer.py --manifest src/eval/checkpoint_eval_stage2_prev_71000/stageb_eval_crops_manifest.jsonl --limit 20

    # With known clef:
    python src/cv/test_analyzer.py --image crop.png --clef clef-G2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import cv2

from src.cv.staff_analyzer import analyze_staff, draw_analysis


def process_single(image_path: str, clef_hint: str | None, output_dir: Path) -> None:
    print(f"\nAnalyzing: {image_path}")
    skeleton = analyze_staff(image_path, clef_hint=clef_hint)
    print(skeleton.summary())

    # Save annotated image
    annotated = draw_analysis(image_path, skeleton)
    out_name = Path(image_path).stem + "_cv_analysis.png"
    out_path = output_dir / out_name
    cv2.imwrite(str(out_path), annotated)
    print(f"  Saved: {out_path}")

    # Print onset details
    for i, oc in enumerate(skeleton.onset_clusters):
        pitches = [n.estimated_pitch or "?" for n in oc.noteheads]
        chord_tag = " [CHORD]" if oc.is_chord else ""
        print(f"  Onset {i}: x={oc.x_center:.0f} notes={pitches}{chord_tag} conf={oc.confidence:.2f}")


def process_manifest(manifest_path: str, limit: int, clef_hint: str | None, output_dir: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entries.append(json.loads(line))

    if limit > 0:
        entries = entries[:limit]

    stats = {"total": 0, "staff_lines_found": 0, "noteheads_total": 0, "chords_total": 0}

    for entry in entries:
        image_path = entry.get("crop_path") or entry.get("image_path", "")
        if not image_path:
            continue
        # Resolve relative paths
        full_path = Path(image_path)
        if not full_path.is_absolute():
            full_path = project_root / full_path
        if not full_path.exists():
            print(f"  Skip (missing): {full_path}")
            continue

        # Use ground truth clef if available
        clef = clef_hint
        gt_tokens = entry.get("gt_tokens") or entry.get("token_sequence") or []
        if not clef and isinstance(gt_tokens, list):
            for t in gt_tokens:
                if str(t).startswith("clef-"):
                    clef = str(t)
                    break

        skeleton = analyze_staff(str(full_path), clef_hint=clef)
        stats["total"] += 1
        if skeleton.staff_lines:
            stats["staff_lines_found"] += 1
        stats["noteheads_total"] += skeleton.total_note_count
        stats["chords_total"] += sum(1 for c in skeleton.onset_clusters if c.is_chord)

        # Save annotated image
        annotated = draw_analysis(str(full_path), skeleton)
        out_name = Path(full_path).stem + "_cv.png"
        cv2.imwrite(str(output_dir / out_name), annotated)

        # Compare with ground truth note count
        gt_notes = sum(1 for t in gt_tokens if str(t).startswith("note-")) if isinstance(gt_tokens, list) else 0
        gt_chords = sum(1 for t in gt_tokens if str(t) == "<chord_start>") if isinstance(gt_tokens, list) else 0
        match_icon = "=" if skeleton.total_note_count == gt_notes else ("~" if abs(skeleton.total_note_count - gt_notes) <= 2 else "X")
        print(f"  [{match_icon}] {Path(full_path).name}: CV={skeleton.total_note_count} notes, "
              f"{sum(1 for c in skeleton.onset_clusters if c.is_chord)} chords | "
              f"GT={gt_notes} notes, {gt_chords} chords | "
              f"staff_lines={len(skeleton.staff_lines.y_positions) if skeleton.staff_lines else 0}")

    print(f"\n--- Summary ---")
    print(f"  Processed: {stats['total']}")
    print(f"  Staff lines found: {stats['staff_lines_found']}/{stats['total']}")
    print(f"  Total noteheads detected: {stats['noteheads_total']}")
    print(f"  Total chords detected: {stats['chords_total']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test CV staff analyzer.")
    parser.add_argument("--image", type=str, help="Single image path.")
    parser.add_argument("--manifest", type=str, help="JSONL manifest with crop entries.")
    parser.add_argument("--limit", type=int, default=20, help="Max samples from manifest.")
    parser.add_argument("--clef", type=str, default=None, help="Clef hint (e.g. clef-G2).")
    parser.add_argument("--output-dir", type=Path, default=Path("src/cv/test_output"),
                        help="Directory for annotated output images.")
    args = parser.parse_args()

    if not args.image and not args.manifest:
        parser.error("Provide --image or --manifest")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        process_single(args.image, args.clef, args.output_dir)
    elif args.manifest:
        process_manifest(args.manifest, args.limit, args.clef, args.output_dir)


if __name__ == "__main__":
    main()
