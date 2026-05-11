"""Stream B: per-stage note-count diagnostic for the inference pipeline.

Runs Stage A + Stage B + Stage D end-to-end on one piece, but hooks the
intermediate boundaries to count tokens matching `^note-` at each stage.
Final count is from re-parsing the written MusicXML via music21.
Compares against the reference .mxl note count.

The most informative comparison is the drop BETWEEN stages: if the raw
decoder produces N notes and the final MusicXML has 0.2*N, the assembly
pipeline is dropping notes. If the raw decoder produces 0.2*N already,
the decoder is the bottleneck.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.pipeline_note_loss \\
        --pdf data\\clarity_demo\\pdf\\clair-de-lune-debussy.pdf \\
        --ref data\\clarity_demo\\mxl\\clair-de-lune-debussy.mxl \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --yolo-weights runs\\detect\\runs\\yolo26m_systems\\weights\\best.pt \\
        --out audit_results\\pipeline_note_loss_clair_de_lune.json
"""
from __future__ import annotations
import argparse
import json
import sys
import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

_NOTE_RE = re.compile(r"^note-")


def _count_note_tokens(tokens) -> int:
    """Count tokens matching `^note-` in a sequence."""
    return sum(1 for t in tokens if isinstance(t, str) and _NOTE_RE.match(t))


def _count_music21_notes(score) -> int:
    """Count Note objects (including those inside Chords) in a music21 score, after stripTies."""
    import music21
    s = score.stripTies()
    flat = s.flatten()
    n_notes = len(flat.getElementsByClass(music21.note.Note))
    # Each Chord contains multiple notes — count them too.
    for ch in flat.getElementsByClass(music21.chord.Chord):
        n_notes += len(ch.notes)
    return n_notes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pdf", type=Path, required=True)
    p.add_argument("--ref", type=Path, required=True,
                   help="Reference .mxl file for ground-truth note count")
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--yolo-weights", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--page-dpi", type=int, default=300)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    args = p.parse_args()

    import music21
    import torch
    from PIL import Image
    import fitz

    from src.inference.system_pipeline import (
        SystemInferencePipeline,
    )
    from src.data.convert_tokens import _split_staff_sequences_for_validation
    from src.pipeline.assemble_score import (
        assemble_score,
        _enforce_global_key_time,
        StaffRecognitionResult,
        StaffLocation,
    )
    from src.pipeline.export_musicxml import (
        StageDExportDiagnostics,
        assembled_score_to_music21_with_diagnostics,
    )

    # --- Reference count ---
    ref_score = music21.converter.parse(str(args.ref))
    ref_notes = _count_music21_notes(ref_score)
    print(f"Reference notes (after stripTies): {ref_notes}")

    # --- Build pipeline (loads model once) ---
    pipeline = SystemInferencePipeline(
        yolo_weights=args.yolo_weights,
        stage_b_ckpt=args.stage_b_ckpt,
        page_dpi=args.page_dpi,
        image_height=args.image_height,
        image_max_width=args.image_max_width,
        max_decode_steps=args.max_decode_steps,
    )

    # --- Stage 1: Raw decoder output (per system) ---
    # Re-run the page+system loop manually so we can capture intermediates.
    all_token_lists = []
    all_locations = []
    with fitz.open(str(args.pdf)) as doc:
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(dpi=args.page_dpi)
            img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
            systems = pipeline._stage_a.detect_systems(img)
            for sys_d in systems:
                x1, y1, x2, y2 = sys_d["bbox_extended"]
                crop = img.crop((int(x1), int(y1), int(x2), int(y2)))
                tokens = pipeline._decode_one_crop(crop)
                all_token_lists.append(tokens)
                all_locations.append({
                    "system_index": sys_d["system_index"],
                    "bbox": sys_d["bbox_extended"],
                    "page_index": page_index,
                    "conf": sys_d["conf"],
                })

    stage1_count = sum(_count_note_tokens(tokens) for tokens in all_token_lists)
    print(f"Stage 1 (raw decoder output): {stage1_count} note tokens across {len(all_token_lists)} systems")

    # --- Stage 2: After staff split ---
    stage2_count = 0
    stage2_skipped_systems = 0
    staves = []  # for stage 3+
    for sys_tokens, sys_loc in zip(all_token_lists, all_locations):
        try:
            per_staff = _split_staff_sequences_for_validation(sys_tokens)
        except ValueError:
            stage2_skipped_systems += 1
            continue
        if not per_staff:
            continue
        n = len(per_staff)
        sys_idx = int(sys_loc["system_index"])
        page_idx = int(sys_loc.get("page_index", 0))
        x1, y1, x2, y2 = sys_loc["bbox"]
        sys_h = float(y2) - float(y1)
        for i, staff_tokens in enumerate(per_staff):
            stage2_count += _count_note_tokens(staff_tokens)
            y_top = float(y1) + i * sys_h / n
            y_bottom = float(y1) + (i + 1) * sys_h / n
            location = StaffLocation(
                page_index=page_idx,
                y_top=y_top, y_bottom=y_bottom,
                x_left=float(x1), x_right=float(x2),
            )
            staves.append(StaffRecognitionResult(
                sample_id=f"page{page_idx:04d}_sys{sys_idx:02d}_staff{i:02d}",
                tokens=list(staff_tokens),
                location=location,
                system_index_hint=sys_idx,
            ))
    print(f"Stage 2 (after staff split): {stage2_count} note tokens across {len(staves)} staves (skipped {stage2_skipped_systems} systems)")

    # --- Stage 3: After _enforce_global_key_time + post_process_tokens (run via assemble_score) ---
    # assemble_score runs _enforce_global_key_time + per-system _normalize_measure_count + post_process_tokens.
    score = assemble_score(staves)
    stage3_count = 0
    for system in score.systems:
        for staff in system.staves:
            stage3_count += _count_note_tokens(staff.tokens)
    print(f"Stage 3 (after _enforce_global_key_time + post_process): {stage3_count} note tokens")

    # --- Stage 4: AssembledScore tokens (identical to stage 3 in current code, but capture separately for symmetry) ---
    stage4_count = stage3_count

    # --- Stage 5: music21 Note count (in-memory) ---
    diagnostics = StageDExportDiagnostics()
    music_score = assembled_score_to_music21_with_diagnostics(score, diagnostics, strict=False)
    stage5_count = _count_music21_notes(music_score)
    print(f"Stage 5 (music21 in-memory Notes): {stage5_count}")

    # --- Stage 6: Re-parse the written MusicXML ---
    out_musicxml = args.out.with_suffix(".musicxml")
    music_score.write("musicxml", str(out_musicxml))
    reparsed = music21.converter.parse(str(out_musicxml))
    stage6_count = _count_music21_notes(reparsed)
    print(f"Stage 6 (re-parsed MusicXML): {stage6_count}")

    # --- Diagnostics ---
    print()
    print("Stage D diagnostics (informational):")
    print(f"  padded_measures: {diagnostics.padded_measures}")
    print(f"  skipped_systems: {diagnostics.skipped_systems}")
    print(f"  skipped_notes: {diagnostics.skipped_notes}")
    print(f"  skipped_chords: {diagnostics.skipped_chords}")
    print(f"  missing_durations: {diagnostics.missing_durations}")
    print(f"  unknown_tokens: {diagnostics.unknown_tokens}")
    print(f"  fallback_rests: {diagnostics.fallback_rests}")

    results = {
        "experiment": "pipeline_note_loss",
        "pdf": str(args.pdf),
        "ref": str(args.ref),
        "checkpoint": str(args.stage_b_ckpt),
        "reference_notes": ref_notes,
        "stages": {
            "1_raw_decoder_output": stage1_count,
            "2_after_staff_split": stage2_count,
            "3_after_enforce_and_post_process": stage3_count,
            "4_assembled_score_tokens": stage4_count,
            "5_music21_in_memory": stage5_count,
            "6_reparsed_musicxml": stage6_count,
        },
        "deltas": {
            "ref_to_stage1": stage1_count - ref_notes,
            "stage1_to_stage2": stage2_count - stage1_count,
            "stage2_to_stage3": stage3_count - stage2_count,
            "stage3_to_stage5": stage5_count - stage3_count,
            "stage5_to_stage6": stage6_count - stage5_count,
        },
        "n_systems": len(all_token_lists),
        "n_staves": len(staves),
        "stage2_skipped_systems": stage2_skipped_systems,
        "stage_d_diagnostics": {
            "padded_measures": diagnostics.padded_measures,
            "skipped_systems": diagnostics.skipped_systems,
            "skipped_notes": diagnostics.skipped_notes,
            "skipped_chords": diagnostics.skipped_chords,
            "missing_durations": diagnostics.missing_durations,
            "unknown_tokens": diagnostics.unknown_tokens,
            "fallback_rests": diagnostics.fallback_rests,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
