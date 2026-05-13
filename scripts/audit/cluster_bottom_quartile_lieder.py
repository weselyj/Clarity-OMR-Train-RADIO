"""Cluster bottom-quartile lieder pieces by observable failure mode.

Two input modes:

1. **Token-dump mode** (`--tokens-dir`): per-piece `<piece>.tokens.jsonl` raw decoder
   dumps emitted by `predict_pdf --dump-tokens`. Highest fidelity but requires
   re-running inference if dumps weren't captured.

2. **Predicted-MXL mode** (`--predicted-mxl-dir`): per-piece `<piece>.musicxml`
   predicted-output files. No re-inference required; clef/time-sig/octave signal
   is preserved in the rendered MXL. Use when token dumps are unavailable.

Reads:
  - --scores-csv (with `piece` or `piece_id` column + `onset_f1`)
  - --tokens-dir/<piece>.tokens.jsonl   (token-dump mode), OR
  - --predicted-mxl-dir/<piece>.musicxml (predicted-MXL mode)
  - --gt-mxl-root/.../<piece>.mxl (ground truth, both modes)

Writes:
  - docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES_CSV = REPO_ROOT / "eval/results/lieder_stage3_v3_best_scores.csv"
DEFAULT_TOKENS_DIR = REPO_ROOT / "eval/results/lieder_stage3_v3_best"
DEFAULT_GT_MXL_ROOT = REPO_ROOT / "data/openscore_lieder/scores"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md"
BOTTOM_QUARTILE_THRESHOLD = 0.10


def load_bottom_quartile(scores_csv: Path, threshold: float = BOTTOM_QUARTILE_THRESHOLD) -> List[str]:
    """Load piece IDs whose onset_f1 is below the bottom-quartile threshold.

    Accepts either ``piece`` (preferred — matches the current eval CSV) or
    ``piece_id`` (legacy) as the identifier column.
    """
    bottom = []
    with scores_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            piece_id = row.get("piece") or row.get("piece_id")
            if piece_id is None:
                raise ValueError(
                    f"CSV missing 'piece' or 'piece_id' column. Headers: {reader.fieldnames}"
                )
            if float(row["onset_f1"]) < threshold:
                bottom.append(piece_id)
    return bottom


def extract_part_clefs(score, with_octaves: bool = False) -> Dict:
    """Extract per-part clef / time-sig / staff-count features from a music21 score.

    The score is treated as a flat sequence of parts (each music21 ``Part`` is one
    staff). This works uniformly for ground-truth MXLs and predicted MXLs because
    both encode each staff as a separate ``Part``.

    Args:
        score: A parsed ``music21.stream.Score`` (or compatible Stream with .parts).
        with_octaves: If True, also compute the median pitch octave per part and
            return ``median_octaves_by_part``. Used by the predicted-MXL path.

    Returns:
        A dict with keys:
            - ``clefs``: list of model-vocab clef tokens, one per part (first clef
              in each part). Parts with no clef are skipped (empty entry omitted).
            - ``time_sig``: ``timeSignature-<ratio>`` from the first explicit
              TimeSignature anywhere in the score, or ``None``.
            - ``staff_count``: total number of parts (one staff per part).
            - ``median_octaves_by_part`` (only if with_octaves=True): list with one
              entry per part, value is the median ``pitch.octave`` across notes in
              the part, or ``None`` for empty parts.
    """
    import music21
    clefs: List[str] = []
    median_octaves: List[Optional[int]] = []
    parts = list(score.parts)
    for part in parts:
        flat = part.flatten()
        # First clef in the part (matches the original gt extractor's behavior).
        first_clef = None
        for clef in flat.getElementsByClass(music21.clef.Clef):
            first_clef = clef
            break
        if first_clef is not None:
            clefs.append(_music21_clef_to_token(first_clef))
        if with_octaves:
            octaves = [p.octave for p in flat.pitches if p.octave is not None]
            if octaves:
                octaves_sorted = sorted(octaves)
                median_octaves.append(octaves_sorted[len(octaves_sorted) // 2])
            else:
                median_octaves.append(None)

    # Time signature: take the first explicit one across the score.
    time_sig: Optional[str] = None
    for ts in score.flatten().getElementsByClass(music21.meter.TimeSignature):
        time_sig = f"timeSignature-{ts.ratioString}"
        break

    result: Dict = {
        "clefs": clefs,
        "time_sig": time_sig,
        "staff_count": len(parts),
    }
    if with_octaves:
        result["median_octaves_by_part"] = median_octaves
    return result


def extract_ground_truth_clefs(mxl_path: Path) -> Tuple[List[List[str]], int, Optional[str]]:
    """Extract ground-truth attributes from a MusicXML source.

    Backward-compatible wrapper around :func:`extract_part_clefs` that returns
    the legacy ``(clefs_by_system, gt_staff_count, gt_time_sig)`` tuple shape
    expected by the token-dump code path.

    Returns:
        (clefs_by_system, gt_staff_count, gt_time_sig)

        - clefs_by_system: single-system-equivalent list (flattened per-part clefs)
          for backwards compatibility with classify_piece's system-fallback logic.
        - gt_staff_count: total staff count across all parts.
        - gt_time_sig: model-format time-signature token from the first explicit
          TimeSignature, or None.
    """
    import music21
    score = music21.converter.parse(str(mxl_path))
    parts_info = extract_part_clefs(score, with_octaves=False)
    # Legacy shape: single-system-equivalent (caller handles per-system fallback).
    return [parts_info["clefs"]], parts_info["staff_count"], parts_info["time_sig"]


def _music21_clef_to_token(clef) -> str:
    """Map a music21 Clef instance to the model's clef vocabulary.

    Model vocab (src/tokenizer/vocab.py):
        clef-G2, clef-F4, clef-C3, clef-C4, clef-C1, clef-G2_8vb, clef-G2_8va

    Order matters: subclasses (Treble8vbClef) MUST be tested before parents
    (TrebleClef) because isinstance is True for both.
    """
    import music21
    # Treble family — subclasses first
    if isinstance(clef, music21.clef.Treble8vbClef):
        return "clef-G2_8vb"
    if isinstance(clef, music21.clef.Treble8vaClef):
        return "clef-G2_8va"
    if isinstance(clef, music21.clef.TrebleClef):
        return "clef-G2"
    # Bass family — vocab only has clef-F4; map 8va/8vb/SubBass to it.
    if isinstance(clef, music21.clef.BassClef):
        return "clef-F4"
    if isinstance(clef, music21.clef.SubBassClef):
        return "clef-F4"
    if isinstance(clef, music21.clef.FBaritoneClef):
        return "clef-F4"
    # C-clef family — match by line.
    if isinstance(clef, music21.clef.AltoClef):
        return "clef-C3"
    if isinstance(clef, music21.clef.TenorClef):
        return "clef-C4"
    if isinstance(clef, music21.clef.SopranoClef):
        return "clef-C1"
    if isinstance(clef, music21.clef.MezzoSopranoClef):
        # Mezzo (C2) is not in vocab; convert_tokens normalizes C2 → C3.
        return "clef-C3"
    if isinstance(clef, music21.clef.CBaritoneClef):
        # CBaritone (C5) is not in vocab; convert_tokens normalizes C5 → C4.
        return "clef-C4"
    # Generic fallback: synthesize from sign+line.
    sign = getattr(clef, "sign", "?")
    line = getattr(clef, "line", "?")
    return f"clef-{sign}{line}"


def classify_piece(piece_tokens: Dict) -> List[str]:
    """Given a piece's predicted tokens + ground truth, return list of failure-mode tags."""
    tags = []
    systems = piece_tokens.get("systems", [])
    gt_clefs = piece_tokens.get("ground_truth_clefs_by_system", [])
    gt_staff_count = piece_tokens.get("ground_truth_staff_count")

    bass_clef_misread = False
    phantom = False
    for sys_idx, system in enumerate(systems):
        staves = system.get("staves", [])
        # Phantom-staff residual: model emitted more staves than GT has.
        # Lieder GT often has 3 staves (vocal + piano RH + piano LH), so a
        # hardcoded >2 threshold falsely fires on every vocal+piano piece.
        if gt_staff_count is not None and gt_staff_count > 0:
            if len(staves) > gt_staff_count:
                phantom = True
        else:
            # Fallback: no GT available — use proxy (length of system-0 GT clef list)
            # if any GT exists, else the old hardcoded >2 heuristic with a warning.
            if gt_clefs:
                proxy_count = len(gt_clefs[0]) if gt_clefs[0] else 2
                if len(staves) > proxy_count:
                    phantom = True
            else:
                # Truly no GT — emit a warning so audits can be re-run later.
                # Don't apply the heuristic — without GT we cannot distinguish a
                # legitimate 3-staff lieder system from a phantom-residual case.
                print(
                    "  WARN: no ground-truth staff count available — "
                    "skipping phantom-staff check for this system"
                )
        # Bass-clef-misread: bottom staff predicted G2, ground truth F4.
        # Lieder clef pairs are stable across systems; if GT doesn't have an entry
        # for this system, fall back to system 0.
        gt_for_sys = gt_clefs[sys_idx] if sys_idx < len(gt_clefs) else (gt_clefs[0] if gt_clefs else [])
        if len(gt_for_sys) >= 2 and len(staves) >= 2:
            bottom_pred = staves[-1].get("clef_pred")
            bottom_gt = gt_for_sys[-1]
            if bottom_pred == "clef-G2" and bottom_gt == "clef-F4":
                bass_clef_misread = True

    if bass_clef_misread:
        tags.append("bass-clef-misread")
    if phantom:
        tags.append("phantom-staff-residual")

    # Key/time-signature residual.  Aggregate per-system pred by taking the first
    # non-None across staves; compare against GT time signature.
    gt_time = piece_tokens.get("ground_truth_time_sig")
    pred_time = None
    if systems:
        # Allow either a system-level pred (set explicitly by callers/tests) or
        # per-staff time_sig_pred populated by _extract_staves_from_token_stream.
        pred_time = systems[0].get("time_sig_pred")
        if pred_time is None:
            for staff in systems[0].get("staves", []):
                if staff.get("time_sig_pred"):
                    pred_time = staff["time_sig_pred"]
                    break
    if gt_time and pred_time and gt_time != pred_time:
        tags.append("key-time-sig-residual")

    if not tags:
        tags.append("other")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Cluster bottom-quartile lieder pieces by observable failure mode. "
            "Supports two input modes: --tokens-dir (raw decoder dumps from "
            "predict_pdf --dump-tokens) or --predicted-mxl-dir (parses predicted "
            "MusicXML files). Exactly one of the two must be supplied."
        )
    )
    ap.add_argument(
        "--scores-csv",
        type=Path,
        default=DEFAULT_SCORES_CSV,
        help=(
            "CSV with per-piece onset_f1. Accepts either 'piece' or 'piece_id' "
            "as the identifier column."
        ),
    )
    ap.add_argument(
        "--tokens-dir",
        type=Path,
        default=None,
        help=(
            "Directory of per-piece <piece>.tokens.jsonl raw decoder dumps. "
            "Mutually exclusive with --predicted-mxl-dir."
        ),
    )
    ap.add_argument(
        "--predicted-mxl-dir",
        type=Path,
        default=None,
        help=(
            "Directory of per-piece <piece>.musicxml predicted-output files. "
            "Mutually exclusive with --tokens-dir."
        ),
    )
    ap.add_argument(
        "--gt-mxl-root",
        type=Path,
        default=DEFAULT_GT_MXL_ROOT,
        help="Directory tree containing ground-truth <piece>.mxl files (searched recursively).",
    )
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD)
    args = ap.parse_args()

    # Validate input mode selection.
    if args.tokens_dir is not None and args.predicted_mxl_dir is not None:
        ap.error("--tokens-dir and --predicted-mxl-dir are mutually exclusive; provide exactly one.")
    if args.tokens_dir is None and args.predicted_mxl_dir is None:
        ap.error(
            "must provide one of --tokens-dir (raw decoder dumps) or "
            "--predicted-mxl-dir (predicted MXL files)."
        )

    pieces = load_bottom_quartile(args.scores_csv)
    print(f"Bottom-quartile pieces (onset_f1 < {BOTTOM_QUARTILE_THRESHOLD}): {len(pieces)}")

    cluster_counts: Dict[str, int] = {}
    cluster_examples: Dict[str, List[str]] = {}
    missing_inputs = 0

    use_mxl_mode = args.predicted_mxl_dir is not None
    for piece_id in pieces:
        if use_mxl_mode:
            mxl_path = args.predicted_mxl_dir / f"{piece_id}.musicxml"
            if not mxl_path.exists():
                print(f"  SKIP {piece_id} — no predicted MXL at {mxl_path}")
                missing_inputs += 1
                continue
            piece_tokens = _build_piece_tokens_from_mxl(piece_id, mxl_path, args.gt_mxl_root)
        else:
            tokens_path = args.tokens_dir / f"{piece_id}.tokens.jsonl"
            if not tokens_path.exists():
                print(f"  SKIP {piece_id} — no tokens dump (re-run predict_pdf --dump-tokens)")
                missing_inputs += 1
                continue
            piece_tokens = _build_piece_tokens(piece_id, tokens_path, args.gt_mxl_root)

        tags = classify_piece(piece_tokens)
        for tag in tags:
            cluster_counts[tag] = cluster_counts.get(tag, 0) + 1
            cluster_examples.setdefault(tag, []).append(piece_id)

    classified = len(pieces) - missing_inputs
    label = "predicted MXLs" if use_mxl_mode else "token dumps"
    print(f"Classified: {classified}/{len(pieces)} (missing {label}: {missing_inputs})")
    _write_report(args.output, len(pieces), cluster_counts, cluster_examples, missing_inputs)
    return 0


def _build_piece_tokens(piece_id: str, tokens_path: Path, gt_root: Path) -> Dict:
    """Parse the --dump-tokens JSONL into the structure classify_piece expects."""
    systems = []
    with tokens_path.open() as f:
        for line in f:
            entry = json.loads(line)
            # Each line is one system; extract per-staff clef + median pitch.
            staves = _extract_staves_from_token_stream(entry["tokens"])
            systems.append({"staves": staves})
    # Locate matching .mxl in gt_root recursively
    mxl_candidates = list(gt_root.rglob(f"{piece_id}.mxl"))
    if mxl_candidates:
        gt_clefs, gt_staff_count, gt_time_sig = extract_ground_truth_clefs(mxl_candidates[0])
    else:
        print(f"  WARN: no GT MXL found for {piece_id} under {gt_root} — phantom + time-sig checks degraded")
        gt_clefs, gt_staff_count, gt_time_sig = [], None, None
    return {
        "systems": systems,
        "ground_truth_clefs_by_system": gt_clefs,
        "ground_truth_staff_count": gt_staff_count,
        "ground_truth_time_sig": gt_time_sig,
    }


def _build_piece_tokens_from_mxl(piece_id: str, predicted_mxl_path: Path, gt_root: Path) -> Dict:
    """Parse a predicted MusicXML into the structure classify_piece expects.

    Unlike the token-dump path, predicted MXLs don't preserve per-system boundaries
    — music21 flattens the score back to a list of parts. We model this as a
    single "system" containing all predicted parts as staves, which is the same
    shape used by :func:`extract_ground_truth_clefs` for GT extraction. This
    matches how ``classify_piece`` handles the single-system-GT case via its
    sys-0 fallback.
    """
    import music21
    score = music21.converter.parse(str(predicted_mxl_path))
    pred = extract_part_clefs(score, with_octaves=True)
    # Predicted MXL doesn't have per-staff time-sig granularity in music21's
    # part model; we attach the score-level time signature as the system pred.
    staves: List[Dict] = []
    median_octaves = pred.get("median_octaves_by_part", [])
    for i, clef in enumerate(pred["clefs"]):
        med_oct = median_octaves[i] if i < len(median_octaves) else None
        staves.append({
            "clef_pred": clef,
            "median_octave_pred": med_oct,
        })
    systems = [{
        "staves": staves,
        "time_sig_pred": pred["time_sig"],
    }]

    # Ground truth (same lookup as the token-dump path).
    mxl_candidates = list(gt_root.rglob(f"{piece_id}.mxl"))
    if mxl_candidates:
        gt_clefs, gt_staff_count, gt_time_sig = extract_ground_truth_clefs(mxl_candidates[0])
    else:
        print(f"  WARN: no GT MXL found for {piece_id} under {gt_root} — phantom + time-sig checks degraded")
        gt_clefs, gt_staff_count, gt_time_sig = [], None, None
    return {
        "systems": systems,
        "ground_truth_clefs_by_system": gt_clefs,
        "ground_truth_staff_count": gt_staff_count,
        "ground_truth_time_sig": gt_time_sig,
    }


def _extract_staves_from_token_stream(tokens: List[str]) -> List[Dict]:
    """Split a system's token stream at <staff_start>/<staff_end> boundaries.

    Returns one dict per staff with:
        - clef_pred: first clef-* token in the chunk, or None
        - median_octave_pred: median octave across note-* tokens, or None
        - time_sig_pred: first timeSignature-* token in the chunk, or None
        - key_sig_pred: first keySignature-* token in the chunk, or None

    Uses the canonical splitter from src.pipeline.post_decode so the staff
    boundary logic stays in one place. <staff_idx_N> markers are optional
    (single-staff systems omit them); the splitter handles either form.
    """
    from src.pipeline.post_decode import split_system_tokens_into_staves
    chunks = split_system_tokens_into_staves(tokens)
    staves: List[Dict] = []
    for chunk in chunks:
        clef = next((t for t in chunk if t.startswith("clef-")), None)
        time_sig = next((t for t in chunk if t.startswith("timeSignature-")), None)
        key_sig = next((t for t in chunk if t.startswith("keySignature-")), None)
        octaves = [_pitch_to_octave(t) for t in chunk if t.startswith("note-")]
        octaves = [o for o in octaves if o is not None]
        median_octave = sorted(octaves)[len(octaves) // 2] if octaves else None
        staves.append({
            "clef_pred": clef,
            "median_octave_pred": median_octave,
            "time_sig_pred": time_sig,
            "key_sig_pred": key_sig,
        })
    return staves


def _pitch_to_octave(note_token: str) -> Optional[int]:
    """note-C4 -> 4. note-Eb5 -> 5. note-Bbb3 -> 3. Return None if unparseable."""
    import re
    m = re.match(r"note-[A-G][b#]{0,2}(-?\d+)", note_token)
    return int(m.group(1)) if m else None


def _write_report(
    output: Path,
    total: int,
    counts: Dict[str, int],
    examples: Dict[str, List[str]],
    missing_tokens: int = 0,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Bottom-Quartile Lieder Failure-Mode Cluster (Stage 3 v3 best.pt)",
        "",
        f"**Date:** 2026-05-13",
        f"**Input:** `eval/results/lieder_stage3_v3_best_scores.csv` — {total} pieces with onset_f1 < {BOTTOM_QUARTILE_THRESHOLD}",
    ]
    if missing_tokens:
        lines.append(
            f"**Missing inputs:** {missing_tokens}/{total} pieces skipped "
            "(no token dump or predicted MXL for these pieces)"
        )
    lines.extend([
        "",
        "## Cluster counts",
        "",
        "| Cluster | Count | % of bottom-quartile |",
        "|---|---:|---:|",
    ])
    sorted_tags = sorted(counts.items(), key=lambda x: -x[1])
    for tag, count in sorted_tags:
        pct = (count / total * 100) if total else 0.0
        lines.append(f"| {tag} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Examples per cluster")
    lines.append("")
    # Sort examples by the same order as cluster counts so the two sections align.
    for tag, _count in sorted_tags:
        ids = examples.get(tag, [])
        lines.append(f"### {tag}")
        for piece_id in ids[:5]:
            lines.append(f"- `{piece_id}`")
        lines.append("")
    lines.append("## Gate check")
    lines.append("")
    bcm = counts.get("bass-clef-misread", 0)
    gate_pct = (bcm / total * 100) if total else 0.0
    lines.append(
        f"`bass-clef-misread` = {bcm}/{total} = {gate_pct:.1f}% of bottom-quartile."
    )
    lines.append("")
    if gate_pct >= 30:
        lines.append("**GATE PASS:** >=30% -> proceed to Phase 1.")
    else:
        lines.append("**GATE FAIL:** <30% -> pause and re-scope before Phase 1.")
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    raise SystemExit(main())
