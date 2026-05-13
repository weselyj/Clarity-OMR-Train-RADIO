"""Cluster bottom-quartile lieder pieces by observable failure mode.

Reads:
  - eval/results/lieder_stage3_v3_best_scores.csv (139 pieces, with onset_f1)
  - eval/results/lieder_stage3_v3_best/<piece_id>.musicxml.diagnostics.json
  - eval/results/lieder_stage3_v3_best/<piece_id>.tokens.jsonl (if available; else re-run predict_pdf)
  - data/openscore_lieder/scores/.../<piece_id>.mxl (for ground-truth clef extraction)

Writes:
  - docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCORES_CSV = REPO_ROOT / "eval/results/lieder_stage3_v3_best_scores.csv"
DEFAULT_TOKENS_DIR = REPO_ROOT / "eval/results/lieder_stage3_v3_best"
DEFAULT_GT_MXL_ROOT = REPO_ROOT / "data/openscore_lieder/scores"
DEFAULT_OUTPUT_MD = REPO_ROOT / "docs/audits/2026-05-13-bottom-quartile-lieder-cluster.md"
BOTTOM_QUARTILE_THRESHOLD = 0.10


def load_bottom_quartile(scores_csv: Path, threshold: float = BOTTOM_QUARTILE_THRESHOLD) -> List[str]:
    bottom = []
    with scores_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if float(row["onset_f1"]) < threshold:
                bottom.append(row["piece_id"])
    return bottom


def extract_ground_truth_clefs(mxl_path: Path) -> List[List[str]]:
    """Return per-system list of clef tokens from a MusicXML source."""
    import music21
    score = music21.converter.parse(str(mxl_path))
    # Extract from each part's first measure; group by system if layout info present.
    # Simplification: return per-part clef tokens for measure 1 (proxy for system 1).
    clefs = []
    for part in score.parts:
        part_clefs = []
        for clef in part.flatten().getElementsByClass(music21.clef.Clef):
            part_clefs.append(_music21_clef_to_token(clef))
            break  # first only
        clefs.append(part_clefs)
    # Return as single-system-equivalent (caller handles per-system pairing).
    return [[c for part in clefs for c in part]]


def _music21_clef_to_token(clef) -> str:
    import music21
    if isinstance(clef, music21.clef.TrebleClef):
        return "clef-G2"
    if isinstance(clef, music21.clef.BassClef):
        return "clef-F4"
    return f"clef-other-{clef.sign}{clef.line}"


def classify_piece(piece_tokens: Dict) -> List[str]:
    """Given a piece's predicted tokens + ground truth, return list of failure-mode tags."""
    tags = []
    systems = piece_tokens.get("systems", [])
    gt_clefs = piece_tokens.get("ground_truth_clefs_by_system", [])

    bass_clef_misread = False
    phantom = False
    for sys_idx, system in enumerate(systems):
        staves = system.get("staves", [])
        # Phantom-staff residual: more than 2 staves
        if len(staves) > 2:
            phantom = True
        # Bass-clef-misread: bottom staff predicted G2, ground truth F4
        if sys_idx < len(gt_clefs) and len(gt_clefs[sys_idx]) >= 2 and len(staves) >= 2:
            bottom_pred = staves[-1].get("clef_pred")
            bottom_gt = gt_clefs[sys_idx][-1]
            if bottom_pred == "clef-G2" and bottom_gt == "clef-F4":
                bass_clef_misread = True

    if bass_clef_misread:
        tags.append("bass-clef-misread")
    if phantom:
        tags.append("phantom-staff-residual")

    # Key/time-signature residual
    gt_time = piece_tokens.get("ground_truth_time_sig")
    pred_time = systems[0].get("time_sig_pred") if systems else None
    if gt_time and pred_time and gt_time != pred_time:
        tags.append("key-time-sig-residual")

    if not tags:
        tags.append("other")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-csv", type=Path, default=DEFAULT_SCORES_CSV)
    ap.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR)
    ap.add_argument("--gt-mxl-root", type=Path, default=DEFAULT_GT_MXL_ROOT)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_MD)
    args = ap.parse_args()

    pieces = load_bottom_quartile(args.scores_csv)
    print(f"Bottom-quartile pieces (onset_f1 < {BOTTOM_QUARTILE_THRESHOLD}): {len(pieces)}")

    cluster_counts: Dict[str, int] = {}
    cluster_examples: Dict[str, List[str]] = {}
    missing_tokens = 0

    for piece_id in pieces:
        tokens_path = args.tokens_dir / f"{piece_id}.tokens.jsonl"
        if not tokens_path.exists():
            print(f"  SKIP {piece_id} — no tokens dump (re-run predict_pdf --dump-tokens)")
            missing_tokens += 1
            continue
        # Build piece_tokens dict from dump + ground truth.
        piece_tokens = _build_piece_tokens(piece_id, tokens_path, args.gt_mxl_root)
        tags = classify_piece(piece_tokens)
        for tag in tags:
            cluster_counts[tag] = cluster_counts.get(tag, 0) + 1
            cluster_examples.setdefault(tag, []).append(piece_id)

    classified = len(pieces) - missing_tokens
    print(f"Classified: {classified}/{len(pieces)} (missing token dumps: {missing_tokens})")
    _write_report(args.output, len(pieces), cluster_counts, cluster_examples, missing_tokens)
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
    gt_clefs = extract_ground_truth_clefs(mxl_candidates[0]) if mxl_candidates else []
    return {"systems": systems, "ground_truth_clefs_by_system": gt_clefs}


def _extract_staves_from_token_stream(tokens: List[str]) -> List[Dict]:
    """Split tokens into staves at <staff_idx_N> markers; extract clef + median pitch per staff."""
    staves: List[Dict] = []
    current: Optional[Dict] = None
    for tok in tokens:
        if tok.startswith("<staff_idx_"):
            if current is not None:
                staves.append(current)
            current = {"clef_pred": None, "pitches": []}
        elif current is not None:
            if tok.startswith("clef-"):
                current["clef_pred"] = tok
            elif tok.startswith("note-"):
                pitch = _pitch_to_octave(tok)
                if pitch is not None:
                    current["pitches"].append(pitch)
    if current is not None:
        staves.append(current)
    for s in staves:
        s["median_octave_pred"] = (
            sorted(s["pitches"])[len(s["pitches"]) // 2] if s["pitches"] else None
        )
        del s["pitches"]
    return staves


def _pitch_to_octave(note_token: str) -> Optional[int]:
    """note-C4 -> 4. note-Eb5 -> 5. Return None if unparseable."""
    import re
    m = re.match(r"note-[A-G][b#]?(-?\d+)", note_token)
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
        lines.append(f"**Missing token dumps:** {missing_tokens}/{total} pieces skipped (re-run predict_pdf --dump-tokens)")
    lines.extend([
        "",
        "## Cluster counts",
        "",
        "| Cluster | Count | % of bottom-quartile |",
        "|---|---:|---:|",
    ])
    for tag, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = (count / total * 100) if total else 0.0
        lines.append(f"| {tag} | {count} | {pct:.1f}% |")
    lines.append("")
    lines.append("## Examples per cluster")
    lines.append("")
    for tag, ids in examples.items():
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
