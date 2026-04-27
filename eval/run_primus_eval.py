"""Run a trained Stage B checkpoint against the PrIMuS test split (monophonic).

Unlike eval/run_lieder_eval.py, this skips Stage A YOLO entirely: PrIMuS
samples are already single-staff crops. We feed the existing
_run_stage_b_inference_with_progress (which detects DoRA, applies it
correctly, and enforces load coverage) with a crops manifest pointing at
the test-split images, then compare predicted token sequences against the
ground truth tokens stored in the same token manifest.

This gives a real Stage 1 quality signal -- the model trained on monophonic
data, evaluated against a held-out monophonic test set. Lieder (run_lieder_eval.py)
is the wrong eval for Stage 1 (polyphonic out-of-distribution).

Metrics:
  exact_match_rate   fraction of samples where predicted tokens == ground truth exactly
  token_accuracy     position-aligned token accuracy (truncated to min length)
  ser                Symbol Error Rate = edit_distance(pred, gt) / len(gt)
                     (the standard PrIMuS metric; lower is better, ~0.02-0.05 is SoTA)

Usage:
    venv\\Scripts\\python -m eval.run_primus_eval \\
        --checkpoint checkpoints/full_radio_stage1/stage1-radio-monophonic-foundation_step_0004000.pt \\
        --name stage1 \\
        --limit 100
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# TEDn / linearized-SER are only available when the caller supplies --musicxml-dir
# pointing to a directory of reference MusicXML files named <sample_id>.musicxml
# alongside a directory of predicted MusicXML files (--pred-musicxml-dir).
# When either directory is absent the two columns are emitted as empty strings.
def _try_compute_tedn(ref_xml: Path, hyp_xml: Path) -> "str | float":
    """Return TEDn float or empty string on any error."""
    try:
        from eval.tedn import compute_tedn
        return compute_tedn(ref_xml, hyp_xml)
    except Exception:
        return ""


def _try_compute_lin_ser(ref_xml: Path, hyp_xml: Path) -> "str | float":
    """Return linearized_ser float or empty string on any error."""
    try:
        from eval.linearized_musicxml import compute_linearized_ser
        return compute_linearized_ser(ref_xml, hyp_xml)
    except Exception:
        return ""


def _edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    """Standard Levenshtein distance over token sequences."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        curr[0] = i
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[len(b)]


def _strip_special(tokens: Sequence[str]) -> List[str]:
    """Strip BOS/EOS/staff-start/staff-end so metrics focus on content."""
    skip = {"<bos>", "<eos>", "<staff_start>", "<staff_end>", "<pad>"}
    return [t for t in tokens if t not in skip]


def _load_primus_test(token_manifest: Path, limit: int | None) -> List[Dict]:
    rows: List[Dict] = []
    with token_manifest.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("dataset") != "primus" or entry.get("split") != "test":
                continue
            rows.append(entry)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="path to trained Stage B .pt checkpoint",
    )
    p.add_argument(
        "--name",
        required=True,
        help="run name (used for results CSV filename)",
    )
    p.add_argument(
        "--token-manifest",
        type=Path,
        default=_REPO_ROOT / "src" / "data" / "manifests" / "token_manifest_full.jsonl",
        help="token manifest with primus test entries (default: token_manifest_full.jsonl)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=100,
        help="cap on number of test pieces (default 100; use 0 for full ~8835 split)",
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="torch device (default: cuda)",
    )
    p.add_argument(
        "--beam-width",
        type=int,
        default=1,
        help="Stage-B beam width (default 1 = greedy)",
    )
    p.add_argument(
        "--max-decode-steps",
        type=int,
        default=256,
        help="Stage-B max decode steps per staff (default 256)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "eval" / "results",
        help="directory for results CSV (default: eval/results)",
    )
    p.add_argument(
        "--musicxml-dir",
        type=Path,
        default=None,
        help=(
            "Optional: directory containing reference MusicXML files named "
            "<sample_id>.musicxml. When provided together with --pred-musicxml-dir, "
            "tedn and linearized_ser columns are populated."
        ),
    )
    p.add_argument(
        "--pred-musicxml-dir",
        type=Path,
        default=None,
        help=(
            "Optional: directory containing predicted MusicXML files named "
            "<sample_id>.musicxml. Required alongside --musicxml-dir to enable "
            "TEDn / linearized_ser metrics."
        ),
    )
    args = p.parse_args()

    limit = None if args.limit <= 0 else int(args.limit)
    print(f"Loading primus test entries from {args.token_manifest} ...", file=sys.stderr)
    entries = _load_primus_test(args.token_manifest, limit)
    if not entries:
        raise SystemExit("No primus test entries found in token manifest.")
    print(f"  loaded {len(entries)} entries", file=sys.stderr)

    # Build a crops manifest. Schema matches what
    # _run_stage_b_inference_with_progress expects: each row has crop_path
    # (string, possibly relative). The function resolves relative paths
    # against project_root.
    work = Path(tempfile.mkdtemp(prefix="primus_eval_"))
    crops_manifest_path = work / "primus_crops.jsonl"
    sample_id_by_index: Dict[int, str] = {}
    with crops_manifest_path.open("w", encoding="utf-8") as fh:
        for idx, entry in enumerate(entries):
            row = {
                "sample_id": entry["sample_id"],
                "crop_path": str(entry["image_path"]),
            }
            sample_id_by_index[idx] = entry["sample_id"]
            fh.write(json.dumps(row) + "\n")

    predictions_path = work / "primus_predictions.jsonl"

    print(f"Running Stage B inference on {len(entries)} samples ...", file=sys.stderr)
    from src.eval.evaluate_stage_b_checkpoint import _run_stage_b_inference_with_progress

    _run_stage_b_inference_with_progress(
        project_root=_REPO_ROOT,
        crops_manifest=crops_manifest_path,
        output_predictions=predictions_path,
        checkpoint=args.checkpoint,
        beam_width=int(args.beam_width),
        max_decode_steps=int(args.max_decode_steps),
        image_height=250,
        image_max_width=2500,
        device_name=str(args.device),
        progress_every_seconds=10.0,
        quiet=False,
        length_penalty_alpha=0.4,
        use_kv_cache=True,
        use_fp16=False,
        quantize=False,
    )

    # Load predictions; key by sample_id.
    pred_by_id: Dict[str, List[str]] = {}
    with predictions_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            pred = json.loads(line)
            pred_by_id[str(pred["sample_id"])] = list(pred.get("tokens", []))

    # Determine whether MusicXML-based metrics are available.
    musicxml_enabled = (
        args.musicxml_dir is not None and args.pred_musicxml_dir is not None
    )
    if musicxml_enabled:
        print(
            f"MusicXML metrics enabled: ref={args.musicxml_dir}  pred={args.pred_musicxml_dir}",
            file=sys.stderr,
        )

    # Compute metrics.
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / f"primus_{args.name}.csv"
    sers: List[float] = []
    token_accs: List[float] = []
    exact_matches = 0
    missing_predictions = 0
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "sample_id",
                "gt_len",
                "pred_len",
                "edit_distance",
                "ser",
                "token_accuracy",
                "exact_match",
                "tedn",
                "linearized_ser",
            ]
        )
        for entry in entries:
            sid = str(entry["sample_id"])
            gt_full = list(entry.get("token_sequence", []))
            pred_full = pred_by_id.get(sid)

            # Compute MusicXML-based metrics if dirs are available.
            tedn_val: "str | float" = ""
            lin_ser_val: "str | float" = ""
            if musicxml_enabled:
                ref_xml = args.musicxml_dir / f"{sid}.musicxml"
                hyp_xml = args.pred_musicxml_dir / f"{sid}.musicxml"
                if ref_xml.exists() and hyp_xml.exists():
                    tedn_val = _try_compute_tedn(ref_xml, hyp_xml)
                    lin_ser_val = _try_compute_lin_ser(ref_xml, hyp_xml)

            if pred_full is None:
                missing_predictions += 1
                writer.writerow([sid, len(gt_full), 0, len(gt_full), 1.0, 0.0, False, tedn_val, lin_ser_val])
                continue
            gt = _strip_special(gt_full)
            pred = _strip_special(pred_full)
            edit = _edit_distance(pred, gt)
            ser = edit / max(1, len(gt))
            min_len = min(len(pred), len(gt))
            matches = sum(1 for i in range(min_len) if pred[i] == gt[i])
            token_acc = matches / max(1, len(gt))
            exact = pred == gt
            if exact:
                exact_matches += 1
            sers.append(ser)
            token_accs.append(token_acc)
            tedn_fmt = f"{tedn_val:.4f}" if isinstance(tedn_val, float) else tedn_val
            lin_ser_fmt = f"{lin_ser_val:.4f}" if isinstance(lin_ser_val, float) else lin_ser_val
            writer.writerow(
                [sid, len(gt), len(pred), edit, f"{ser:.4f}", f"{token_acc:.4f}", exact, tedn_fmt, lin_ser_fmt]
            )

    print()
    print(f"=== PrIMuS Eval Results ({args.name}) ===")
    print(f"Pieces evaluated: {len(sers)} / {len(entries)} (missing: {missing_predictions})")
    if sers:
        print(f"Mean SER:         {statistics.mean(sers):.4f}   (lower is better; SoTA ~0.02-0.05)")
        print(f"Median SER:       {statistics.median(sers):.4f}")
        print(f"Min SER:          {min(sers):.4f}")
        print(f"Max SER:          {max(sers):.4f}")
        print(f"Mean token acc:   {statistics.mean(token_accs):.4f}")
        print(f"Exact-match rate: {exact_matches}/{len(sers)} = {exact_matches / max(1, len(sers)):.4f}")
    print(f"Per-sample CSV:   {csv_path}")
    print()


if __name__ == "__main__":
    main()
