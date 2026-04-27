"""Score per-piece metric results for the 4 canonical demo pieces.

Two-pass design:
- Pass 1 (run_clarity_demo_eval.py): inference only — writes predicted MusicXML
  and optional diagnostics sidecars to --predictions-dir.
- Pass 2 (this script): scoring only — subprocess-isolates per-piece metric
  computation so each piece's music21/zss memory is fully reclaimed after the
  subprocess exits. The parent process stays small throughout.

Each piece is scored in up to TWO fresh child processes (eval._score_one_piece):

  1. Cheap pair (onset_f1 + linearized_ser): 60 s timeout. These metrics finish
     in <2 s even for large scores (<=43 MB data) and are always attempted first.
  2. Tedn-only: 300 s timeout.  May time out for very large scores (Clair de
     Lune peaks >37 GB); when it does the cheap metrics are still recovered.

If only cheap metrics were requested (or only tedn), a single subprocess is
used.  The split only activates when the requested set includes both tedn and
at least one cheap metric.

Usage:
    venv-cu132\\Scripts\\python -m eval.score_demo_eval \\
        --predictions-dir eval/results/clarity_demo_stage2_best \\
        --reference-dir data/clarity_demo/mxl \\
        --name stage2_best

Output: eval/results/clarity_demo_<name>.csv
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Path to our venv's Python -- same one that ran inference
VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv-cu132" / "Scripts" / "python.exe"

# Timeout for the cheap-pair subprocess (onset_f1 + linearized_ser)
CHEAP_TIMEOUT_SEC = 60

# Timeout for tedn (may OOM/hang on very large scores like Clair de Lune)
TEDN_TIMEOUT_SEC = 300

# Backward-compat alias
SCORE_TIMEOUT_SEC = TEDN_TIMEOUT_SEC

# Metrics considered "cheap" -- fast and low-memory
CHEAP_METRICS = frozenset({"onset_f1", "linearized_ser"})

# Canonical demo stems -- must match run_clarity_demo_eval.py
DEMO_STEMS = [
    "clair-de-lune-debussy",
    "fugue-no-2-bwv-847-in-c-minor",
    "gnossienne-no-1",
    "prelude-in-d-flat-major-op31-no1-scriabin",
]

CSV_HEADER = [
    "piece", "onset_f1", "tedn", "linearized_ser",
    "stage_d_skipped_notes", "stage_d_skipped_chords",
    "stage_d_missing_durations", "stage_d_malformed_spans",
    "stage_d_unknown_tokens", "stage_d_fallback_rests",
    "stage_d_raised_count", "stage_d_first_error",
    "score_failure_reason",
]


def _read_stage_d_diag(pred_path: Path) -> tuple:
    """Return the 8 Stage-D diagnostic CSV values for *pred_path*."""
    diag_path = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
    try:
        raw = json.loads(diag_path.read_text(encoding="utf-8"))
        raised = raw.get("raised_during_part_append", [])
        first_error = raised[0].get("error_message", "") if raised else ""
        return (
            raw.get("skipped_notes"),
            raw.get("skipped_chords"),
            raw.get("missing_durations"),
            raw.get("malformed_spans"),
            raw.get("unknown_tokens"),
            raw.get("fallback_rests"),
            len(raised),
            first_error,
        )
    except Exception:
        return (None, None, None, None, None, None, None, None)


def _run_subprocess(
    pred_path: Path,
    ref_path: Path,
    metrics: list[str],
    timeout: int,
) -> dict:
    """Run eval._score_one_piece in a subprocess and return the parsed JSON dict.

    On success: returns dict with metric keys populated.
    On failure (timeout, crash, bad JSON): returns dict with 'error' key set to
    a short description string.
    """
    cmd = [
        str(VENV_PYTHON), "-m", "eval._score_one_piece",
        "--pred", str(pred_path),
        "--ref", str(ref_path),
        "--metrics", ",".join(metrics),
    ]
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(repo_root),
        )
        if result.returncode != 0:
            stderr_snippet = (result.stderr or "")[-500:].strip()
            return {"error": f"subprocess exit {result.returncode}: {stderr_snippet}"}
        # Last non-empty line of stdout should be the JSON payload
        lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not lines:
            return {"error": "subprocess produced no output"}
        payload = json.loads(lines[-1])
        return payload
    except subprocess.TimeoutExpired:
        return {"error": f"subprocess timeout after {timeout}s"}
    except json.JSONDecodeError as e:
        return {"error": f"JSON decode error: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def score_piece_subprocess(
    stem: str,
    pred_path: Path,
    ref_path: Path,
    metrics: list[str],
) -> dict:
    """Score one piece, splitting cheap and tedn metrics into separate subprocesses.

    When *metrics* contains both tedn and at least one cheap metric
    (onset_f1 / linearized_ser), two subprocess calls are made:

      1. Cheap pair (onset_f1 + linearized_ser present in *metrics*) with a
         60 s timeout.
      2. Tedn-only with a 300 s timeout.

    The results are merged into a single dict.  Partial failures are surfaced in
    the 'error' key with a prefix indicating which group failed, e.g.
    "tedn: subprocess timeout after 300s".  If both groups fail the reasons are
    joined with " | ".

    When *metrics* is a strict subset of one group (e.g. only tedn, or only
    onset_f1), a single subprocess call is used with the appropriate timeout.

    Returns a dict with metric-value keys plus optionally an 'error' key
    describing any failure(s).
    """
    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]

    # Decide whether to split into two calls
    need_split = bool(cheap_requested) and bool(tedn_requested)

    if not need_split:
        # Single call -- use appropriate timeout
        timeout = CHEAP_TIMEOUT_SEC if not tedn_requested else TEDN_TIMEOUT_SEC
        return _run_subprocess(pred_path, ref_path, metrics, timeout)

    # --- Split path: two subprocess calls ---
    combined: dict = {}
    failure_parts: list[str] = []

    # 1. Cheap pair
    cheap_result = _run_subprocess(pred_path, ref_path, cheap_requested, CHEAP_TIMEOUT_SEC)
    if "error" in cheap_result:
        failure_parts.append(f"cheap-pair: {cheap_result['error']}")
    else:
        combined.update({k: v for k, v in cheap_result.items() if k != "error"})

    # 2. Tedn-only
    tedn_result = _run_subprocess(pred_path, ref_path, tedn_requested, TEDN_TIMEOUT_SEC)
    if "error" in tedn_result:
        failure_parts.append(f"tedn: {tedn_result['error']}")
    else:
        combined.update({k: v for k, v in tedn_result.items() if k != "error"})

    if failure_parts:
        combined["error"] = " | ".join(failure_parts)

    return combined


def main() -> None:
    p = argparse.ArgumentParser(
        description="Score predicted MusicXMLs against reference MXLs, "
                    "subprocess-isolating per-piece metric computation."
    )
    p.add_argument(
        "--predictions-dir", type=Path, required=True,
        help="Directory containing predicted .musicxml files (output of run_clarity_demo_eval.py)",
    )
    p.add_argument(
        "--reference-dir", type=Path, required=True,
        help="Directory containing reference .mxl files",
    )
    p.add_argument(
        "--name", required=True,
        help="Run name (used for output CSV filename: eval/results/clarity_demo_<name>.csv)",
    )
    p.add_argument(
        "--metrics",
        default="tedn,linearized_ser,onset_f1",
        help="Comma-separated list of metrics to compute (default: tedn,linearized_ser,onset_f1)",
    )
    args = p.parse_args()

    if not args.predictions_dir.exists():
        raise SystemExit(f"FATAL: predictions-dir not found: {args.predictions_dir}")
    if not args.reference_dir.exists():
        raise SystemExit(f"FATAL: reference-dir not found: {args.reference_dir}")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    valid_metrics = {"tedn", "linearized_ser", "onset_f1"}
    unknown = set(metrics) - valid_metrics
    if unknown:
        raise SystemExit(f"FATAL: unknown metrics: {unknown}. Valid: {valid_metrics}")

    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]
    splitting = bool(cheap_requested) and bool(tedn_requested)

    print(f"Run name:        {args.name}")
    print(f"Predictions dir: {args.predictions_dir}")
    print(f"Reference dir:   {args.reference_dir}")
    print(f"Metrics:         {metrics}")
    print(f"Pieces:          {len(DEMO_STEMS)}")
    if splitting:
        print(f"Scoring mode:    split (cheap={cheap_requested} @{CHEAP_TIMEOUT_SEC}s,"
              f" tedn @{TEDN_TIMEOUT_SEC}s)")
    else:
        timeout = CHEAP_TIMEOUT_SEC if not tedn_requested else TEDN_TIMEOUT_SEC
        print(f"Scoring mode:    single subprocess @{timeout}s")
    print()

    repo_root = Path(__file__).resolve().parents[1]
    n_total = len(DEMO_STEMS)
    rows: list[tuple] = []

    for i, stem in enumerate(DEMO_STEMS, 1):
        pred = args.predictions_dir / f"{stem}.musicxml"
        ref = args.reference_dir / f"{stem}.mxl"

        if not pred.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: predicted XML not found at {pred}")
            rows.append((stem, None, None, None) + (None,) * 8 + ("predicted_xml_missing",))
            continue
        if not ref.exists():
            print(f"[{i}/{n_total}] SKIP {stem}: reference MXL not found at {ref}")
            rows.append((stem, None, None, None) + (None,) * 8 + ("reference_mxl_missing",))
            continue

        print(f"[{i}/{n_total}] scoring {stem} ...")

        # Stage D diagnostics are read in-process (fast JSON read, no music21)
        stage_d_cols = _read_stage_d_diag(pred)

        # Metric scoring runs in subprocess(es) -- OS reclaims all music21/zss memory
        # when the child exits, preventing the 86 GB committed-memory OOM seen in v1.
        # cheap metrics and tedn are split into separate calls so a tedn timeout
        # does not discard already-computed onset_f1 / linearized_ser values.
        payload = score_piece_subprocess(stem, pred, ref, metrics)

        failure_reason = payload.get("error", None)
        f1 = payload.get("onset_f1") if "onset_f1" in metrics else None
        tedn = payload.get("tedn") if "tedn" in metrics else None
        lin_ser = payload.get("linearized_ser") if "linearized_ser" in metrics else None

        rows.append((stem, f1, tedn, lin_ser) + stage_d_cols + (failure_reason,))

        tedn_str = f"{tedn:.4f}" if tedn is not None else "N/A"
        lin_str = f"{lin_ser:.4f}" if lin_ser is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        if failure_reason:
            # Partial success: some metrics may still be populated
            print(f"[{i}/{n_total}] PARTIAL/FAIL {stem}: {failure_reason}")
            print(f"             onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}")
        else:
            print(f"[{i}/{n_total}] {stem}: onset_f1={f1_str}  tedn={tedn_str}  lin_ser={lin_str}")

    csv_path = (repo_root / "eval/results" / f"clarity_demo_{args.name}.csv").resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_HEADER)
        w.writerows(rows)
    print(f"\nResults written to {csv_path}")

    valid = [row[1] for row in rows if row[1] is not None]
    failed_count = sum(1 for row in rows if row[1] is None)
    if not valid:
        print(f"\nNo pieces scored successfully ({failed_count}/{n_total} failed/skipped).")
        return

    mean_f1 = statistics.mean(valid)
    med_f1 = statistics.median(valid)
    min_f1 = min(valid)
    max_f1 = max(valid)
    print(f"\n=== Clarity Demo Scoring Results ({args.name}) ===")
    print(f"Pieces evaluated: {len(valid)} / {n_total} (failed/skipped: {failed_count})")
    print(f"Mean onset-F1:   {mean_f1:.4f}")
    print(f"Median onset-F1: {med_f1:.4f}")
    print(f"Min onset-F1:    {min_f1:.4f}")
    print(f"Max onset-F1:    {max_f1:.4f}")


if __name__ == "__main__":
    main()
