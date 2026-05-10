"""Shared subprocess-isolated scoring infrastructure for both demo and lieder evals.

Imported by eval.score_demo_eval and eval.score_lieder_eval.  Contains:

  - CSV_HEADER              — canonical column list (piece + metrics + stage_d + failure)
  - _read_stage_d_diag()   — reads <pred>.diagnostics.json sidecar, returns 9-tuple
  - _run_subprocess()      — invokes eval._score_one_piece, returns parsed JSON dict
  - score_metric_group_subprocess() — run ONE metric group in a fresh subprocess
  - score_piece_subprocess() — thin sequential wrapper over score_metric_group_subprocess
  - _build_reference_index() — builds a dict[stem, Path] from a reference directory
  - _resolve_venv_python()   — cross-platform venv Python path detection

The split design is:
  1. Cheap pair (onset_f1 + linearized_ser): CHEAP_TIMEOUT_SEC (60 s)
  2. Tedn-only: TEDN_TIMEOUT_SEC (300 s)

When both groups are requested, two child processes are launched so that a tedn
timeout or OOM does not discard already-computed onset_f1 / linearized_ser.
"""
import json
import subprocess
import sys
import time
import threading
from pathlib import Path
from typing import Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

# Path to our venv's Python -- same one that ran inference.
# score_demo_eval uses venv-cu132; lieder eval uses venv.
# Callers can override by passing venv_python explicitly.
_DEFAULT_VENV_PYTHON = Path(__file__).resolve().parents[1] / "venv-cu132" / "Scripts" / "python.exe"

# Timeout for the cheap-pair subprocess (onset_f1 + linearized_ser)
CHEAP_TIMEOUT_SEC = 60

# Timeout for tedn (may OOM/hang on very large scores like Clair de Lune)
TEDN_TIMEOUT_SEC = 300

# Metrics considered "cheap" -- fast and low-memory
CHEAP_METRICS = frozenset({"onset_f1", "linearized_ser"})

# Canonical CSV header shared by demo and lieder CSVs
CSV_HEADER = [
    "piece", "onset_f1", "tedn", "linearized_ser",
    "stage_d_skipped_notes", "stage_d_skipped_chords",
    "stage_d_missing_durations", "stage_d_malformed_spans",
    "stage_d_unknown_tokens", "stage_d_fallback_rests",
    "stage_d_raised_count", "stage_d_first_error",
    "stage_d_skipped_systems",
    "score_failure_reason",
]


def _read_stage_d_diag(pred_path: Path) -> tuple:
    """Return the 9 Stage-D diagnostic CSV values for *pred_path*.

    Looks for <pred_path>.diagnostics.json alongside the MusicXML output.
    Returns a tuple of 9 values (all None if the sidecar is absent or unreadable).
    Logs a warning to stderr when parsing fails (but still returns all-None so
    the row pipeline is not interrupted).

    The 9 values, in CSV column order:
        skipped_notes, skipped_chords, missing_durations, malformed_spans,
        unknown_tokens, fallback_rests, raised_count, first_error, skipped_systems_count
    """
    diag_path = pred_path.with_suffix(pred_path.suffix + ".diagnostics.json")
    try:
        raw = json.loads(diag_path.read_text(encoding="utf-8"))
        raised = raw.get("raised_during_part_append", [])
        first_error = raised[0].get("error_message", "") if raised else ""
        skipped_systems_count = int(raw.get("skipped_systems") or 0)
        return (
            raw.get("skipped_notes"),
            raw.get("skipped_chords"),
            raw.get("missing_durations"),
            raw.get("malformed_spans"),
            raw.get("unknown_tokens"),
            raw.get("fallback_rests"),
            len(raised),
            first_error,
            skipped_systems_count,
        )
    except Exception as exc:
        print(f"[warn] Stage D sidecar parse failed for {pred_path}: {exc}", file=sys.stderr)
        return (None, None, None, None, None, None, None, None, None)


def _build_reference_index(reference_dir: Path) -> "dict[str, Path]":
    """Build a stem -> Path index from all .mxl files under *reference_dir*.

    Scans the directory tree once at startup. Raises SystemExit with a clear
    error message if any duplicate stems are detected — silently choosing the
    wrong reference would produce incorrect evaluation results, which is worse
    than a slower run.

    Returns:
        dict mapping each stem (filename without extension) to its Path.
    """
    refs: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}

    for path in reference_dir.rglob("*.mxl"):
        stem = path.stem
        if stem in refs:
            # Track all duplicates before failing so the error is actionable.
            if stem not in duplicates:
                duplicates[stem] = [refs[stem]]
            duplicates[stem].append(path)
        else:
            refs[stem] = path

    if duplicates:
        lines = ["FATAL: duplicate reference stems detected (ambiguous reference lookup):"]
        for stem, paths in sorted(duplicates.items()):
            for p in paths:
                lines.append(f"  {stem}: {p}")
        raise SystemExit("\n".join(lines))

    return refs


def _resolve_venv_python(explicit: Optional[Path] = None) -> Path:
    """Resolve the Python interpreter to use for scoring subprocesses.

    Fallback chain (first match wins):
      1. *explicit* — the --python CLI argument if provided.
      2. sys.executable — if it can `import eval` (i.e. repo is on its path).
      3. repo_root / "venv-cu132/Scripts/python.exe"  (Windows production)
      4. repo_root / "venv/Scripts/python.exe"         (Windows fallback)
      5. repo_root / "venv-cu132/bin/python"           (Linux/macOS production)
      6. repo_root / "venv/bin/python"                 (Linux/macOS fallback)
      7. Raises SystemExit with a clear message listing all paths checked.
    """
    checked: list[str] = []

    # 1. Explicit argument
    if explicit is not None:
        if explicit.exists():
            return explicit
        checked.append(f"--python arg: {explicit} (not found)")

    # 2. sys.executable if it can import eval
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import eval"],
            capture_output=True,
            cwd=str(_REPO_ROOT),
            timeout=10,
        )
        if result.returncode == 0:
            return Path(sys.executable)
        checked.append(f"sys.executable ({sys.executable}): cannot import eval")
    except Exception as e:
        checked.append(f"sys.executable ({sys.executable}): {e}")

    # 3–6. Repo-local venvs
    candidates = [
        _REPO_ROOT / "venv-cu132" / "Scripts" / "python.exe",
        _REPO_ROOT / "venv" / "Scripts" / "python.exe",
        _REPO_ROOT / "venv-cu132" / "bin" / "python",
        _REPO_ROOT / "venv" / "bin" / "python",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
        checked.append(str(candidate))

    raise SystemExit(
        "FATAL: Could not locate a Python interpreter for scoring subprocesses.\n"
        "Paths checked:\n" + "\n".join(f"  {p}" for p in checked) + "\n"
        "Pass --python /path/to/python explicitly."
    )


def _run_subprocess(
    pred_path: Path,
    ref_path: Path,
    metrics: list[str],
    timeout: int,
    *,
    venv_python: "Path | None" = None,
    stderr_log: "Path | None" = None,
    child_memory_limit_gb: float = 0.0,
) -> dict:
    """Run eval._score_one_piece in a subprocess and return the parsed JSON dict.

    On success: returns dict with metric keys populated.
    On failure (timeout, crash, bad JSON): returns dict with 'error' key set to
    a short description string.

    Args:
        pred_path: Path to the predicted MusicXML file.
        ref_path: Path to the reference MXL file.
        metrics: List of metric names to compute.
        timeout: Subprocess timeout in seconds.
        venv_python: Optional explicit Python interpreter path.
        stderr_log: If set, stderr is redirected to this file path. On non-zero
            return code, the last 8 KB of this file is read into the 'error' field.
            If None, stderr is captured in-memory (legacy behavior).
        child_memory_limit_gb: Linux-only: set RLIMIT_AS on the child process.
            0 (default) disables the cap. Has no effect on non-Linux platforms.
            CAVEAT: RLIMIT_AS caps virtual address space, not RSS. Some Python
            and scientific libraries reserve more virtual memory than they actively
            use, so this may need tuning. If it causes false failures, raise it
            to 24 GB or disable with 0.
    """
    python = venv_python or _DEFAULT_VENV_PYTHON
    cmd = [
        str(python), "-m", "eval._score_one_piece",
        "--pred", str(pred_path),
        "--ref", str(ref_path),
        "--metrics", ",".join(metrics),
    ]
    repo_root = Path(__file__).resolve().parents[1]

    # Build preexec_fn for Linux memory cap
    preexec_fn = None
    if child_memory_limit_gb > 0 and sys.platform == "linux":
        _cap_gb = child_memory_limit_gb

        def _set_memory_limit():
            import resource
            max_bytes = int(_cap_gb * 1024 ** 3)
            resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

        preexec_fn = _set_memory_limit

    try:
        if stderr_log is not None:
            stderr_log.parent.mkdir(parents=True, exist_ok=True)
            with stderr_log.open("w", encoding="utf-8") as err_fh:
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=err_fh,
                    text=True,
                    timeout=timeout,
                    cwd=str(repo_root),
                    preexec_fn=preexec_fn,
                )
            if result.returncode != 0:
                # Read last 8 KB of stderr log for error message
                try:
                    raw = stderr_log.read_bytes()
                    snippet = raw[-8192:].decode("utf-8", errors="replace").strip()
                except Exception:
                    snippet = "(could not read stderr log)"
                return {"error": f"subprocess exit {result.returncode}: {snippet}"}
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(repo_root),
                preexec_fn=preexec_fn,
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


def score_metric_group_subprocess(
    pred_path: Path,
    ref_path: Path,
    metrics: list[str],
    timeout: int,
    *,
    venv_python: "Path | None" = None,
    stderr_log: "Path | None" = None,
    child_memory_limit_gb: float = 0.0,
    instrumentation_log: "Path | None" = None,
    stem: "str | None" = None,
    group_name: "str | None" = None,
) -> dict:
    """Run ONE metric group in a fresh subprocess.

    This is the lower-level dispatch primitive used by the two-lane scheduler.
    Callers pass a specific group (e.g. ['onset_f1', 'linearized_ser'] or ['tedn'])
    rather than the full metric list.

    Args:
        pred_path: Path to the predicted MusicXML file.
        ref_path: Path to the reference MXL file.
        metrics: Metric names for this group (e.g. ['onset_f1', 'linearized_ser'] or ['tedn']).
        timeout: Subprocess timeout in seconds.
        venv_python: Optional explicit Python interpreter path.
        stderr_log: If set, stderr is redirected to this file. On failure, last 8 KB read.
        child_memory_limit_gb: Linux-only RLIMIT_AS cap in GB. 0 = disabled.
        instrumentation_log: If set, a JSONL line is appended with timing + return code.
        stem: Piece stem name (for instrumentation log). Inferred from pred_path if None.
        group_name: Group label for instrumentation log (e.g. 'cheap' or 'tedn').

    Returns:
        dict with metric-value keys, or {'error': '...'} on failure.
    """
    _stem = stem or pred_path.stem
    _group = group_name or ("tedn" if metrics == ["tedn"] else "cheap")
    start_time = time.monotonic()

    result = _run_subprocess(
        pred_path,
        ref_path,
        metrics,
        timeout,
        venv_python=venv_python,
        stderr_log=stderr_log,
        child_memory_limit_gb=child_memory_limit_gb,
    )

    end_time = time.monotonic()
    duration = end_time - start_time
    is_timeout = "error" in result and "timeout" in result.get("error", "")
    return_code = 0 if "error" not in result else (
        -1 if is_timeout else 1
    )
    stderr_size = 0
    if stderr_log is not None and stderr_log.exists():
        try:
            stderr_size = stderr_log.stat().st_size
        except Exception:
            stderr_size = 0

    if instrumentation_log is not None:
        try:
            instrumentation_log.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "piece": _stem,
                "group": _group,
                "start": start_time,
                "end": end_time,
                "duration_sec": round(duration, 3),
                "return_code": return_code,
                "timeout": is_timeout,
                "stderr_size_bytes": stderr_size,
            }
            with instrumentation_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception:
            pass  # Instrumentation failures must not abort scoring

    return result


def score_piece_subprocess(
    pred_path: Path,
    ref_path: Path,
    metrics: list[str],
    *,
    venv_python: "Path | None" = None,
    cheap_timeout: int = CHEAP_TIMEOUT_SEC,
    tedn_timeout: int = TEDN_TIMEOUT_SEC,
    parallel_metric_groups: bool = False,
    stderr_dir: "Path | None" = None,
    child_memory_limit_gb: float = 0.0,
    instrumentation_log: "Path | None" = None,
) -> dict:
    """Score one piece, splitting cheap and tedn metrics into separate subprocesses.

    Thin sequential wrapper around score_metric_group_subprocess. Preserved for
    backward compatibility with callers that import it directly.

    When *metrics* contains both tedn and at least one cheap metric
    (onset_f1 / linearized_ser), two subprocess calls are made:

      1. Cheap pair (onset_f1 + linearized_ser present in *metrics*) with
         *cheap_timeout* (default 60 s).
      2. Tedn-only with *tedn_timeout* (default 300 s).

    The results are merged into a single dict.  Partial failures are surfaced in
    the 'error' key with a prefix indicating which group failed, e.g.
    "tedn: subprocess timeout after 300s".  If both groups fail the reasons are
    joined with " | ".

    When *metrics* is a strict subset of one group (e.g. only tedn, or only
    onset_f1), a single subprocess call is used with the appropriate timeout.

    Args:
        pred_path: Path to the predicted MusicXML file.
        ref_path: Path to the reference MXL file.
        metrics: List of metric names to compute.
        venv_python: Optional explicit Python interpreter path.
        cheap_timeout: Timeout in seconds for the cheap-pair subprocess.
        tedn_timeout: Timeout in seconds for the tedn subprocess.
        parallel_metric_groups: When True AND both cheap and tedn metrics are
            requested, run the two subprocesses concurrently using
            ThreadPoolExecutor(max_workers=2). This can reduce per-piece wall
            time by ~cheap_timeout seconds when cheap metrics finish quickly.

            MEMORY TRADEOFF: With parallel_metric_groups=True, both the cheap
            and tedn subprocesses are alive simultaneously within one piece,
            roughly doubling per-piece peak memory from ~2.5 GB to ~5 GB.
            When combined with --jobs N, total RAM is approximately:
              parent_ram + jobs * 2 * per_subprocess_peak_ram
            Do not enable unless memory headroom is confirmed. Default is False.
        stderr_dir: If set, per-group stderr is redirected to files under this
            directory: <stem>_cheap.stderr.log and <stem>_tedn.stderr.log.
        child_memory_limit_gb: Linux-only RLIMIT_AS cap in GB. 0 = disabled.
        instrumentation_log: If set, timing records are appended as JSONL.

    Returns a dict with metric-value keys plus optionally an 'error' key
    describing any failure(s).
    """
    cheap_requested = [m for m in metrics if m in CHEAP_METRICS]
    tedn_requested = [m for m in metrics if m == "tedn"]

    # Decide whether to split into two calls
    need_split = bool(cheap_requested) and bool(tedn_requested)

    stem = pred_path.stem

    def _cheap_stderr_log():
        if stderr_dir is not None:
            return stderr_dir / f"{stem}_cheap.stderr.log"
        return None

    def _tedn_stderr_log():
        if stderr_dir is not None:
            return stderr_dir / f"{stem}_tedn.stderr.log"
        return None

    if not need_split:
        # Single call -- use appropriate timeout
        timeout = cheap_timeout if not tedn_requested else tedn_timeout
        g_name = "tedn" if tedn_requested else "cheap"
        sl = _tedn_stderr_log() if tedn_requested else _cheap_stderr_log()
        return score_metric_group_subprocess(
            pred_path, ref_path, metrics, timeout,
            venv_python=venv_python,
            stderr_log=sl,
            child_memory_limit_gb=child_memory_limit_gb,
            instrumentation_log=instrumentation_log,
            stem=stem,
            group_name=g_name,
        )

    # --- Split path: two subprocess calls ---
    if parallel_metric_groups:
        # Run cheap and tedn subprocesses concurrently within this piece.
        # See docstring for memory tradeoff discussion.
        from concurrent.futures import ThreadPoolExecutor, as_completed

        combined: dict = {}
        failure_parts: list[str] = []

        def _run_cheap():
            return "cheap", score_metric_group_subprocess(
                pred_path, ref_path, cheap_requested, cheap_timeout,
                venv_python=venv_python,
                stderr_log=_cheap_stderr_log(),
                child_memory_limit_gb=child_memory_limit_gb,
                instrumentation_log=instrumentation_log,
                stem=stem,
                group_name="cheap",
            )

        def _run_tedn():
            return "tedn", score_metric_group_subprocess(
                pred_path, ref_path, tedn_requested, tedn_timeout,
                venv_python=venv_python,
                stderr_log=_tedn_stderr_log(),
                child_memory_limit_gb=child_memory_limit_gb,
                instrumentation_log=instrumentation_log,
                stem=stem,
                group_name="tedn",
            )

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(_run_cheap), pool.submit(_run_tedn)]
            for fut in as_completed(futures):
                group, result = fut.result()
                if "error" in result:
                    failure_parts.append(f"{group}-pair: {result['error']}" if group == "cheap" else f"tedn: {result['error']}")
                else:
                    combined.update({k: v for k, v in result.items() if k != "error"})

        if failure_parts:
            combined["error"] = " | ".join(failure_parts)
        return combined

    # Sequential split (default)
    combined_seq: dict = {}
    failure_parts_seq: list[str] = []

    # 1. Cheap pair
    cheap_result = score_metric_group_subprocess(
        pred_path, ref_path, cheap_requested, cheap_timeout,
        venv_python=venv_python,
        stderr_log=_cheap_stderr_log(),
        child_memory_limit_gb=child_memory_limit_gb,
        instrumentation_log=instrumentation_log,
        stem=stem,
        group_name="cheap",
    )
    if "error" in cheap_result:
        failure_parts_seq.append(f"cheap-pair: {cheap_result['error']}")
    else:
        combined_seq.update({k: v for k, v in cheap_result.items() if k != "error"})

    # 2. Tedn-only
    tedn_result = score_metric_group_subprocess(
        pred_path, ref_path, tedn_requested, tedn_timeout,
        venv_python=venv_python,
        stderr_log=_tedn_stderr_log(),
        child_memory_limit_gb=child_memory_limit_gb,
        instrumentation_log=instrumentation_log,
        stem=stem,
        group_name="tedn",
    )
    if "error" in tedn_result:
        failure_parts_seq.append(f"tedn: {tedn_result['error']}")
    else:
        combined_seq.update({k: v for k, v in tedn_result.items() if k != "error"})

    if failure_parts_seq:
        combined_seq["error"] = " | ".join(failure_parts_seq)

    return combined_seq
