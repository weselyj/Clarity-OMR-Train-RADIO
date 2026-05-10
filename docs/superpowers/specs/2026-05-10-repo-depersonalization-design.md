# Repo De-personalization and External-User Readiness — Design

Date: 2026-05-10
Status: Approved (brainstorming complete; awaiting plan)
Source review: `/home/ari/docs/clarity_omr_train_radio_repo_gap_review_2026-05-10.md` (codex, 2026-05-10)

## Context

Clarity-OMR-Train-RADIO has been developed as a single-author scratchpad on a homelab Windows GPU box (`10.10.1.29`) with intermittent local Linux work for unit tests. The repo currently mixes canonical project artifacts (training/inference code, design specs, training data preparation) with personal artifacts (one-off PowerShell launchers, hardcoded `/home/ari` and `C:\Users\Jonathan Wesely\` paths, session-state handoffs, results CSVs from particular runs, top-level `NOTES.md`, training-step logs in `src/train/`).

A 2026-05-10 codex gap review surfaced six findings. Three are code-level (eval defaults, decode-step alignment, missing scoring column); three are doc-level (CUDA-only environment is implicit not explicit, plan command snippets are stale, two `superpowers/` doc trees are out of sync).

This spec turns the repo into something a third party can clone, install on a CUDA-capable machine, and use — without first reverse-engineering the author's environment.

## Goals

1. Codex's six findings are resolved.
2. The repo declares "CUDA-capable GPU required" prominently and consistently across README, `requirements.txt`, install docs, and test gating.
3. No path on the canonical surface (README, `docs/` non-archive, `scripts/` non-archive, `src/`) references `/home/ari`, `C:\Users\Jonathan Wesely\`, `seder`, `10.10.1.29`, or "the user". Personal artifacts move to `archive/` (in-tree, preserved for history).
4. README is restructured into a thin landing page plus a documentation set under `docs/` (QUICKSTART, HARDWARE, INSTALL, TRAINING, EVALUATION, ARCHITECTURE, paths). The dense architecture content is preserved but relocated.
5. Eval results data (CSVs, JSONLs, training-step logs) is removed from the canonical tree; a `docs/RESULTS.md` stub holds the place for a future Hugging Face release.
6. The two `superpowers/` doc trees are reconciled: in-repo `docs/superpowers/` is canonical; the external `/home/ari/docs/superpowers/` gets a stub README pointing to canonical (sync of recent material from repo→external is already complete).

## Non-Goals

1. Actual Hugging Face upload (model weights, model card, scoring bundle). `docs/RESULTS.md` is a stub link target; the upload itself is a follow-up.
2. Backfilling the archived `lieder_subproject4_scores.csv` with the new `stage_d_skipped_systems` column. Code change lands; backfill requires a GPU-box rerun and is a follow-up.
3. CI for the test suite. Test gating in this spec makes local failures informative; CI is separate.
4. Pruning or restructuring the external `/home/ari/docs/superpowers/` tree's older content. That tree is the author's personal scratchpad; only the repo→external sync (done) is in scope.
5. Substantive content changes to retained docs (`docs/perf/`, `docs/kern_converter_limitations.md`, `docs/stage_a_brace_margin_known_gap.md`, `docs/omr_layout_analysis_audit.md`). De-personalization grep edits only.
6. Reorganizing or rewriting `src/` source code outside the three targeted code fixes (#2, #3, #6 from codex). General refactor is out of scope.

## Codex Findings — Resolution Map

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | High | Tests fail import-time on local Linux/CPU env; `docs/locations.md` claims they work locally | Strict CUDA-only stance. Drop the local-tests-work claim. Add `tests/conftest.py` that auto-skips CUDA-required test directories with a clear "CUDA required" reason. README + `docs/HARDWARE.md` declare CUDA-capable GPU required. |
| 2 | High | `eval/run_lieder_eval.py --stage-a-weights` defaults to sibling-repo path `~/Clarity-OMR/info/yolo.pt`; risks silent wrong-YOLO usage | Default to `runs/detect/runs/yolo26m_systems/weights/best.pt`. Validate file exists at startup; fail with actionable message if missing. |
| 3 | High | `eval/run_lieder_eval.py --max-decode-steps` defaults to 256; pipeline default is 2048; help text says "per staff" | Default to 2048. Help text → "per system crop". |
| 4 | Medium | Subproject 4 plan command snippets reference flags that do not exist (`--stage-a-yolo`, `--predictions-dir`, `--status-jsonl`) | Update snippets in `docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md` to use `--stage-a-weights`, `--output-dir`, and the actual auto-generated status path. |
| 5 | Medium | In-repo and external `superpowers/` trees out of sync | Repo→external sync of recent material done during brainstorming. External tree gets a stub `README.md` declaring repo-canonical. |
| 6 | Medium | `skipped_systems` recorded in StageDExportDiagnostics sidecar but absent from scoring CSV header/parse path | Add `stage_d_skipped_systems` to `eval/_scoring_utils.py CSV_HEADER` and `_read_stage_d_diag` parse logic. Default to empty/0 when sidecar missing. |

## Design

### 1. End-State Tree

```
README.md                          # slim landing — what it is, hardware requirement, doc index
LICENSE
requirements.txt                   # CUDA-only; comments rewritten
.gitignore                         # add eval/results/, runs/, checkpoints/, src/train/training_steps*.jsonl

docs/
  QUICKSTART.md                    # NEW — clone → install → smoke inference → smoke training
  HARDWARE.md                      # NEW — required GPUs, VRAM, OS support, what's tested
  INSTALL.md                       # NEW — Linux cu132 + Windows cu132 paths
  TRAINING.md                      # rename of TRAINING_COMMANDS.md; merges TRAINING_COMMANDS_UBUNTU.md
  EVALUATION.md                    # NEW — running lieder eval and scoring it
  ARCHITECTURE.md                  # NEW — relocated dense architecture content from README
  RESULTS.md                       # NEW — stub linking to (future) HuggingFace release
  paths.md                         # NEW — repo-relative canonical paths (replaces locations.md)
  kern_converter_limitations.md    # KEEP (de-personalize grep only)
  stage_a_brace_margin_known_gap.md # KEEP (de-personalize grep only)
  omr_layout_analysis_audit.md     # KEEP (de-personalize grep only)
  perf/                            # KEEP (project history; de-personalize grep only)
  superpowers/
    specs/                         # KEEP — canonical project history
    plans/                         # KEEP
    audits/                        # KEEP
    handoffs/                      # EMPTIED — all entries archived (see Section 4)

archive/
  per_staff/                       # EXISTING — untouched
  scripts/                         # NEW — personal PowerShell launchers, perf benchmarks, homelab backup scripts
  handoffs/                        # NEW — all of docs/superpowers/handoffs/* moved here
  notes/                           # NEW — top-level NOTES.md and any one-off scratch
  results/                         # NEW — eval CSVs/JSONs, baseline_pre_rebuild/, training-step logs
  manifests/                       # NEW — synthetic_systems_eval_fresh.jsonl and similar personal-run-specific manifests

scripts/                           # canonical entrypoints only
  setup_venv_cu132.ps1             # KEEP, generalize $repo derivation if needed
  setup_venv_cu132.sh              # NEW — Linux equivalent
  cu132_venv_sitecustomize.py      # KEEP — referenced by setup_venv_cu132.ps1
  train_yolo.py                    # KEEP
  build_*.py                       # KEEP — data prep entrypoints
  derive_*.py                      # KEEP
  audit_*.py                       # KEEP
  rederive_*.py                    # KEEP
  retokenize_with_staff_markers.py # KEEP
  build_encoder_cache.py           # KEEP
  check_encoder_resume.py          # KEEP
  compare_step_logs.py             # READ-VERIFY: archive if it references specific run names/paths; keep if generic
  smoketest_bracket_detector.py    # KEEP
  visualize_audiolabs_systems.py   # KEEP
  enumerate_radio_modules.py       # READ-VERIFY: archive if author-debug; keep if generic enumeration entrypoint
  debug_burleigh_synthetic.py      # archive (personal debug script — name carries the "Burleigh" debugging context)
  breakdown_audit.py               # READ-VERIFY: archive if specific to a particular audit run
  convert_omr_layout.py            # KEEP
  build_sparse_augment_manifest.py # KEEP

eval/results/                      # recreated empty with .gitkeep + README pointing to docs/RESULTS.md
src/train/training_steps*.jsonl    # → archive/results/
src/data/manifests/synthetic_systems_eval_fresh.jsonl  # → archive/manifests/

NOTES.md                           # → archive/notes/
analyze_data.py                    # READ-VERIFY: likely → archive/scripts/ (top-level placement suggests scratch)
logs/                              # READ-VERIFY: likely → archive/notes/logs/ if it contains personal log captures
```

### 2. README Rewrite

Slim README structure (~120 lines, link-out heavy):

1. Title + one-paragraph project description.
2. **Hardware requirement banner**: "This project requires a CUDA-capable GPU. CPU-only execution is not supported. See [docs/HARDWARE.md](docs/HARDWARE.md)."
3. What it does — the 3-stage architecture diagram (already impersonal, kept verbatim).
4. System-level vs per-staff — short version (~5 lines); detail moves to `docs/ARCHITECTURE.md`.
5. Project status table — updated:
   - Subproject 4 (per-system inference + lieder eval) row added: shipped 2026-05-10.
   - Stage 3 row updated: training complete (val_loss 0.148 at step 4000); the "in flight on `feat/stage3-encoder-cache`" line is stale and removed.
6. Quickstart pointer — "See [docs/QUICKSTART.md](docs/QUICKSTART.md)."
7. Documentation index — terse table of links to QUICKSTART / HARDWARE / INSTALL / TRAINING / EVALUATION / ARCHITECTURE / paths / RESULTS.
8. Repository structure — top-level dirs only; per-file breakdown moves to `docs/ARCHITECTURE.md`.
9. License.

Content relocation map:

| Current README section | New home |
|---|---|
| Stage A training data table + label derivation pipeline | `docs/ARCHITECTURE.md` |
| Stage B encoder/decoder/positional bridge/DoRA detail | `docs/ARCHITECTURE.md` |
| Token vocabulary table + encoding example | `docs/ARCHITECTURE.md` |
| Grammar FSA constraints | `docs/ARCHITECTURE.md` |
| Loss function / training stability / data augmentation | `docs/ARCHITECTURE.md` |
| Evaluation metrics + decision gates | `docs/EVALUATION.md` |
| Architecture rationale (selected) | `docs/ARCHITECTURE.md` |
| References | `docs/ARCHITECTURE.md` (References section) |
| Installation block | `docs/INSTALL.md` |
| Data preparation steps | `docs/QUICKSTART.md` (smoke version) — full retrain steps live in `docs/TRAINING.md` |
| Training commands (Stage A + Stage B) | `docs/TRAINING.md` (merges existing TRAINING_COMMANDS*.md) |

De-personalization edits in retained content:

- README line 495 ("(in user repo)") → drop; reference becomes the in-repo spec path.
- All `/home/ari/...`, `C:\Users\Jonathan Wesely\...` → repo-relative.
- "PR #39, merged 2026-05-08" — fine to keep (project history, not personal).

### 3. Code Fixes

**#2 — `eval/run_lieder_eval.py` Stage A default**

```python
# old
parser.add_argument("--stage-a-weights", default=os.path.expanduser("~/Clarity-OMR/info/yolo.pt"))

# new
DEFAULT_STAGE_A_WEIGHTS = "runs/detect/runs/yolo26m_systems/weights/best.pt"
parser.add_argument("--stage-a-weights", default=DEFAULT_STAGE_A_WEIGHTS,
                    help="Path to Stage A YOLO weights (system-level detector).")

# in main(), after parsing:
if not Path(args.stage_a_weights).is_file():
    parser.error(
        f"Stage A weights not found at {args.stage_a_weights}. "
        f"Train Stage A first (see docs/TRAINING.md) or pass --stage-a-weights."
    )
```

**#3 — decode-step alignment**

```python
# old
parser.add_argument("--max-decode-steps", type=int, default=256,
                    help="Max decode steps per staff during inference.")

# new
parser.add_argument("--max-decode-steps", type=int, default=2048,
                    help="Max decode steps per system crop during inference. "
                         "Aligns with SystemInferencePipeline default.")
```

**#4 — stale plan commands**

In `docs/superpowers/plans/2026-05-10-radio-subproject4-implementation.md`, replace any usage of:
- `--stage-a-yolo` → `--stage-a-weights`
- `--predictions-dir` → `--output-dir`
- `--status-jsonl <path>` → drop (no override flag exists; status path is `eval/results/lieder_<name>_inference_status.jsonl`).

**#6 — `stage_d_skipped_systems` in scoring CSV**

In `eval/_scoring_utils.py`:

```python
# CSV_HEADER add a new column at end (preserves backward compatibility for readers that index by name)
CSV_HEADER = (
    ...,
    "stage_d_skipped_systems",
)

# _read_stage_d_diag — parse skipped_systems list from sidecar JSON
def _read_stage_d_diag(diag_path: Path) -> dict:
    ...
    skipped = data.get("skipped_systems", [])
    return {
        ...,
        "stage_d_skipped_systems": len(skipped),
    }
```

When the sidecar is missing or unreadable, the field defaults to `0` (matches existing missing-sidecar behavior for other diag fields).

### 4. Scratchpad Triage Rules

**Rule:** anything that captures a specific run, a specific environment, or a specific session moves to `archive/`. Anything that's a general-purpose entrypoint or reproducible config stays canonical.

**Personal scripts → `archive/scripts/`:**

- `cu132_phase{3,4,6}_*.ps1` (6 files) — one-time perf benchmarks
- `mvp_inner.ps1`, `launch_mvp.ps1`
- `backup_data_to_nas.ps1`, `restore_data_*.ps1` (3 files)
- `lieder_eval_stage3_*.ps1`, `lieder_full_inner.ps1` (3 files)
- `index_full_*.ps1` (2 files)
- `multi_dpi_render_*.ps1`, `synthetic_*.ps1`, `synthetic_regen_*.ps1` (6 files)
- `full_radio_stage{1,2,3}_{inner,launch}.ps1` (6 files)
- `train_yolo26m_phase{1,2}_launch.ps1`, `train_yolo_inner.ps1` (3 files)
- `build_full_manifest.cmd`
- `debug_burleigh_synthetic.py`
- Any `scripts/*.py` flagged "verify during execution" that turn out to be personal

**Canonical scripts kept** (and generalized where they hardcode personal paths):

- `setup_venv_cu132.ps1` — already uses `$env:USERPROFILE`, no path generalization needed; verified has no personal references.
- `setup_venv_cu132.sh` (NEW) — Linux equivalent. Responsibilities: create `venv-cu132/`, install nightly torch+torchvision (cu132), install `nvidia-cudnn-cu13`, install `requirements.txt`, install pytest. Note: the Windows-specific DLL `sitecustomize.py` workaround does not apply to Linux.
- `cu132_venv_sitecustomize.py`
- All `*.py` data-prep / audit / training entrypoints

**Results data → `archive/results/`:**

- `eval/results/baseline_davit_lieder.csv`
- `eval/results/lieder_subproject4_scores.csv`
- `eval/results/lieder_subproject4_inference_status.jsonl`
- `eval/results/lieder_stage3_v2_smoke.csv`
- `eval/results/lieder_mvp.csv`
- `eval/results/per_dataset_*.json` (4 files)
- `eval/results/baseline_pre_rebuild/` (entire directory)
- `src/train/training_steps*.jsonl` (5 files)
- `src/data/manifests/synthetic_systems_eval_fresh.jsonl` (this file specifically; `token_manifest_stage3.jsonl` stays as a documented format reference)

`eval/results/` is recreated as an empty directory containing `.gitkeep` and a `README.md` saying:

> This directory holds eval outputs from local runs. Canonical published results live at the link in [docs/RESULTS.md](../docs/RESULTS.md). Files written here by `eval.run_lieder_eval` are gitignored.

`.gitignore` adds `eval/results/*` (with `!eval/results/.gitkeep` and `!eval/results/README.md` exceptions), `runs/`, `checkpoints/`, and `src/train/training_steps*.jsonl`.

**Other scratchpad → `archive/notes/`:**

- `NOTES.md` (top-level)
- `logs/` (top-level) — verify content during execution; archive if personal log captures
- `analyze_data.py` (top-level) — verify; archive if scratch

**Handoffs:**

All of `docs/superpowers/handoffs/*` → `archive/handoffs/`. The directory is left empty (no canonical handoff retained). If a release-note artifact is later wanted, an impersonal one is written fresh.

### 5. External `/home/ari/docs/superpowers/` Tree

Repo→external sync of recent material is **already complete** (performed during brainstorming):

- Copied 2026-05-10 Subproject 4 specs, plan, and three handoffs from in-repo → external.
- Overwrote external `2026-05-10-radio-stage3-phase2-mid-handoff.md` with the in-repo version that carries the SUPERSEDED banner.

Remaining work:

- Drop a `/home/ari/docs/superpowers/README.md` declaring repo-canonical:

> The canonical project specs, plans, audits, and handoffs for Clarity-OMR-Train-RADIO live in the repo at [`docs/superpowers/`](https://github.com/weselyj/Clarity-OMR-Train-RADIO/tree/main/docs/superpowers). This local tree is a personal session-notes archive and may contain stale material; treat the repo as authoritative.

No other external-tree changes are in scope.

### 6. Test Gating for CUDA-Only

Add `tests/conftest.py` (and a sibling `eval/tests/conftest.py` if needed) with a `pytest_collection_modifyitems` hook:

```python
# tests/conftest.py
from pathlib import Path
import pytest

CUDA_REQUIRED_DIRS = {"inference", "pipeline", "cli", "models", "train"}
SKIP_REASON = "CUDA required (this project requires a CUDA-capable GPU; see docs/HARDWARE.md)"

def pytest_collection_modifyitems(config, items):
    try:
        import torch
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False
    if cuda_available:
        return
    skip = pytest.mark.skip(reason=SKIP_REASON)
    for item in items:
        parts = Path(str(item.fspath)).parts
        if any(p in CUDA_REQUIRED_DIRS for p in parts):
            item.add_marker(skip)
```

`pytest.ini` declares the `cuda` marker for opt-in/opt-out usage:

```
[pytest]
testpaths = tests src/tests eval/tests
markers =
    cuda: tests that require a CUDA-capable GPU
```

Pure-Python tests under `tests/data/`, `tests/tokenizer/`, `tests/decoding/` continue to run on any environment. CUDA-required tests skip cleanly with an informative reason on non-CUDA boxes.

We do NOT add CPU torch pinning, lazy ultralytics imports, or version workarounds. Strict CUDA-only means the failure mode is "you need CUDA," not "let me work around your lack of CUDA."

### 7. De-personalization Audit Checklist

Run during execution; every hit gets reviewed:

```
/home/ari
C:\\Users\\Jonathan
Jonathan Wesely
seder
10\.10\.1\.29
ari-homelab
in user repo
~/Clarity-OMR/
```

(Exception: clone URLs in README/INSTALL pointing to `github.com/weselyj/Clarity-OMR-Train-RADIO` are fine.)

Author/voice scan: README and any retained docs use impersonal voice ("this project", "the trainer", "users"). First-person material moves to archive or gets rewritten.

Path conventions for `docs/paths.md`:

| Artifact | Path (relative to repo root) | Source |
|---|---|---|
| Stage A weights | `runs/detect/runs/yolo26m_systems/weights/best.pt` | After Stage A training |
| Stage 3 v2 checkpoint | `checkpoints/full_radio_stage3_v2/_best.pt` | After Stage 3 training |
| Lieder eval data | `data/openscore_lieder/{scores,eval_pdfs}/` | User clone of OpenScore Lieder |
| Encoder cache | `data/encoder_cache/` | Built via `scripts/build_encoder_cache.py` |
| Eval outputs | `eval/results/lieder_<name>*` | Gitignored; written by `eval.run_lieder_eval` |

## Verification (Success Criteria)

The implementation is complete when all of the following hold:

1. **De-personalization grep is clean.** `grep -rE "/home/ari|C:\\\\Users\\\\Jonathan|seder|10\\.10\\.1\\.29" docs/ README.md scripts/*.{ps1,sh,py} src/ requirements.txt` returns zero hits in canonical paths. Hits inside `archive/` are accepted.
2. **CUDA-only declaration is consistent.** README banner, `requirements.txt` comments, `docs/HARDWARE.md`, `docs/INSTALL.md`, and `tests/conftest.py` skip-reason all align on "CUDA required".
3. **Pure-Python tests run on a CPU box.** `pytest tests/data tests/tokenizer tests/decoding -q` exits 0 (or with only pre-existing failures unrelated to this effort).
4. **CUDA-required tests skip cleanly on a CPU box.** `pytest tests/inference tests/models tests/cli tests/pipeline tests/train eval/tests -q` reports `s` (skipped) not `E` (error) for the affected files; skip reason matches `SKIP_REASON`.
5. **`--help` works on a CUDA box without import errors.** `python -m src.cli.run_system_inference --help` and `python -m eval.run_lieder_eval --help` print without traceback. (Verified on GPU box during execution.)
6. **README link integrity.** Every link in the slim README resolves to an existing file or section in the repo.
7. **Codex code findings verifiable in source.** `eval/run_lieder_eval.py` shows the new defaults; `eval/_scoring_utils.py CSV_HEADER` includes `stage_d_skipped_systems`; the Subproject 4 plan snippets use the actual flag names.
8. **External tree carries the stub README.** `/home/ari/docs/superpowers/README.md` exists and points to the in-repo canonical tree.
9. **`docs/superpowers/handoffs/` is empty in repo.** Contents moved to `archive/handoffs/`.
10. **`eval/results/` is empty save for `.gitkeep` + `README.md`.** `.gitignore` excludes future eval outputs.

## Out-of-Scope / Follow-Ups

1. **HuggingFace release** — model card, weights upload, scoring artifact bundle. `docs/RESULTS.md` is a placeholder; the actual release is a separate effort, tracked as the natural follow-up to this cleanup.
2. **Backfill `lieder_subproject4_scores.csv`** with the new `stage_d_skipped_systems` column. Requires a GPU-box rerun of `eval.score_lieder_eval` against the existing 50-piece sidecar set.
3. **CI for the test suite.** Test gating here just makes failures informative; running tests automatically is a separate effort.
4. **Pruning the external `/home/ari/docs/superpowers/` tree's older content.** That tree is the author's personal scratchpad; only the repo→external sync (done) is in scope.
5. **General `src/` refactor.** Out of scope; only the three targeted code fixes (#2/#3/#6) are in this spec.
