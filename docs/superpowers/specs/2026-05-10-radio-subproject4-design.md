# Subproject 4 — Per-System End-to-End Inference Pipeline

**Status:** design approved 2026-05-10 — ready for implementation plan
**Branch off:** `main` at HEAD `83ffcea` (post-PR #43)
**Predecessor:** [`2026-05-10-per-system-cleanup-wrap.md`](../handoffs/2026-05-10-per-system-cleanup-wrap.md) (handoff that scoped this work)

## Goal

Replace the archived per-staff inference pipeline with a per-system end-to-end pipeline that takes a PDF and produces MusicXML, then run a 50-piece lieder corpus eval as the architectural ship-gate that Phase 2 was supposed to provide before the format mismatch was discovered.

The ship-gate question is **"does the per-system pipeline produce non-broken output that beats the per-staff `0.067` baseline?"** — answered by `onset_f1`. A specific target value is not part of the gate; "the eval ran cleanly and produced a number" is.

## Non-goals

- Re-train Stage 3 v2 (existing checkpoint at `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` is the production weights for this work).
- Optimize for inference speed beyond what's needed to finish 50 pieces in a few hours.
- Beat any specific `onset_f1` threshold. Subproject 4 ships when the eval runs cleanly. The number it produces informs the next phase but doesn't gate this one.

## Architecture choice

Library + thin CLI for inference; **two-pass eval** that keeps inference and metric scoring in separate processes (per archived design — see below):

- A `SystemInferencePipeline` library class loads YOLO + Stage B + tokenizer once and exposes `run_pdf` / `run_page` / `run_system_crop` methods.
- A thin CLI (`run_system_inference`) instantiates the library and calls `run_pdf` once — the smoke and one-off path.
- **Phase 1 (inference)**: `eval/run_lieder_eval.py` imports the library, loads YOLO + Stage B once, loops in-process over 50 pieces. For each piece it writes the predicted `.musicxml` and the `.musicxml.diagnostics.json` sidecar immediately, plus a status JSONL row. **No metric computation in this loop.**
- **Phase 2 (scoring)**: `eval/score_lieder_eval.py` runs after Phase 1, subprocess-isolating per-piece metric computation so music21/zss memory is fully reclaimed between pieces.

The two-pass split is **load-bearing**, not a stylistic choice. The archived `run_lieder_eval.py` documents that an earlier in-process scoring run hit 43 GB committed memory at piece 6/20 and had to be killed before pagefile exhaustion (PR #26 motivation, `archive/per_staff/eval/run_lieder_eval.py:1-13`). Putting scoring back in-process would reintroduce that exact OOM.

The library still benefits the inference phase: YOLO + Stage B are loaded **once** instead of 50×. Per-piece try/except + immediate `.musicxml` + sidecar + status JSONL writes give resilience without the 50× model-reload cost of the archived per-piece subprocess inference pattern. If a single piece OOMs *during inference* (rare on a 24 GB+ GPU for inference), the run continues. We can add a `--subprocess` fallback flag for inference if repeated OOMs ever appear.

## Components

### NEW — `src/inference/system_pipeline.py` (~250 lines)

```python
class SystemInferencePipeline:
    def __init__(
        self,
        yolo_weights: Path,
        stage_b_ckpt: Path,
        *,
        device: str = "cuda",
        beam_width: int = 1,
        max_decode_steps: int = 2048,
        page_dpi: int = 300,
        image_height: int = 250,
        image_max_width: int = 2500,
        length_penalty_alpha: float = 0.4,
        use_fp16: bool = False,
        quantize: bool = False,
    ): ...

    def run_pdf(
        self,
        pdf_path: Path,
        *,
        diagnostics: Optional[StageDExportDiagnostics] = None,
    ) -> AssembledScore: ...
    def run_page(self, page_image: Image.Image, page_index: int = 0) -> List[StaffRecognitionResult]: ...
    def run_system_crop(self, crop: Image.Image, system_index: int, system_location: dict) -> List[StaffRecognitionResult]: ...

    def export_musicxml(
        self,
        score: AssembledScore,
        out_path: Path,
        *,
        diagnostics: Optional[StageDExportDiagnostics] = None,
    ) -> None:
        """Write .musicxml + .musicxml.diagnostics.json sidecar via the
        _with_diagnostics export path."""
```

`__init__` loads:
- `YoloStageASystems` (below).
- Stage B checkpoint via the new shared loader `load_stage_b_for_inference` (below). The pipeline never calls `_prepare_model_for_inference` directly — the loader does, as the final step of an 8-step sequence.
- Tokenizer via `build_default_vocabulary()` (no external vocab artifact; the tokenizer ships with the repo).

`run_pdf` chains: PyMuPDF render → `run_page` per page → flatten + `assemble_score_from_system_predictions` → optionally records skip events into the caller-supplied `StageDExportDiagnostics`.

`export_musicxml` writes both the `.musicxml` and the `.musicxml.diagnostics.json` sidecar (matching the contract that the lifted `run_lieder_eval.py` and `score_lieder_eval.py` expect — see `eval/_scoring_utils.py:253, 333`).

### NEW — `src/inference/checkpoint_load.py` (~60 lines)

Factored from `src/eval/evaluate_stage_b_checkpoint.py:285-345`. Single shared loader that produces a fully-prepared inference-ready Stage B model:

```python
@dataclass
class StageBInferenceBundle:
    model: nn.Module                  # loaded, eval-mode
    decode_model: nn.Module           # post-_prepare_model_for_inference
    vocab: OMRVocabulary              # build_default_vocabulary()
    token_to_idx: Dict[str, int]
    use_fp16: bool                    # whether decode_model expects fp16 inputs
    factory_cfg: ModelFactoryConfig   # for downstream introspection / shape checks


def load_stage_b_for_inference(
    checkpoint_path: Path,
    device: torch.device,
    *,
    use_fp16: bool = False,
    quantize: bool = False,
) -> StageBInferenceBundle:
    """Build vocab → torch.load payload → infer ModelFactoryConfig →
    build_stage_b_components → load_stage_b_checkpoint → eval() →
    _prepare_model_for_inference. Returns the full bundle."""
```

This is the single source of truth for "how to load a Stage B checkpoint for inference". Used by:
- `SystemInferencePipeline.__init__`
- Future debugging / notebook code
- (Recommended follow-up) `evaluate_stage_b_checkpoint.py` refactored to call this instead of inlining the 8-step sequence.

### NEW — `src/models/yolo_stage_a_systems.py` (~80 lines)

```python
class YoloStageASystems:
    def __init__(self, weights_path: Path, *, conf: float = 0.25, imgsz: int = 1920): ...
    def detect_systems(self, page_image: Image.Image) -> List[dict]:
        # Each dict: {system_index, bbox_extended (x1, y1, x2, y2), conf}
        # Sorted top-to-bottom by y_center.
        # bbox_extended = extend_left_for_brace applied to raw YOLO bbox.
```

Internals: `yolo_common._yolo_predict_to_boxes` → sort by y_center → assign `system_index = 0..M-1` → `extend_left_for_brace(boxes, page_w=page_image.width)`.

### EDIT — `src/pipeline/assemble_score.py` (+50 lines)

Add a single new function next to `assemble_score`:

```python
def assemble_score_from_system_predictions(
    system_token_lists: Sequence[List[str]],
    system_locations: Sequence[Dict],   # [{system_index, bbox, page_index, conf}, ...]
    *,
    sample_id_prefix: str = "",
) -> AssembledScore:
    """Compose StaffRecognitionResult list from per-system token sequences and
    delegate to the existing assemble_score()."""
```

Per system:
1. `_split_staff_sequences_for_validation(tokens)` → `List[List[str]]` of per-staff sequences.
2. `N = len(per_staff_lists)`.
3. Even-split the system bbox vertically: staff `i` gets `y_top = sys_y1 + i * sys_h / N`, `y_bottom = sys_y1 + (i + 1) * sys_h / N`.
4. Build a `StaffRecognitionResult` per staff with `system_index_hint = system_index` and the even-split `StaffLocation`.

Final: pass the flat `StaffRecognitionResult` list to existing `assemble_score()`.

### NEW — `src/cli/run_system_inference.py` (~70 lines)

argparse CLI:
- `--pdf PATH` (required)
- `--out PATH` (required, `.musicxml` output; sibling `.musicxml.diagnostics.json` written alongside)
- `--yolo-weights PATH` (required)
- `--stage-b-ckpt PATH` (required)
- `--device {cuda,cpu}` (default: cuda)
- `--beam-width N` (default: 1)
- `--max-decode-steps N` (default: 2048)
- `--page-dpi N` (default: 300)
- `--length-penalty-alpha F` (default: 0.4)
- `--fp16` (flag, default off)
- `--quantize` (flag, default off; uses existing `_prepare_model_for_inference` quantize path)

**No `--vocab` flag.** The tokenizer is built in-process via `build_default_vocabulary()`. The repo does not ship an external vocab artifact and four production code paths already use the in-code default.

Body: instantiate `SystemInferencePipeline`, build a fresh `StageDExportDiagnostics`, call `pipeline.run_pdf(pdf_path, diagnostics=diags)`, then `pipeline.export_musicxml(score, out_path, diagnostics=diags)`. The `export_musicxml` helper writes both the `.musicxml` and the `.musicxml.diagnostics.json` sidecar.

### LIFT + EDIT — `eval/run_lieder_eval.py` (~50 lines changed in 700-line file)

**Stays inference-only** — preserves the archived header comment and PR #26 motivation. The change is to the inference call, not the architecture.

From `archive/per_staff/eval/run_lieder_eval.py`. Change `run_inference()` from a subprocess invocation of `python -m src.pdf_to_musicxml` (now archived) to an in-process call on a single `SystemInferencePipeline` instance owned by `main()`. Per-piece behavior:

1. Wrap inference in try/except; status is `ok`, `oom`, `decode_error`, `yolo_error`, `assembly_error`, etc.
2. On success, call `pipeline.export_musicxml(score, out_path, diagnostics=diags)` — writes both the `.musicxml` and `.musicxml.diagnostics.json` sidecar.
3. Write status JSONL row immediately (existing pattern).
4. **Do not score in this loop** — no `score_one_piece` call, no music21/zss state accumulates.

Keep all existing orchestration: per-piece workdir, ETA logging, status manifest, CSV resume from JSONL.

After the inference loop completes, optionally orchestrate scoring via:

```python
if args.run_scoring:
    subprocess.run([
        sys.executable, "-m", "eval.score_lieder_eval",
        "--predictions-dir", str(predictions_dir),
        "--ground-truth-dir", str(gt_dir),
        "--out-csv", str(scores_csv),
    ] + (["--tedn"] if args.tedn else []), check=False)
```

This `--run-scoring` flag is convenience for "one command, two phases." Default off so a Phase 1 run can be inspected before scoring.

Add `--tedn` flag (default off) — passed through to the scoring subprocess only.

### LIFT + EDIT — `eval/score_lieder_eval.py` (~minor)

From `archive/per_staff/eval/score_lieder_eval.py`. Keep its existing subprocess-per-piece isolation and adaptive memory throttle (`--memory-limit-gb`, `_wait_for_memory_budget`, `--child-memory-limit-gb`). The OOM history bakes these protections in for a reason; do not remove them.

Add `--tedn` flag (default off); skip TEDN computation when not set. Otherwise unchanged — scoring is format-agnostic.

### Existing — `eval/lieder_split.py` (already live, no change)

74-line module already in the live tree. Provides `get_eval_pieces()` (145-piece deterministic eval split), `split_hash()`, and friends. The 50-piece subset is taken via the eval driver's existing `--max-pieces 50` flag.

### Reused unchanged

- `src/inference/decoder_runtime.py` (`_prepare_model_for_inference`, `_encode_staff_image`, `_decode_stage_b_tokens`, `_load_stage_b_crop_tensor`)
- `src/data/yolo_common.py` (`iou_xyxy`, `_yolo_predict_to_boxes`)
- `src/models/system_postprocess.py:extend_left_for_brace`
- `src/data/convert_tokens.py:_split_staff_sequences_for_validation`
- `src/pipeline/assemble_score.py:assemble_score` and `group_staves_into_systems`
- `src/pipeline/export_musicxml.py:assembled_score_to_music21_with_diagnostics` and `StageDExportDiagnostics` (the **diagnostics-aware** export path; the new pipeline writes both `.musicxml` and `.musicxml.diagnostics.json` per the contract that `eval/_scoring_utils.py:253, 333` expects)
- `src/checkpoint_io.py:load_stage_b_checkpoint` (called by the new `load_stage_b_for_inference` helper)
- `src/train/model_factory.py:build_stage_b_components`, `model_factory_config_from_checkpoint_payload`, `ModelFactoryConfig`
- `src/tokenizer/vocab.py:build_default_vocabulary` and `OMRVocabulary`

## Data flow

```
PDF
 │
 │ PyMuPDF render at page_dpi
 ▼
List[PIL.Image]  (one per PDF page)
 │
 │ for each page → YoloStageASystems.detect_systems
 ▼
List[{system_index, bbox_extended, conf}]   (sorted top-to-bottom)
 │
 │ for each system: page_image.crop(bbox_extended)
 ▼
PIL.Image (system crop)
 │
 │ decoder_runtime._encode_staff_image → memory tensor
 │ decoder_runtime._decode_stage_b_tokens → token list
 ▼
List[str]  (system tokens, e.g. <bos><staff_start><staff_idx_0>...<staff_end><staff_start><staff_idx_1>...<staff_end><eos>)
 │
 │ collect (system_tokens, system_location) pairs across all pages
 ▼
assemble_score_from_system_predictions
 │
 │  per system: _split_staff_sequences_for_validation → List[List[str]] (per-staff)
 │  per staff: even-split system bbox, build StaffRecognitionResult with system_index_hint
 ▼
flat List[StaffRecognitionResult]
 │
 │ assemble_score (existing)
 ▼
AssembledScore
 │
 │ assembled_score_to_music21_with_diagnostics (existing)
 │ + write .musicxml.diagnostics.json sidecar
 ▼
.musicxml file + .musicxml.diagnostics.json sidecar
```

(The diagnostics sidecar is consumed by `eval/score_lieder_eval.py` and `eval/_scoring_utils.py`. Skipping it would silently break Stage-D skip-counter aggregation in the scorer.)

## Eval driver behavior

### Phase 1 — inference loop (in-process, single process)

```python
pipeline = SystemInferencePipeline(yolo_weights=..., stage_b_ckpt=...)

for piece_id, pdf_path, _gt_mxl_path in iter_pieces(get_eval_pieces(), max_pieces=50):
    if piece_id in already_completed_status:
        continue
    t_start = time.monotonic()
    diags = StageDExportDiagnostics()
    try:
        score = pipeline.run_pdf(pdf_path, diagnostics=diags)
        pipeline.export_musicxml(score, out_path, diagnostics=diags)
        # writes <out_path> + <out_path>.diagnostics.json
        record = {"piece_id": piece_id, "status": "ok",
                  "musicxml": str(out_path),
                  "wall_sec": time.monotonic() - t_start}
    except Exception as e:
        record = {"piece_id": piece_id, "status": f"failed:{type(e).__name__}",
                  "error": str(e)[:200],
                  "wall_sec": time.monotonic() - t_start}
        log.exception("piece %s inference failed", piece_id)
    write_status_jsonl(status_path, record)   # immediate flush
```

**No metric scoring in this loop.** music21/zss state never accumulates across pieces.

### Phase 2 — scoring (subprocess-per-piece, separate process tree)

After Phase 1, either:
- run `python -m eval.score_lieder_eval --predictions-dir … --ground-truth-dir … --out-csv … [--tedn]` manually, **or**
- pass `--run-scoring` to `run_lieder_eval` to have it spawn the scorer as the final step.

`score_lieder_eval` already isolates per-piece scoring in its own subprocess and uses `_wait_for_memory_budget` to throttle TEDN under memory pressure. We keep that exactly as-is.

This split is not optional — it's why Subproject 4's eval can complete without the OOM that killed the earlier 20-piece run at piece 6.

## Ship-gate

Subproject 4 ships when **all** of:

1. **Smoke on `lc6623145`**: Phase 1 produces a `.musicxml` + `.musicxml.diagnostics.json` pair that loads in music21 without exception; Phase 2 produces a CSV row for the piece with a numeric `onset_f1`. **No specific F1 threshold gates this** — per the non-goals, Subproject 4 does not gate on the value.
2. **Sanity check (informational, not gating):** smoke `onset_f1` should be **well above** the broken-pipeline baseline of `0.067`. If the smoke piece scores at or near `0.067`, that's a strong signal something is still wired wrong (likely Stage A format mismatch or sidecar loss); the eval is not measuring the model. Treat any near-baseline result as a stop-and-investigate signal, not a green light to run the corpus eval.
3. **Full 50-piece Phase 1 + Phase 2 run completes**: every piece has a status JSONL row from Phase 1 and (where Phase 1 succeeded) a CSV row from Phase 2. No orphan crashes that lose pieces.
4. **At least 80% Phase 1 `status=ok`** AND **at least 80% Phase 2 scored without timeout/OOM**: ≥ 40 of 50 pieces produce a scored prediction end-to-end. Sanity floor, not a quality target.
5. **Aggregate `mean(onset_f1)` reported** across pieces with valid scores. No threshold on the value itself.

If criterion 4 fails (>10 pieces drop in either phase), do not ship — investigate the failure cluster and fix. If 1, 3, and 5 pass but the aggregate `onset_f1` is disappointing, that's a finding for the next phase, not a Subproject 4 blocker. The **architectural** ship-gate is "the eval ran cleanly and produced a real number," not "the number is good."

## Test plan

Per TDD discipline, write each test before its implementation.

| Test file | Coverage |
|---|---|
| `tests/models/test_yolo_stage_a_systems.py` | Fake YOLO model with stub predictions → verify top-to-bottom sort, `extend_left_for_brace` applied once, `system_index` assignment, `bbox_extended` clamped to page bounds |
| `tests/pipeline/test_assemble_from_system_predictions.py` | Synthetic system token lists with multi-staff sequences → verify `_split_staff_sequences_for_validation` integration, even-split y coords, `system_index_hint` preservation, delegation to `assemble_score` produces expected `AssembledScore` shape |
| `tests/inference/test_checkpoint_load.py` (NEW per review #5) | `load_stage_b_for_inference` — mock `torch.load` + `load_stage_b_checkpoint` + heavy RADIO construction, but exercise the real chain through `model_factory_config_from_checkpoint_payload` and `build_stage_b_components`. Asserts: vocab size matches checkpoint, `decode_model.eval() == True`, `use_fp16` flag round-trips, the bundle is fully populated |
| `tests/inference/test_system_pipeline.py` | Mock YOLO + mock the Stage B bundle from above → verify single-page chain produces N `StaffRecognitionResult` instances with correct `page_index`, `system_index_hint`, ordering. **Asserts the pipeline does NOT call `_prepare_model_for_inference` directly** — only via the bundle |
| `tests/inference/test_system_pipeline_pdf.py` | Pre-rendered single-page fixture image (no real PDF) + mocks → verify end-to-end `run_pdf` shape, that `export_musicxml` writes both `.musicxml` AND `.musicxml.diagnostics.json` (NEW per review #4), and that `run_pdf` / `run_page` produce consistent results |
| `tests/cli/test_run_system_inference.py` (NEW per review #5) | argparse smoke: instantiate the CLI parser with `--pdf X --out Y --yolo-weights Z --stage-b-ckpt W` and confirm parsing succeeds **without a `--vocab` flag**. Mocks the pipeline construction; asserts no extra required flags creep in |
| `eval/tests/test_run_lieder_eval.py` (lift + EDIT per review #1) | Resume-from-status, per-piece try/except, status JSONL writes; **and** asserts the inference loop does NOT call `score_one_piece` or any music21-importing scorer, and that `--run-scoring` spawns a subprocess (does not import scoring code in-process) |

Smoke on `lc6623145` is GPU-box-only (real models). Not a unit test; runs as part of the eval driver's first invocation.

## Risks

1. **Brace-margin double-counting**. Stage A v15 YOLO may already include some leftward margin per `project_clarity_omr_brace_signal.md`. Applying `extend_left_for_brace` again could over-extend and clip into the staff above. **Mitigation**: apply the extension in exactly one place (inside `YoloStageASystems.detect_systems`); verify visually on the smoke piece that the leftmost cropped column makes sense.

2. **OOM during inference**. Low risk for inference on a 24 GB+ GPU but possible. **Mitigation**: per-piece try/except writes `status=oom`; run continues. Add `--subprocess` flag as a follow-up only if we hit repeated failures.

3. **Page with zero detected systems**. **Mitigation**: pipeline logs warning, returns `[]` for that page, eval driver continues. Final MusicXML has a gap; existing scorer tolerates missing pages.

4. **`<staff_end>` count mismatch with visual staves**. Robust by design — we trust the model's emitted structure, not a visual estimate. Even-split divides by emitted `N`.

5. **Multi-page assembly**. `StaffLocation.page_index` already supports this. The unit-level `test_system_pipeline_pdf.py` covers single-page; multi-page coverage comes from the smoke + corpus eval (most lieder pieces span 1-3 pages). If the smoke piece is single-page, pick a second multi-page piece from the eval split for an additional manual smoke before the full 50-piece run.

6. **TEDN timeout flakiness**. Only relevant when `--tedn` is on. Archived driver handles via timeout + status code; we keep that behavior.

## Open items deferred to PR review

- Exact CLI flag names (bikeshed at review).
- Logging format (default: match archived driver's existing format).

## Deferred follow-ups (post-Subproject 4)

These were considered during design but are explicitly out of scope for this subproject:

- **TorchAO inference quantization** (`docs/clarity_omr_radio_torchao_evaluation.md`). After Subproject 4 produces a baseline `onset_f1`, evaluate TorchAO `Int8WeightOnlyConfig` and `Int8DynamicActivationInt8WeightConfig` on the Stage B decoder behind a `--quantize-backend torchao-int8-weight` flag. Update the existing `decoder_runtime._quantize_decoder` GPU path from `int8_weight_only()` → `Int8WeightOnlyConfig()` (current API). Do NOT promote to default; do NOT quantize the RADIO encoder (torch.hub model, compile hazard).
- **Refactor `evaluate_stage_b_checkpoint.py`** to call the new `load_stage_b_for_inference` helper instead of inlining the 8-step sequence — keeps a single source of truth for "how to load a Stage B checkpoint for inference."
- **Subprocess-per-piece inference fallback** if any 50-piece run shows repeated mid-run inference OOMs (not metric-side OOMs — those are already handled by Phase 2's existing isolation).

## References

- Handoff that scoped this work: [`docs/superpowers/handoffs/2026-05-10-per-system-cleanup-wrap.md`](../handoffs/2026-05-10-per-system-cleanup-wrap.md)
- Path reference: [`docs/locations.md`](../../locations.md)
- External design review: `/home/ari/docs/clarity_omr_radio_subproject4_review.md` (drove the two-pass eval, explicit checkpoint loader, no-vocab-flag, and diagnostics-sidecar fixes)
- TorchAO follow-up evaluation: `/home/ari/docs/clarity_omr_radio_torchao_evaluation.md`
- Per-staff archive: [`archive/per_staff/`](../../../archive/per_staff/) (sources to lift `run_lieder_eval.py`, `score_lieder_eval.py`)
- Source loader to factor: [`src/eval/evaluate_stage_b_checkpoint.py`](../../../src/eval/evaluate_stage_b_checkpoint.py) (lines 285-345 — the 8-step Stage B inference loading sequence becomes `load_stage_b_for_inference`)
- Existing system data prep (training-side): [`src/data/yolo_aligned_systems.py`](../../../src/data/yolo_aligned_systems.py)
- Shared YOLO helpers: [`src/data/yolo_common.py`](../../../src/data/yolo_common.py)
- Decoder runtime (mode-agnostic): [`src/inference/decoder_runtime.py`](../../../src/inference/decoder_runtime.py)
- Brace margin postprocess: [`src/models/system_postprocess.py`](../../../src/models/system_postprocess.py)
- Token splitter: [`src/data/convert_tokens.py`](../../../src/data/convert_tokens.py) (look for `_split_staff_sequences_for_validation`)
- Existing assembler: [`src/pipeline/assemble_score.py`](../../../src/pipeline/assemble_score.py)
- Diagnostics-aware exporter: [`src/pipeline/export_musicxml.py`](../../../src/pipeline/export_musicxml.py) (`assembled_score_to_music21_with_diagnostics`, `StageDExportDiagnostics`)
- Sidecar consumer (proves the contract): [`eval/_scoring_utils.py`](../../../eval/_scoring_utils.py)
