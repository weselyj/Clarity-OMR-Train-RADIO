# OMR Training & Validation Commands (Windows / PowerShell)

This document is the practical runbook for training and checking the pipeline in this repository.

## 0) Environment setup

```powershell
Set-Location C:\Users\clq\Documents\GitHub\omr
# Production setup: run the cu132 venv creation script.
# This installs torch nightly cu132, cuDNN 9.21.01, project deps,
# and writes sitecustomize.py for DLL path resolution.
.\scripts\setup_venv_cu132.ps1

# Activate after creation (or on subsequent sessions):
.\venv-cu132\Scripts\Activate.ps1
```

## 1) Build canonical dataset manifest

```powershell
python src\data\index.py `
  --output-manifest src\data\manifests\master_manifest.jsonl `
  --output-summary src\data\manifests\master_manifest_summary.json
```

Quick check:

```powershell
python -c "import json; s=json.load(open(r'src\data\manifests\master_manifest_summary.json', encoding='utf-8')); print('total_samples=', s['total_samples']); print('datasets=', list(s['datasets'].keys()))"
```

## 2) Convert sources to token manifest

```powershell
python -m src.data.convert_tokens `
  --input-manifest src\data\manifests\master_manifest.jsonl `
  --output-manifest src\data\manifests\token_manifest.jsonl `
  --output-summary src\data\manifests\token_manifest_summary.json `
  --allow-failures
```

Quick check:

```powershell
python -c "import json; s=json.load(open(r'src\data\manifests\token_manifest_summary.json', encoding='utf-8')); print('converted=', s['converted_samples']); print('failed=', s['failed_samples'])"
```

## 3) Synthetic full-page generation

### 3a) Dry-run planning

```powershell
python src\data\generate_synthetic.py `
  --mode dry-run `
  --max-scores 200 `
  --styles leipzig-default,bravura-compact,gootville-wide `
  --output-dir data\processed\synthetic
```

### 3b) Actual render job

```powershell
python src\data\generate_synthetic.py `
  --mode render `
  --max-scores 2000 `
  --max-pages-per-score 3 `
  --styles leipzig-default,bravura-compact,gootville-wide `
  --workers 6 `
  --output-dir data\processed\synthetic `
  --write-png `
  --roundtrip-validate
```

Quick check:

```powershell
python -c "import json; p=r'data\processed\synthetic\manifests\synthetic_pages.jsonl'; rows=[json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]; bad=sum(int(r.get('staff_crop_count',0)!=r.get('paired_staff_tokens',0)) for r in rows); print({'pages':len(rows),'mismatch_pages':bad,'mismatch_rate':bad/max(len(rows),1)})"
```

### 3c) Optional legacy merge file (not required)

Stage-B commands below read base + synthetic manifests directly. Keep this merge only if you need a single-file artifact.

```powershell
$base = "src\data\manifests\token_manifest.jsonl"
$synth = "data\processed\synthetic\manifests\synthetic_token_manifest.jsonl"
$out = "src\data\manifests\token_manifest_train.jsonl"
Get-Content $base, $synth | Set-Content $out
```

Quick check:

```powershell
python -c "import json; from collections import Counter; c=Counter(); p=r'src\data\manifests\token_manifest_train.jsonl'; [c.update([json.loads(line).get('dataset','')]) for line in open(p, encoding='utf-8') if line.strip()]; print('dataset_counts=', dict(c))"
```

## 4) Stage A (YOLOv8m) training

If you already prepared curated splits in `info\` (recommended for label QC), first filter obvious blank/low-ink pages:

```powershell
python src\data\filter_low_ink_samples.py `
  --project-root C:\Users\clq\Documents\GitHub\omr `
  --train-split C:\Users\clq\Documents\GitHub\omr\info\train.txt `
  --val-split C:\Users\clq\Documents\GitHub\omr\info\val.txt `
  --test-split C:\Users\clq\Documents\GitHub\omr\info\test.txt `
  --ink-threshold 0.005
```

Then train YOLOv8m from the regenerated synthetic manifest (recommended):

```powershell
python src\train\train_yolo_stage_a.py `
  --page-manifest C:\Users\clq\Documents\GitHub\omr\data\processed\synthetic\manifests\synthetic_pages.jsonl `
  --split-dir C:\Users\clq\Documents\GitHub\omr\data\processed\synthetic\yolo_splits `
  --model-checkpoint yolov8m.pt `
  --epochs 100 `
  --imgsz 1920 `
  --batch-size 8 `
  --device 0 `
  --run-project runs\stage_a `
  --run-name yolov8m-stage-a
```

Reuse existing `data.yaml` (if `yolo_splits` already built):

```powershell
python src\train\train_yolo_stage_a.py `
  --data-yaml C:\Users\clq\Documents\GitHub\omr\data\processed\synthetic\yolo_splits\data.yaml `
  --model-checkpoint yolov8m.pt `
  --epochs 100 `
  --imgsz 1920 `
  --batch-size 8 `
  --device 0 `
  --run-project runs\stage_a `
  --run-name yolov8m-stage-a
```

`train_yolo_stage_a.py` now defaults to a clean-score augmentation profile (mosaic/fliplr/erasing disabled, mild HSV/scale/translate, higher cls gain, cosine LR, rectangular batches).

Validate YOLO checkpoint:

```powershell
python -c "from ultralytics import YOLO; m=YOLO(r'runs\stage_a\yolov8m-stage-a\weights\best.pt'); print(m.val(data=r'data\processed\synthetic\yolo_splits\data.yaml'))"
```

Live dashboard for Stage A (reads YOLO `results.csv` while training is running):

```powershell
python src\train\monitor_dashboard.py `
  --yolo-results runs\stage_a\yolov8m-stage-a\results.csv `
  --refresh-ms 2000 `
  --tail-limit 40
```

## 5) Stage B curriculum configuration check

```powershell
python src\train\train.py `
  --token-manifest src\data\manifests\token_manifest.jsonl,data\processed\synthetic\manifests\synthetic_token_manifest.jsonl `
  --stage-configs configs\train_stage1.yaml,configs\train_stage2.yaml,configs\train_stage3.yaml `
  --mode dry-run `
  --output-summary src\train\curriculum_plan.json
```

Quick check:

```powershell
python -c "import json; s=json.load(open(r'src\train\curriculum_plan.json', encoding='utf-8')); print([(x['stage_name'], x['warnings']) for x in s['stages']])"
```

## 6) Stage B full curriculum training execution

```powershell
python src\train\train.py `
  --token-manifest src\data\manifests\token_manifest.jsonl,data\processed\synthetic\manifests\synthetic_token_manifest.jsonl `
  --stage-configs configs\train_stage1.yaml,configs\train_stage2.yaml,configs\train_stage3.yaml `
  --mode execute `
  --validation-batches 2 `
  --checkpoint-dir src\train\checkpoints `
  --step-log src\train\training_steps.jsonl `
  --output-summary src\train\curriculum_execute.json
```

Quick check:

```powershell
python -c "import json; s=json.load(open(r'src\train\curriculum_execute.json', encoding='utf-8')); print([(x['stage_name'], x['steps_executed'], x['planned_total_steps'], x['truncated_by_max_steps']) for x in s['stages']])"
```

### 6b) Optional short debug run (before long training)

```powershell
python src\train\train.py `
  --token-manifest src\data\manifests\token_manifest.jsonl,data\processed\synthetic\manifests\synthetic_token_manifest.jsonl `
  --stage-configs configs\train_stage1.yaml `
  --mode execute `
  --max-steps-per-stage 50 `
  --validation-batches 1 `
  --checkpoint-dir src\train\checkpoints_debug `
  --step-log src\train\training_steps_debug.jsonl `
  --output-summary src\train\curriculum_execute_debug.json
```

### 6c) Training health monitor (LR / loss / gradients)

```powershell
python src\train\monitor_training.py `
  --step-log src\train\training_steps.jsonl `
  --window 20 `
  --spike-factor 3.0 `
  --grad-threshold 100 `
  --output src\train\training_health_summary.json
```

Fail fast in automation if unhealthy:

```powershell
python src\train\monitor_training.py `
  --step-log src\train\training_steps.jsonl `
  --fail-on-alert
```

### 6d) Live dashboard (auto-refresh, alerts, recent steps)

```powershell
python src\train\monitor_dashboard.py `
  --step-log src\train\training_steps.jsonl `
  --window 20 `
  --spike-factor 3.0 `
  --grad-threshold 100 `
  --refresh-ms 2000 `
  --tail-limit 40
```

## 7) Decoding, assembly, and MusicXML export

Run Stage-B inference from Stage-A crops:

```powershell
python src\cli.py stage-b `
  --project-root C:\Users\clq\Documents\GitHub\omr `
  --crops-manifest <PATH_TO_STAGE_A_CROPS_JSONL> `
  --checkpoint <PATH_TO_STAGE_B_CHECKPOINT> `
  --output-predictions <PATH_TO_STAGE_B_PREDICTIONS_JSONL> `
  --beam-width 5 `
  --max-decode-steps 512
```

Assemble staff-level predictions:

```powershell
python src\cli.py assemble `
  --staff-predictions <PATH_TO_STAFF_PREDICTIONS_JSONL> `
  --output-assembly output\assembled_score.json
```

Export to MusicXML:

```powershell
python src\cli.py export `
  --assembly-manifest output\assembled_score.json `
  --output-musicxml output\score.musicxml
```

End-to-end (Stage A + assembly + export):

```powershell
python src\cli.py run `
  --image <PATH_TO_PAGE_IMAGE> `
  --weights <PATH_TO_YOLO_WEIGHTS> `
  --stage-b-checkpoint <PATH_TO_STAGE_B_CHECKPOINT> `
  --work-dir output\run_work `
  --output-musicxml output\score.musicxml
```

End-to-end with externally precomputed Stage-B predictions:

```powershell
python src\cli.py run `
  --image <PATH_TO_PAGE_IMAGE> `
  --weights <PATH_TO_YOLO_WEIGHTS> `
  --staff-predictions <PATH_TO_STAGE_B_PREDICTIONS_JSONL> `
  --work-dir output\run_work `
  --output-musicxml output\score.musicxml
```

## 8) Evaluation + ablation template output

Predictions JSONL rows must include:
- `sample_id`
- `dataset`
- `pred_tokens`
- `gt_tokens`
- optional: `pred_musicxml_path`

Run eval:

```powershell
python src\eval\run_eval.py `
  --predictions <PATH_TO_EVAL_PREDICTIONS_JSONL> `
  --output-summary src\eval\evaluation_summary.json `
  --output-ablation-template src\eval\ablation_template.json
```

Quick check:

```powershell
python -c "import json; s=json.load(open(r'src\eval\evaluation_summary.json', encoding='utf-8')); print('overall=', s['overall']); print('musicxml_validity_rate=', s['musicxml_validity_rate'])"
```

## 9) Token/image pairing debug (reconstruct from tokens)

Reconstruct a single staff image from one token-manifest sample and compare against the saved crop:

```powershell
python src\eval\reconstruct_tokens_image.py `
  --sample-id "<SAMPLE_ID_FROM_SYNTHETIC_TOKEN_MANIFEST>" `
  --token-manifest data\processed\synthetic\manifests\synthetic_token_manifest.jsonl `
  --comparison-mode crop `
  --output-dir src\eval\reconstruct_debug
```

Reconstruct a page from all staff token sequences tied to a page id, then compare against the full rendered page:

```powershell
python src\eval\reconstruct_tokens_image.py `
  --page-id "<PAGE_ID_FROM_SYNTHETIC_PAGES_MANIFEST>" `
  --token-manifest data\processed\synthetic\manifests\synthetic_token_manifest.jsonl `
  --page-manifest data\processed\synthetic\manifests\synthetic_pages.jsonl `
  --output-dir src\eval\reconstruct_debug
```

Each run writes:
- `*_reconstructed.musicxml`
- `*_reconstructed.png`
- `*_comparison.png` (if reference image exists)
- `*_summary.json` (includes MSE/SSIM)

## 10) Compliance check against `omr-final-plan.md`

The compliance audit JSON (`src\eval\compliance_audit.json`) was generated during the initial
implementation phase. It is kept as a historical reference but the compliance-audit workflow
is no longer actively run. See `docs/omr-final-plan.md` for the authoritative design record.

Expected summary:
- `pass: 15`
- `fail: 0`

## How to tell if training is correct

Use this checklist during runs:

1. **Data integrity**
   - `master_manifest_summary.json` has expected non-zero counts per dataset.
   - `token_manifest_summary.json` has very low or zero failures.

2. **Stage A (YOLO) learning signal**
   - Training/validation losses decrease.
   - mAP rises and stabilizes.
   - Visual spot-checks show correct class (`staff`).

3. **Stage B curriculum correctness**
   - `curriculum_plan.json` has no warnings for required datasets.
   - Execute-mode run gives finite losses (no NaN/Inf).
   - Sequence lengths match stage design (256/384/512).

4. **Decoder/assembly validity**
   - Constrained decoding outputs grammar-valid token streams.
   - Assembly keeps measure counts and key/time signatures aligned across system staves.

5. **MusicXML output quality**
   - Export validation reports `measure_mismatches` near zero.
   - `musicxml_validity_rate` trends toward target (plan target >98%).

6. **OMR metrics trend**
   - SER decreases over checkpoints.
   - Pitch/rhythm accuracy and structural F1 increase on held-out sets.
