# OMR Training & Validation Commands (Ubuntu / Bash)

This is the Linux/Ubuntu runbook equivalent of `TRAINING_COMMANDS.md`.

## 0) Environment setup

```bash
cd /path/to/omr
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1) Build canonical dataset manifest

```bash
python3 src/data/index.py \
  --output-manifest src/data/manifests/master_manifest.jsonl \
  --output-summary src/data/manifests/master_manifest_summary.json
```

Quick check:

```bash
python3 -c "import json; s=json.load(open('src/data/manifests/master_manifest_summary.json', encoding='utf-8')); print('total_samples=', s['total_samples']); print('datasets=', list(s['datasets'].keys()))"
```

## 2) Convert sources to token manifest

```bash
python3 -m src.data.convert_tokens \
  --input-manifest src/data/manifests/master_manifest.jsonl \
  --output-manifest src/data/manifests/token_manifest.jsonl \
  --output-summary src/data/manifests/token_manifest_summary.json \
  --allow-failures
```

Quick check:

```bash
python3 -c "import json; s=json.load(open('src/data/manifests/token_manifest_summary.json', encoding='utf-8')); print('converted=', s['converted_samples']); print('failed=', s['failed_samples'])"
```

## 3) Synthetic full-page generation

### 3a) Dry-run planning

```bash
python3 src/data/generate_synthetic.py \
  --mode dry-run \
  --max-scores 200 \
  --styles leipzig-default,bravura-compact,gootville-wide \
  --output-dir data/processed/synthetic
```

### 3b) Actual render job

```bash
python3 src/data/generate_synthetic.py \
  --mode render \
  --max-scores 2000 \
  --max-pages-per-score 3 \
  --styles leipzig-default,bravura-compact,gootville-wide \
  --workers 6 \
  --output-dir data/processed/synthetic \
  --write-png \
  --roundtrip-validate
```

Quick check:

```bash
python3 -c "import json; p='data/processed/synthetic/manifests/synthetic_pages.jsonl'; rows=[json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]; bad=sum(int(r.get('staff_crop_count',0)!=r.get('paired_staff_tokens',0)) for r in rows); print({'pages':len(rows),'mismatch_pages':bad,'mismatch_rate':bad/max(len(rows),1)})"
```

### 3c) Optional legacy merge file (not required)

Stage-B commands below read base + synthetic manifests directly. Keep this merge only if you need a single-file artifact.

```bash
base="src/data/manifests/token_manifest.jsonl"
synth="data/processed/synthetic/manifests/synthetic_token_manifest.jsonl"
out="src/data/manifests/token_manifest_train.jsonl"
cat "$base" "$synth" > "$out"
```

Quick check:

```bash
python3 -c "import json; from collections import Counter; c=Counter(); p='src/data/manifests/token_manifest_train.jsonl'; [c.update([json.loads(line).get('dataset','')]) for line in open(p, encoding='utf-8') if line.strip()]; print('dataset_counts=', dict(c))"
```

## 4) Stage A (YOLOv8m) training

If you already prepared curated splits in `info/` (recommended for label QC), first filter obvious blank/low-ink pages:

```bash
python3 src/data/filter_low_ink_samples.py \
  --project-root /workspace/omr \
  --train-split /workspace/omr/info/train.txt \
  --val-split /workspace/omr/info/val.txt \
  --test-split /workspace/omr/info/test.txt \
  --ink-threshold 0.005
```

Then train Stage A from the regenerated synthetic manifest (recommended, run from `/workspace/omr`):

```bash
python3 src/train/train_yolo_stage_a.py \
  --page-manifest /workspace/omr/data/processed/synthetic/manifests/synthetic_pages.jsonl \
  --split-dir /workspace/omr/data/processed/synthetic/yolo_splits \
  --model-checkpoint yolov8m.pt \
  --epochs 100 \
  --imgsz 1920 \
  --batch-size 8 \
  --device 0 \
  --run-project runs/stage_a \
  --run-name yolov8m-stage-a
```

Reuse existing `data.yaml` (if `yolo_splits` already built):

```bash
python3 src/train/train_yolo_stage_a.py \
  --data-yaml /workspace/omr/data/processed/synthetic/yolo_splits/data.yaml \
  --model-checkpoint yolov8m.pt \
  --epochs 100 \
  --imgsz 1920 \
  --batch-size 8 \
  --device 0 \
  --run-project runs/stage_a \
  --run-name yolov8m-stage-a
```

`train_yolo_stage_a.py` now defaults to a clean-score augmentation profile (mosaic/fliplr/erasing disabled, mild HSV/scale/translate, higher cls gain, cosine LR, rectangular batches).

If no GPU is available, use `--device cpu`.

Validate YOLO checkpoint:

```bash
python3 -c "from ultralytics import YOLO; m=YOLO('runs/stage_a/yolov8m-stage-a/weights/best.pt'); print(m.val(data='/workspace/omr/data/processed/synthetic/yolo_splits/data.yaml'))"
```

Live dashboard for Stage A (epoch-level updates from `results.csv`):

```bash
python3 src/train/monitor_dashboard.py \
  --yolo-results runs/stage_a/yolov8m-stage-a/results.csv \
  --refresh-ms 2000 \
  --tail-limit 40
```

## 5) Stage B curriculum configuration check

```bash
python3 src/train/train.py \
  --token-manifest src/data/manifests/token_manifest.jsonl,data/processed/synthetic/manifests/synthetic_token_manifest.jsonl \
  --stage-configs configs/train_stage1.yaml,configs/train_stage2.yaml,configs/train_stage3.yaml \
  --mode dry-run \
  --output-summary src/train/curriculum_plan.json
```

## 6) Stage B full curriculum training execution

```bash
python3 src/train/train.py \
  --token-manifest src/data/manifests/token_manifest.jsonl,data/processed/synthetic/manifests/synthetic_token_manifest.jsonl \
  --stage-configs configs/train_stage1.yaml,configs/train_stage2.yaml,configs/train_stage3.yaml \
  --mode execute \
  --validation-batches 2 \
  --checkpoint-dir src/train/checkpoints \
  --step-log src/train/training_steps.jsonl \
  --output-summary src/train/curriculum_execute.json
```

Training health monitor:

```bash
python3 src/train/monitor_training.py \
  --step-log src/train/training_steps.jsonl \
  --window 20 \
  --spike-factor 3.0 \
  --grad-threshold 100 \
  --output src/train/training_health_summary.json
```

Live dashboard for Stage B:

```bash
python3 src/train/monitor_dashboard.py \
  --step-log src/train/training_steps.jsonl \
  --window 20 \
  --spike-factor 3.0 \
  --grad-threshold 100 \
  --refresh-ms 2000 \
  --tail-limit 40
```

## 7) Decoding, assembly, and MusicXML export

```bash
python3 src/cli.py stage-b \
  --project-root /path/to/omr \
  --crops-manifest <PATH_TO_STAGE_A_CROPS_JSONL> \
  --checkpoint <PATH_TO_STAGE_B_CHECKPOINT> \
  --output-predictions <PATH_TO_STAGE_B_PREDICTIONS_JSONL> \
  --beam-width 5 \
  --max-decode-steps 512
```

```bash
python3 src/cli.py assemble \
  --staff-predictions <PATH_TO_STAFF_PREDICTIONS_JSONL> \
  --output-assembly output/assembled_score.json
```

```bash
python3 src/cli.py export \
  --assembly-manifest output/assembled_score.json \
  --output-musicxml output/score.musicxml
```

## 8) Evaluation

```bash
python3 src/eval/run_eval.py \
  --predictions <PATH_TO_EVAL_PREDICTIONS_JSONL> \
  --output-summary src/eval/evaluation_summary.json \
  --output-ablation-template src/eval/ablation_template.json
```

## 9) Token/image pairing debug

```bash
python3 src/eval/reconstruct_tokens_image.py \
  --sample-id "<SAMPLE_ID_FROM_SYNTHETIC_TOKEN_MANIFEST>" \
  --token-manifest data/processed/synthetic/manifests/synthetic_token_manifest.jsonl \
  --comparison-mode crop \
  --output-dir src/eval/reconstruct_debug
```

## 10) Compliance check

```bash
cat src/eval/compliance_audit.json
```
