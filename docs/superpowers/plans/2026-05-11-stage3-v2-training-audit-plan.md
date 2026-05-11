# Stage 3 v2 Training Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the audit designed in [`docs/superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md`](../specs/2026-05-11-stage3-v2-training-audit-design.md) — three diagnostic scripts (Phase A) plus a conditional code-review walkthrough (Phase B), producing a single audit report whose Recommendation triggers the next sub-project.

**Architecture:** Three single-purpose Python scripts under `scripts/audit/`, each runnable standalone on seder. Each script writes a JSON results file consumed by the final report. Phase B is read-only code review (no scripts). Final report is human-written markdown synthesizing the JSON evidence plus Phase B findings.

**Tech Stack:** Python 3.13, PyTorch (CUDA on seder), PIL, music21. All scripts use the existing `venv-cu132` on seder. No new third-party dependencies.

**Spec sections this plan implements:**
- Phase A → Tasks 1-7
- Phase A decision gate → Task 8
- Phase B (conditional) → Task 9
- Deliverable / report → Task 10

**Pre-existing context worth noting before starting:**
- **Cache builder vs train vs inference resampling mismatch:** [`scripts/build_encoder_cache.py:93`](../../scripts/build_encoder_cache.py#L93) uses `Image.LANCZOS`; [`src/train/train.py:407`](../../src/train/train.py#L407) and [`src/inference/decoder_runtime.py:31`](../../src/inference/decoder_runtime.py#L31) both use `Image.Resampling.BILINEAR`. This is a leading hypothesis for Phase A2; the audit script should be sensitive enough to detect this if it's real.
- **Inference saves crop to temp PNG, then re-reads:** [`src/inference/system_pipeline.py:155-185`](../../src/inference/system_pipeline.py#L155) — round-trip through PNG (lossless but worth verifying no PIL mode coercion happens).
- **Stage 3 v2 was trained with frozen encoder:** checkpoint named `stage3-radio-systems-frozen-encoder_best.pt`. Frozen encoder → cached features used during training → A2 parity is the load-bearing check.
- **CUDA-required tests:** Tests in `tests/inference/`, `tests/pipeline/`, `tests/cli/`, `tests/models/`, `tests/train/` are skipped on local (per `tests/conftest.py`). All audit scripts run on seder; smoke tests for these scripts also run on seder.

**Where the audit runs:**
- Audit scripts execute on **seder** (Windows, CUDA 13, `venv-cu132`) via SSH.
- Sample data, checkpoints, and the encoder cache all live on seder under `C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\`.
- Result JSON files are read locally via `scp` for analysis and report-writing.

**Branch strategy:** New branch `audit/stage3-v2-training` off `main` at `1759781` (current HEAD with the spec just committed). All audit code lives on this branch. The branch produces a single PR adding `scripts/audit/` + the audit report; PR can merge regardless of audit outcome (the scripts have long-term diagnostic value).

---

## File Structure

**New files (audit branch):**
- `scripts/audit/__init__.py` — empty package marker
- `scripts/audit/_sample_picker.py` — shared utility to pick N training samples per corpus
- `scripts/audit/a1_preprocessing_parity.py` — compares training vs inference image tensors
- `scripts/audit/a2_encoder_parity.py` — compares cached vs live encoder features
- `scripts/audit/a3_decoder_on_training.py` — runs end-to-end inference on training data, compares to labels
- `tests/audit/__init__.py` — empty package marker
- `tests/audit/test_sample_picker.py` — unit test for sample picker (CUDA-gated free; runs locally)
- `docs/audits/2026-05-11-stage3-v2-training-audit.md` — the final report (filled at Task 10)

**Generated artifacts (not in git, on seder):**
- `audit_results/a1_preprocessing.json`
- `audit_results/a2_encoder.json`
- `audit_results/a3_decoder.json`

---

## Task 1: Sample picker utility

**Files:**
- Create: `scripts/audit/__init__.py`
- Create: `scripts/audit/_sample_picker.py`
- Create: `tests/audit/__init__.py`
- Create: `tests/audit/test_sample_picker.py`

- [ ] **Step 1.1: Write the failing test for sample picker**

Create `tests/audit/test_sample_picker.py`:

```python
"""Unit tests for the audit sample picker."""
from __future__ import annotations
import json
from pathlib import Path


def _write_manifest(path: Path, entries: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


def test_picks_n_samples_per_corpus(tmp_path: Path):
    from scripts.audit._sample_picker import pick_audit_samples

    # Build fake manifest with 100 entries across 4 corpora
    manifest = tmp_path / "manifest.jsonl"
    entries = []
    for corpus in ("synthetic_v2", "grandstaff", "primus", "cameraprimus"):
        for i in range(25):
            entries.append({
                "sample_id": f"{corpus}_{i:03d}",
                "dataset": corpus,
                "split": "train" if i < 20 else "val",
                "image_path": f"/fake/{corpus}/{i:03d}.png",
                "token_sequence": ["<bos>", "clef-G2", "<eos>"],
            })
    _write_manifest(manifest, entries)

    samples = pick_audit_samples(manifest, n_per_corpus=5, seed=42)

    # 4 corpora x 5 samples = 20
    assert len(samples) == 20
    # 5 from each corpus
    from collections import Counter
    counts = Counter(s["dataset"] for s in samples)
    assert all(c == 5 for c in counts.values()), f"got {counts}"
    # Only train split
    assert all(s["split"] == "train" for s in samples)
    # Deterministic with same seed
    samples2 = pick_audit_samples(manifest, n_per_corpus=5, seed=42)
    assert [s["sample_id"] for s in samples] == [s["sample_id"] for s in samples2]


def test_handles_missing_corpus(tmp_path: Path):
    """If a corpus has 0 train entries, return what's available without crashing."""
    from scripts.audit._sample_picker import pick_audit_samples

    manifest = tmp_path / "manifest.jsonl"
    entries = [
        {"sample_id": "a", "dataset": "synthetic_v2", "split": "train",
         "image_path": "/x.png", "token_sequence": []},
        {"sample_id": "b", "dataset": "synthetic_v2", "split": "train",
         "image_path": "/y.png", "token_sequence": []},
    ]
    _write_manifest(manifest, entries)

    samples = pick_audit_samples(manifest, n_per_corpus=5, seed=42)

    # Only 2 available, all from synthetic_v2
    assert len(samples) == 2
    assert all(s["dataset"] == "synthetic_v2" for s in samples)
```

- [ ] **Step 1.2: Run test to verify it fails**

Run: `python3 -m pytest tests/audit/test_sample_picker.py -v`
Expected: 2 failures — `ModuleNotFoundError: No module named 'scripts.audit._sample_picker'`

- [ ] **Step 1.3: Create the empty package markers**

Create `scripts/audit/__init__.py` (empty file).
Create `tests/audit/__init__.py` (empty file).

- [ ] **Step 1.4: Implement the sample picker**

Create `scripts/audit/_sample_picker.py`:

```python
"""Pick a stratified sample of training entries for the audit.

The audit needs ~20 training samples spread across all 4 corpora so the
parity / round-trip experiments aren't biased toward whichever corpus
happens to come first in the manifest. Deterministic given a seed so the
audit is reproducible.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List


def pick_audit_samples(
    manifest_path: Path,
    *,
    n_per_corpus: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """Pick `n_per_corpus` train-split entries from each corpus.

    Returns entries in deterministic order (corpus-name asc, then
    sample_id asc among the picked set) so downstream scripts that
    iterate `samples` in order produce stable output.
    """
    rng = random.Random(seed)
    by_corpus: Dict[str, List[Dict]] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("split") != "train":
                continue
            corpus = entry.get("dataset", "unknown")
            by_corpus.setdefault(corpus, []).append(entry)

    picked: List[Dict] = []
    for corpus in sorted(by_corpus):
        pool = by_corpus[corpus]
        k = min(n_per_corpus, len(pool))
        chosen = rng.sample(pool, k) if k > 0 else []
        chosen.sort(key=lambda e: e["sample_id"])
        picked.extend(chosen)
    return picked


if __name__ == "__main__":  # pragma: no cover
    import sys
    manifest = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    for entry in pick_audit_samples(manifest, n_per_corpus=n, seed=seed):
        print(f"{entry['dataset']:<20} {entry['sample_id']}")
```

- [ ] **Step 1.5: Run test to verify it passes**

Run: `python3 -m pytest tests/audit/test_sample_picker.py -v`
Expected: 2 PASS

- [ ] **Step 1.6: Commit**

```bash
git checkout -b audit/stage3-v2-training
git add scripts/audit/__init__.py scripts/audit/_sample_picker.py tests/audit/__init__.py tests/audit/test_sample_picker.py
git commit -m "feat(audit): add stratified sample picker for training-audit experiments

Picks N train-split entries per corpus with deterministic ordering.
First building block for the Stage 3 v2 training audit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 2: A1 — Image preprocessing parity script

**Goal:** Verify that the training-time and inference-time image preprocessing pipelines produce identical tensors when given the same input image.

**Files:**
- Create: `scripts/audit/a1_preprocessing_parity.py`

- [ ] **Step 2.1: Write the script skeleton with CLI**

Create `scripts/audit/a1_preprocessing_parity.py`:

```python
"""A1: Image preprocessing parity between training and inference pipelines.

For each audit sample, load the image through both the training-time
preprocessing path (src.train.train._load_raster_image_tensor) and the
inference-time path (src.inference.decoder_runtime._load_stage_b_crop_tensor),
then compare the resulting tensors element-wise.

Pass criterion: tensors bit-identical OR within fp32 tolerance (max abs
diff <= 1e-6) for every sample. Any larger diff is a finding worth
reporting in the audit.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a1_preprocessing_parity \\
        --manifest data\\processed\\<combined_manifest>.jsonl \\
        --n-per-corpus 5 \\
        --out audit_results\\a1_preprocessing.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True,
                   help="Path to combined training manifest (jsonl)")
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    from scripts.audit._sample_picker import pick_audit_samples
    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} audit samples")

    results = _run_parity(samples, args)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    _print_summary(results)


def _run_parity(samples, args):
    raise NotImplementedError  # filled in next step


def _print_summary(results):
    raise NotImplementedError


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.2: Implement the parity logic**

Replace the two `NotImplementedError` stubs with:

```python
def _run_parity(samples, args):
    import numpy as np
    import torch
    from PIL import Image
    from src.train.train import _load_raster_image_tensor
    from src.inference.decoder_runtime import _load_stage_b_crop_tensor

    per_sample = []
    for sample in samples:
        img_path = _REPO / sample["image_path"]
        if not img_path.exists():
            per_sample.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "status": "missing_image",
                "image_path": str(img_path),
            })
            continue

        # Training-time path. Returns (tensor[1, H, W], content_width).
        train_tensor, train_cw = _load_raster_image_tensor(
            img_path, height=args.image_height, max_width=args.image_max_width
        )

        # Inference-time path. Returns tensor[1, 1, H, W] on a torch device.
        inf_tensor = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=torch.device("cpu"),
        )

        # Normalize shapes for comparison.
        if train_tensor.dim() == 3:
            train_normalized = train_tensor.unsqueeze(0)  # [1, 1, H, W]
        else:
            train_normalized = train_tensor
        if inf_tensor.dim() == 4 and inf_tensor.shape[1] == 1:
            inf_normalized = inf_tensor
        else:
            inf_normalized = inf_tensor

        train_arr = train_normalized.detach().cpu().numpy().astype(np.float32)
        inf_arr = inf_normalized.detach().cpu().numpy().astype(np.float32)

        if train_arr.shape != inf_arr.shape:
            per_sample.append({
                "sample_id": sample["sample_id"],
                "dataset": sample["dataset"],
                "status": "shape_mismatch",
                "train_shape": list(train_arr.shape),
                "inference_shape": list(inf_arr.shape),
            })
            continue

        diff = train_arr - inf_arr
        max_abs = float(np.max(np.abs(diff)))
        mean_abs = float(np.mean(np.abs(diff)))
        nonzero_pixels = int(np.sum(diff != 0))
        total_pixels = int(np.prod(diff.shape))
        per_sample.append({
            "sample_id": sample["sample_id"],
            "dataset": sample["dataset"],
            "status": "compared",
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "nonzero_pixel_fraction": nonzero_pixels / total_pixels,
            "shape": list(train_arr.shape),
        })

    by_status = {}
    for r in per_sample:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1
    overall_max = max(
        (r["max_abs_diff"] for r in per_sample if r["status"] == "compared"),
        default=None,
    )
    pass_criterion = overall_max is not None and overall_max <= 1e-6
    return {
        "experiment": "a1_preprocessing_parity",
        "args": vars_serializable(args),
        "n_samples": len(per_sample),
        "by_status": by_status,
        "overall_max_abs_diff": overall_max,
        "pass": pass_criterion,
        "per_sample": per_sample,
    }


def vars_serializable(args):
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def _print_summary(results):
    print()
    print(f"=== A1: Image preprocessing parity ===")
    print(f"Samples: {results['n_samples']}")
    print(f"Status:  {results['by_status']}")
    print(f"Max abs diff (overall): {results['overall_max_abs_diff']}")
    print(f"PASS: {results['pass']}")
```

- [ ] **Step 2.3: Push to seder and run on 2 samples first to smoke test**

```bash
scp scripts/audit/_sample_picker.py '10.10.1.29:audit_sample_picker.py'
scp scripts/audit/a1_preprocessing_parity.py '10.10.1.29:audit_a1.py'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_sample_picker.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\_sample_picker.py"'
ssh 10.10.1.29 'move /Y "%USERPROFILE%\audit_a1.py" "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO\scripts\audit\a1_preprocessing_parity.py"'
```

Find a manifest path (the audit covers all 4 corpora, so use the combined manifest used for Stage 3 training):

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir data\processed /B | findstr manifest'
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir data\processed\synthetic_v2\manifests /B'
```

Expected: identify the path of the combined manifest used by the Stage 3 trainer. Likely `data\processed\<combined>\manifest.jsonl` or per-corpus manifests. Document the actual manifest in the next step's command.

Run with `--n-per-corpus 2` for smoke:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a1_preprocessing_parity --manifest <RESOLVED_MANIFEST_PATH> --n-per-corpus 2 --out audit_results\a1_smoke.json'
```

Expected: prints `Wrote audit_results\\a1_smoke.json` and a summary with 8 samples and a PASS or FAIL flag. No tracebacks.

- [ ] **Step 2.4: Run full A1 (5 per corpus, 20 total)**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a1_preprocessing_parity --manifest <MANIFEST> --n-per-corpus 5 --out audit_results\a1_preprocessing.json'
```

Pull results locally:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a1_preprocessing.json' /tmp/a1.json
python3 -c "import json; d=json.load(open('/tmp/a1.json')); print('PASS' if d['pass'] else 'FAIL'); print('max abs diff:', d['overall_max_abs_diff']); [print(f\"  {r['dataset']:<20} {r['sample_id'][:40]:<40} {r['status']} max={r.get('max_abs_diff','-')}\") for r in d['per_sample']]"
```

Expected outputs:
- PASS → tensors match, A1 is clean. Continue.
- FAIL → real finding. Note which corpora and how large the diff. Stop and record the finding before moving on (this might be the only bug in training).

- [ ] **Step 2.5: Commit**

```bash
git add scripts/audit/a1_preprocessing_parity.py
git commit -m "feat(audit): A1 image preprocessing parity script

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 3: A2 — Encoder output parity script

**Goal:** Verify that the cached encoder features (used during Stage 3 training) match what the encoder produces live on the same images at inference time.

**Files:**
- Create: `scripts/audit/a2_encoder_parity.py`

- [ ] **Step 3.1: Implement the A2 script**

Create `scripts/audit/a2_encoder_parity.py`:

```python
"""A2: Encoder output parity between cached features and live encoder.

Stage 3 v2 was trained with a frozen encoder; encoder features for every
training sample were cached to disk and loaded during training. At
inference time the encoder runs live on the input image. If these two
feature distributions differ, the decoder learned to translate one and
sees the other at inference — train/eval skew that no amount of decoder
training can fix.

Pass criterion: max abs diff per sample <= 1e-2 (allowing for fp16 vs
fp32 noise). Mean abs diff <= 1e-3. Any systematic bias (mean diff with
consistent sign across samples) is also a finding.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a2_encoder_parity \\
        --manifest <MANIFEST> \\
        --cache-root <CACHE_ROOT> \\
        --cache-hash16 ac8948ae4b5be3e9 \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --n-per-corpus 5 \\
        --out audit_results\\a2_encoder.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--cache-root", type=Path, required=True,
                   help="Root of the encoder cache (e.g. data/cache/...)")
    p.add_argument("--cache-hash16", type=str, required=True,
                   help="16-char cache hash (from Phase 0; e.g. ac8948ae4b5be3e9)")
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import numpy as np
    import torch
    from src.data.encoder_cache import read_cache_entry, cache_entry_exists
    from src.inference.checkpoint_load import load_stage_b_for_inference
    from src.inference.decoder_runtime import _load_stage_b_crop_tensor
    from src.models.radio_stage_b import _encode_staff_image
    from scripts.audit._sample_picker import pick_audit_samples

    device = torch.device("cuda")
    bundle = load_stage_b_for_inference(args.stage_b_ckpt, device, use_fp16=False)

    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} samples; comparing cached vs live encoder features...")

    per_sample = []
    for sample in samples:
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        img_path = _REPO / sample["image_path"]

        if not cache_entry_exists(args.cache_root, args.cache_hash16, dataset, sample_id):
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "cache_miss",
            })
            continue
        if not img_path.exists():
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "missing_image",
            })
            continue

        cached = read_cache_entry(args.cache_root, args.cache_hash16, dataset, sample_id)

        pixel_values = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=device,
        )
        with torch.no_grad():
            live = _encode_staff_image(bundle.decode_model, pixel_values)

        cached_t = cached.to(device=device, dtype=torch.float32) if hasattr(cached, "to") else torch.as_tensor(cached, device=device, dtype=torch.float32)
        live_t = live.to(dtype=torch.float32)

        if cached_t.shape != live_t.shape:
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "shape_mismatch",
                "cached_shape": list(cached_t.shape),
                "live_shape": list(live_t.shape),
            })
            continue

        diff = (cached_t - live_t).detach().cpu().numpy()
        max_abs = float(abs(diff).max())
        mean_abs = float(abs(diff).mean())
        mean_signed = float(diff.mean())

        per_sample.append({
            "sample_id": sample_id, "dataset": dataset,
            "status": "compared",
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "mean_signed_diff": mean_signed,
            "shape": list(cached_t.shape),
        })

    by_status = {}
    for r in per_sample:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1

    compared = [r for r in per_sample if r["status"] == "compared"]
    overall_max = max((r["max_abs_diff"] for r in compared), default=None)
    overall_mean = sum(r["mean_abs_diff"] for r in compared) / len(compared) if compared else None

    pass_criterion = (
        overall_max is not None
        and overall_max <= 1e-2
        and overall_mean is not None
        and overall_mean <= 1e-3
    )

    results = {
        "experiment": "a2_encoder_parity",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_samples": len(per_sample),
        "by_status": by_status,
        "overall_max_abs_diff": overall_max,
        "overall_mean_abs_diff": overall_mean,
        "pass": pass_criterion,
        "per_sample": per_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"=== A2: Encoder output parity ===")
    print(f"Samples: {len(per_sample)}, status: {by_status}")
    print(f"Max abs diff: {overall_max}")
    print(f"Mean abs diff: {overall_mean}")
    print(f"PASS: {pass_criterion}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Resolve cache root and manifest path on seder**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && dir data\cache /B 2>nul'
```

Expected: directories matching the cache hash `ac8948ae4b5be3e9` (16 chars). Record the cache root path.

If the manifest path used in Task 2 was a per-corpus manifest, find the combined one used by Stage 3 training:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && type configs\stage3_radio_systems_v2.yaml 2>nul | findstr /I manifest cache'
```

(Adjust the config filename based on what exists; likely `stage3_radio_systems*.yaml` or similar. Look for the `cached_data_ratio`, `cache_root`, `cache_hash16` keys that the trainer reads.)

- [ ] **Step 3.3: Smoke-run A2 on 2 samples per corpus**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a2_encoder_parity --manifest <MANIFEST> --cache-root <CACHE_ROOT> --cache-hash16 ac8948ae4b5be3e9 --stage-b-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 2 --out audit_results\a2_smoke.json'
```

Expected: 8 samples compared, prints PASS or FAIL with max/mean diffs. No tracebacks.

- [ ] **Step 3.4: Run full A2 (5 per corpus)**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a2_encoder_parity --manifest <MANIFEST> --cache-root <CACHE_ROOT> --cache-hash16 ac8948ae4b5be3e9 --stage-b-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results\a2_encoder.json'
```

Pull and inspect:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a2_encoder.json' /tmp/a2.json
python3 -c "import json; d=json.load(open('/tmp/a2.json')); print('PASS' if d['pass'] else 'FAIL'); print('max:', d['overall_max_abs_diff'], 'mean:', d['overall_mean_abs_diff']); [print(f\"  {r['dataset']:<20} {r['sample_id'][:40]:<40} {r['status']} max={r.get('max_abs_diff','-')}\") for r in d['per_sample']]"
```

Expected outputs:
- PASS → cached features and live encoder agree. The encoder cache isn't the bug.
- FAIL → real finding. **This is the leading hypothesis** based on the LANCZOS vs BILINEAR mismatch noticed during plan-writing. If FAIL, the audit recommendation will likely include either "rebuild encoder cache with the inference-time preprocessing" or "switch training to live encoder".

- [ ] **Step 3.5: Commit**

```bash
git add scripts/audit/a2_encoder_parity.py
git commit -m "feat(audit): A2 encoder output parity (cached vs live)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 4: A3 — Decoder-on-training-data script

**Goal:** End-to-end inference on training samples should approximately reproduce their labels. If token accuracy on training data is low, the model didn't converge or something is broken between training and inference.

**Files:**
- Create: `scripts/audit/a3_decoder_on_training.py`

- [ ] **Step 4.1: Implement the A3 script**

Create `scripts/audit/a3_decoder_on_training.py`:

```python
"""A3: Decoder behavior on training data.

The model has been trained on these exact samples (with whatever
augmentation was active). At inference time, on the same images, the
predicted token sequence should closely match the ground-truth labels.

Pass criteria (triage thresholds, not certification):
  - token accuracy >= 80% averaged across samples
  - exact-match sequence rate >= 30%
  - per-class accuracy for time-sig tokens >= 80%
  - per-class accuracy for key-sig tokens >= 80%

Lower numbers indicate the model didn't memorize its training data
(unusual for a typical autoregressive transformer with enough capacity),
which would suggest either training didn't converge or there's a
preprocessing skew that A1/A2 didn't catch.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.a3_decoder_on_training \\
        --manifest <MANIFEST> \\
        --stage-b-ckpt checkpoints\\full_radio_stage3_v2\\stage3-radio-systems-frozen-encoder_best.pt \\
        --n-per-corpus 5 \\
        --out audit_results\\a3_decoder.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from collections import Counter

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def _token_accuracy(predicted: list, target: list) -> float:
    """Per-position accuracy over min(len(predicted), len(target)).

    Padding predicted to target length is not done; we only score
    positions both sequences have. This is conservative: it counts a
    short prediction as 100% accurate on its prefix, but exact-match
    catches sequences that should have been longer.
    """
    n = min(len(predicted), len(target))
    if n == 0:
        return 0.0
    correct = sum(1 for a, b in zip(predicted[:n], target[:n]) if a == b)
    return correct / n


def _per_class_accuracy(predicted: list, target: list, prefix: str) -> tuple[int, int]:
    """Returns (correct, total) for tokens whose target starts with prefix."""
    correct, total = 0, 0
    n = min(len(predicted), len(target))
    for i in range(n):
        if target[i].startswith(prefix):
            total += 1
            if predicted[i] == target[i]:
                correct += 1
    return correct, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--stage-b-ckpt", type=Path, required=True)
    p.add_argument("--yolo-weights", type=Path, default=None,
                   help="Optional. If provided, runs Stage A on the staff image source page; otherwise treats the training image as a single pre-cropped staff/system.")
    p.add_argument("--image-height", type=int, default=250)
    p.add_argument("--image-max-width", type=int, default=2500)
    p.add_argument("--max-decode-steps", type=int, default=2048)
    p.add_argument("--n-per-corpus", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import torch
    from src.inference.checkpoint_load import load_stage_b_for_inference
    from src.inference.decoder_runtime import _load_stage_b_crop_tensor
    from src.inference.system_pipeline import _encode_staff_image, _decode_stage_b_tokens
    from scripts.audit._sample_picker import pick_audit_samples

    device = torch.device("cuda")
    bundle = load_stage_b_for_inference(args.stage_b_ckpt, device, use_fp16=False)

    samples = pick_audit_samples(args.manifest, n_per_corpus=args.n_per_corpus, seed=args.seed)
    print(f"Selected {len(samples)} samples; running decoder on each...")

    per_sample = []
    for sample in samples:
        sample_id = sample["sample_id"]
        dataset = sample["dataset"]
        img_path = _REPO / sample["image_path"]
        target = sample.get("token_sequence", [])

        if not img_path.exists():
            per_sample.append({
                "sample_id": sample_id, "dataset": dataset,
                "status": "missing_image",
            })
            continue

        pixel_values = _load_stage_b_crop_tensor(
            img_path,
            image_height=args.image_height,
            image_max_width=args.image_max_width,
            device=device,
        )
        with torch.no_grad():
            memory = _encode_staff_image(bundle.decode_model, pixel_values)
            predicted = _decode_stage_b_tokens(
                model=bundle.model,
                pixel_values=pixel_values,
                vocabulary=bundle.vocab,
                beam_width=1,
                max_decode_steps=args.max_decode_steps,
                length_penalty_alpha=0.4,
                _precomputed={
                    "decode_model": bundle.decode_model,
                    "memory": memory,
                    "token_to_idx": bundle.token_to_idx,
                    "use_fp16": False,
                },
            )

        token_acc = _token_accuracy(predicted, target)
        exact_match = predicted == target
        ts_c, ts_n = _per_class_accuracy(predicted, target, "timeSignature-")
        ks_c, ks_n = _per_class_accuracy(predicted, target, "keySignature-")

        per_sample.append({
            "sample_id": sample_id, "dataset": dataset,
            "status": "compared",
            "predicted_len": len(predicted),
            "target_len": len(target),
            "token_accuracy": token_acc,
            "exact_match": exact_match,
            "timeSig_correct": ts_c, "timeSig_total": ts_n,
            "keySig_correct": ks_c, "keySig_total": ks_n,
            "predicted_first_50": predicted[:50],
            "target_first_50": target[:50],
        })

    compared = [r for r in per_sample if r["status"] == "compared"]
    n_compared = len(compared) or 1

    mean_token_acc = sum(r["token_accuracy"] for r in compared) / n_compared
    exact_match_rate = sum(1 for r in compared if r["exact_match"]) / n_compared

    ts_c_total = sum(r["timeSig_correct"] for r in compared)
    ts_n_total = sum(r["timeSig_total"] for r in compared)
    ks_c_total = sum(r["keySig_correct"] for r in compared)
    ks_n_total = sum(r["keySig_total"] for r in compared)
    ts_acc = ts_c_total / ts_n_total if ts_n_total else None
    ks_acc = ks_c_total / ks_n_total if ks_n_total else None

    pass_criterion = (
        mean_token_acc >= 0.80
        and exact_match_rate >= 0.30
        and (ts_acc is None or ts_acc >= 0.80)
        and (ks_acc is None or ks_acc >= 0.80)
    )

    results = {
        "experiment": "a3_decoder_on_training",
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "n_samples": len(per_sample),
        "mean_token_accuracy": mean_token_acc,
        "exact_match_rate": exact_match_rate,
        "timeSig_accuracy": ts_acc,
        "keySig_accuracy": ks_acc,
        "pass": pass_criterion,
        "per_sample": per_sample,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print()
    print(f"=== A3: Decoder on training data ===")
    print(f"Samples compared: {n_compared}")
    print(f"Mean token accuracy: {mean_token_acc:.3f}")
    print(f"Exact match rate:    {exact_match_rate:.3f}")
    print(f"timeSig accuracy:    {ts_acc}")
    print(f"keySig accuracy:     {ks_acc}")
    print(f"PASS: {pass_criterion}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Smoke-run A3 on 2 samples per corpus**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a3_decoder_on_training --manifest <MANIFEST> --stage-b-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 2 --out audit_results\a3_smoke.json'
```

Expected: 8 samples, prints aggregate accuracy numbers and PASS/FAIL.

- [ ] **Step 4.3: Run full A3 (5 per corpus)**

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -m scripts.audit.a3_decoder_on_training --manifest <MANIFEST> --stage-b-ckpt checkpoints\full_radio_stage3_v2\stage3-radio-systems-frozen-encoder_best.pt --n-per-corpus 5 --out audit_results\a3_decoder.json'
```

Pull and inspect:

```bash
scp '10.10.1.29:Clarity-OMR-Train-RADIO/audit_results/a3_decoder.json' /tmp/a3.json
python3 -c "import json; d=json.load(open('/tmp/a3.json')); print('PASS' if d['pass'] else 'FAIL'); print('tok_acc:', d['mean_token_accuracy'], 'exact:', d['exact_match_rate'], 'ts:', d['timeSig_accuracy'], 'ks:', d['keySig_accuracy'])"
```

Expected outputs:
- High token accuracy (>= 90%) → model learned its training data well; gap is generalization
- Mid token accuracy (50-80%) → model partially learned; might be heavy augmentation or partial convergence
- Low token accuracy (< 50%) → model did NOT learn its training data, foundation is broken

- [ ] **Step 4.4: Commit**

```bash
git add scripts/audit/a3_decoder_on_training.py
git commit -m "feat(audit): A3 decoder-on-training round-trip script

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 5: Phase A decision gate

**Goal:** Synthesize A1/A2/A3 results and decide whether to proceed to Phase B.

- [ ] **Step 5.1: Write a one-paragraph Phase A summary**

In a scratch markdown buffer (will become part of the final report at Task 10), write a short paragraph for each sub-experiment:

```markdown
### Phase A summary

**A1 — Image preprocessing parity:** <PASS|FAIL>. Max abs diff: <NUMBER>.
<one-line interpretation>

**A2 — Encoder output parity:** <PASS|FAIL>. Max abs diff: <NUMBER>,
mean abs diff: <NUMBER>. <one-line interpretation>

**A3 — Decoder on training data:** <PASS|FAIL>. Mean token accuracy:
<NUMBER>, exact-match rate: <NUMBER>, timeSig accuracy: <NUMBER>,
keySig accuracy: <NUMBER>. <one-line interpretation>
```

- [ ] **Step 5.2: Decide on Phase B**

Apply the decision rule from the spec:

- All three Phase A experiments PASS → proceed to Phase B (Task 9).
- Any Phase A experiment FAILS → STOP. Record the finding in the report (Task 10) with a clear recommendation. Do NOT proceed to Phase B yet. The failure itself likely explains the eval gap; fix-and-verify-loop is the appropriate next step before more auditing.

Mark a note in the report: which path was taken and why.

---

## Task 6 (conditional, only if Phase A passes): Phase B component audit

**Files:** none new; this is a code-review pass.

This task is structured as a checklist. For each component, read the named file(s), check against the criterion, and record findings (file path + line number + observation) in the report (Task 10). No code changes during this task — findings become part of the audit recommendation.

- [ ] **Step 6.1: Dataloader audit**

Read `src/train/train.py` `StageBDataset` class (line ~574) and `build_stage_b_sampler` (line ~843). Verify:
- `num_workers > 0` in DataLoader construction (find the DataLoader instantiation; likely later in the file)
- `pin_memory=True`
- `persistent_workers=True`
- `shuffle=True` for train sampler, `shuffle=False` for val
- Worker init function sets per-worker seed (existing function `stage_b_worker_init_fn` at line 979 — verify it's actually used)
- Train/val splits are read from `split` field of manifest entries — verify no entry can appear in both

Record findings.

- [ ] **Step 6.2: Augmentation audit**

Read `src/train/train.py` `_apply_online_augmentations` (line 511) and `_load_entry_image_tensor` (line 476). Verify:
- Augmentations only applied on train, not val (look for a `is_train` or `split=='train'` gate)
- Augmentations do NOT crop in ways that hide the time-sig glyph at the start of staff images
- Augmentation random seed handled correctly (worker-local RNG, not global)

Record findings.

- [ ] **Step 6.3: Model architecture audit**

Read `src/models/radio_stage_b.py`. Verify:
- Causal mask applied in decoder self-attention (look for `mask_type="causal"` or triangular mask construction)
- Positional encoding present and correct (sinusoidal or learned)
- Vocab size used by embedding layer matches output projection layer
- Encoder→decoder cross-attention wires through the right path (memory tensor shape consistent between training and inference)

Cross-reference with `_encode_staff_image` (in the same file or wherever it's defined) — same module used by both train and inference?

Record findings.

- [ ] **Step 6.4: Loss function audit**

Find the loss computation in `src/train/train.py` (search for `cross_entropy`, `loss =`, or similar; likely in the train loop body). Verify:
- Standard `torch.nn.functional.cross_entropy` with `ignore_index` set to padding token id
- No accidental weighting that down-weights rare tokens (no `weight=` argument to CE unless deliberate)
- Label smoothing usage: if applied, value matches design intent (typically 0.0 or 0.1)
- Loss reduction (mean vs sum) consistent between train and val

Record findings with specific line numbers.

- [ ] **Step 6.5: Optimizer + schedule audit**

Read `_build_optimizer` (line 1004) and `_build_scheduler` (line 1102) in `src/train/train.py`. Verify against PyTorch tuning guide standards:
- AdamW with betas (0.9, 0.999) and eps 1e-8 (or documented deviation)
- Weight decay in 0.01-0.1 range (transformer-appropriate)
- Per-parameter-group weight decay (no decay on biases or LayerNorm params) — check if implemented
- Peak LR within 1e-4 to 5e-4 for decoder fine-tune (or documented deviation)
- Warmup followed by cosine or linear decay
- End-of-training LR not 0 (final LR is typically peak / 10 or peak / 100)

Record findings.

- [ ] **Step 6.6: Gradient handling audit**

Find gradient handling around `loss.backward()` in the train loop. Verify:
- Gradient clipping with `clip_grad_norm_` (typical `max_norm=1.0`)
- Gradient accumulation correctly scales loss (`loss / accum_steps` before backward)
- Training logs from past runs don't contain NaN/Inf warnings (`grep -i nan training_logs/...` on seder if logs are present)

Record findings.

- [ ] **Step 6.7: Mixed precision audit**

Search `src/train/train.py` for `autocast`, `GradScaler`, `fp16`, `bfloat16`. Verify:
- If FP16 in use: `GradScaler` properly wraps optimizer; `scale_loss` and `step` called correctly
- If BF16 in use: no GradScaler needed (BF16 has the same dynamic range as FP32)
- Inference dtype matches training dtype (no FP16 ↔ BF16 mismatch)

Record findings.

- [ ] **Step 6.8: Checkpoint integrity audit**

On seder:

```bash
ssh 10.10.1.29 'cd /D "C:\Users\Jonathan Wesely\Clarity-OMR-Train-RADIO" && venv-cu132\Scripts\python.exe -c "import torch; ckpt=torch.load(r''checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt'', map_location=''cpu'', weights_only=False); print(''keys:'', list(ckpt.keys())[:10] if isinstance(ckpt, dict) else type(ckpt)); print(''step:'', ckpt.get(''step'') if isinstance(ckpt, dict) else ''N/A''); print(''val_loss:'', ckpt.get(''val_loss'') if isinstance(ckpt, dict) else ''N/A'')"'
```

Verify:
- `step` corresponds to the claimed best (4000 per memory)
- `val_loss` corresponds to the claimed value (0.148)
- Loading the model state via `load_stage_b_for_inference` covers all weight keys (the script we already run prints "99.4% coverage, missing=0, unexpected=8" — unexpected keys are a yellow flag worth understanding)

Record findings about the unexpected keys (`positional_bridge._orig_mod.original_module.*`) — are they redundant copies from PEFT/DoRA wrapping, or actually missing functionality?

- [ ] **Step 6.9: Validation loop audit**

Find the validation step in the train loop. Verify:
- `model.eval()` set during val
- `torch.no_grad()` wrapping
- Same loss function as train (not a different metric)
- val data drawn from `split == "val"` only
- No augmentation applied during val
- val data not overlapping with train (`sample_id` disjoint between train and val splits in the manifest)

Record findings.

- [ ] **Step 6.10: Commit Phase B findings**

(No code commit, but commit the running notes to the audit report draft.)

```bash
git add docs/audits/2026-05-11-stage3-v2-training-audit.md
git commit -m "audit: Phase B component findings (WIP)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Task 7: Final audit report

**Files:**
- Create or finalize: `docs/audits/2026-05-11-stage3-v2-training-audit.md`

- [ ] **Step 7.1: Write the report shell**

Create `docs/audits/2026-05-11-stage3-v2-training-audit.md`:

```markdown
# Stage 3 v2 Training Audit Report

**Date:** 2026-05-11
**Plan:** [docs/superpowers/plans/2026-05-11-stage3-v2-training-audit-plan.md](../superpowers/plans/2026-05-11-stage3-v2-training-audit-plan.md)
**Spec:** [docs/superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md](../superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md)
**Checkpoint audited:** `checkpoints/full_radio_stage3_v2/stage3-radio-systems-frozen-encoder_best.pt` (step 4000, val_loss 0.148)
**Branch:** `audit/stage3-v2-training`

## TL;DR

<one-paragraph summary: bug found / no bug found; recommendation>

## Phase A: Round-trip data verification

### A1 — Image preprocessing parity

**Result:** <PASS / FAIL>
**Evidence:** `audit_results/a1_preprocessing.json` — max abs diff <NUMBER> across <N> samples.

<one-paragraph interpretation; if FAIL, identify the line(s) of code responsible>

### A2 — Encoder output parity (cached vs live)

**Result:** <PASS / FAIL>
**Evidence:** `audit_results/a2_encoder.json` — max abs diff <NUMBER>, mean abs diff <NUMBER>.

<one-paragraph interpretation. If FAIL, flag the LANCZOS-vs-BILINEAR hypothesis: `scripts/build_encoder_cache.py:93` uses LANCZOS while `src/inference/decoder_runtime.py:31` uses BILINEAR; if our cache was built with LANCZOS and inference uses BILINEAR, the decoder learned to translate one encoder distribution and sees a different one at inference.>

### A3 — Decoder on training data

**Result:** <PASS / FAIL>
**Evidence:** `audit_results/a3_decoder.json` — mean token accuracy <X>, exact-match rate <Y>, timeSig accuracy <Z>, keySig accuracy <W>.

<one-paragraph interpretation. Compare per-corpus accuracy; if one corpus (e.g., synthetic_v2) is much higher than another (e.g., primus), note the corpus-level bias.>

## Phase B: Process audit

<Only filled if Phase A passed cleanly. Otherwise note "Not run — Phase A surfaced finding X; we recommend fix-and-verify before pursuing Phase B".>

### B1 Dataloader
### B2 Augmentation
### B3 Model architecture
### B4 Loss
### B5 Optimizer + schedule
### B6 Gradient handling
### B7 Mixed precision
### B8 Checkpoint integrity
### B9 Validation loop

<For each: PASS / FINDING / N/A, with file:line references and one-paragraph interpretation>

## Bug list (sorted by likely impact)

1. **[HIGH/MEDIUM/LOW]** <bug name> — <one-sentence description with file:line ref>
2. ...

## Recommendation

<Exactly one of:>

- **Foundation is sound, gap is genuine generalization failure.** Proceed to retraining-strategy sub-project (rebalanced sampling, aux head, larger image height, etc.).
- **Foundation has bug X; fix and re-evaluate before any retraining decision.** Specific fix: <description>. Re-run `eval/run_clarity_demo_radio_eval.py` after fix to see if scores rise meaningfully without retraining.
- **Foundation has bug X requiring retraining to validate.** Specific fix: <description>. Then re-train Stage 3 with the corrected pipeline and re-evaluate against demo + lieder corpora.

## Out of scope

- Architecture comparison (DaViT vs RADIO) — separate parallel thread
- Audiveris benchmark — separate parallel thread
- Stage 1 / Stage 2 training audit — would expand to these only if Phase A surfaced upstream evidence (and the audit did/did not)
```

- [ ] **Step 7.2: Fill in the actual numbers from Phase A and (if run) Phase B**

Replace each `<...>` placeholder with the real findings. Make the TL;DR concrete: name the bug (if any), name the recommendation, give the supporting number.

- [ ] **Step 7.3: Commit the report and open PR**

```bash
git add docs/audits/2026-05-11-stage3-v2-training-audit.md
git commit -m "audit: Stage 3 v2 training audit final report

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
git push -u origin audit/stage3-v2-training

gh pr create --title "audit: Stage 3 v2 training audit + scripts" --body "$(cat <<'EOF'
## Summary

Adds the Stage 3 v2 training audit per spec
`docs/superpowers/specs/2026-05-11-stage3-v2-training-audit-design.md`:

- `scripts/audit/_sample_picker.py` — stratified sample selection across 4 corpora
- `scripts/audit/a1_preprocessing_parity.py` — train vs inference image-tensor diff
- `scripts/audit/a2_encoder_parity.py` — cached vs live encoder feature diff
- `scripts/audit/a3_decoder_on_training.py` — end-to-end inference round-trip on training data
- `docs/audits/2026-05-11-stage3-v2-training-audit.md` — final report

## Result

<Replace with TL;DR from report>

## Test plan

- [ ] Reviewer: confirm seder pytest runs `tests/audit/test_sample_picker.py` (2 tests, pure-python, not CUDA-gated)
- [ ] Reviewer: spot-check the audit_results/*.json files attached or referenced in the report match the report's claims

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist

- [x] **Spec coverage:** Each section of the spec maps to tasks. Phase A → Tasks 2-4 + 5; Phase B → Task 6; Deliverable → Task 7.
- [x] **Placeholder scan:** Real placeholders only exist where the audit results dictate the content (the `<...>` markers in Task 7's report template). All code blocks contain complete code.
- [x] **Type consistency:** `pick_audit_samples`, `_load_stage_b_crop_tensor`, `_load_raster_image_tensor`, `_encode_staff_image`, `_decode_stage_b_tokens` — names match the actual code in src/. The `bundle.decode_model` / `bundle.model` / `bundle.vocab` / `bundle.token_to_idx` accessors match the `StageBInferenceBundle` shape seen in `src/inference/system_pipeline.py:155-185`.
- [x] **Path resolution:** Manifest path and cache root are explicitly named as `<MANIFEST>` and `<CACHE_ROOT>` placeholders to be resolved at execution time (Tasks 2.3 and 3.2 contain the discovery commands). This is intentional — the actual paths depend on what's on seder and shouldn't be hard-coded into the plan.
