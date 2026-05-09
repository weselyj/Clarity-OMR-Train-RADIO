#!/usr/bin/env python3
"""Pre-flight ready check for Stage 3 Phase 1 training launch.

Verifies on the local clone (or via SSH on the GPU box, depending on argv):
1. ``configs/train_stage3_radio_systems.yaml`` parses and tier fields validate.
2. ``data/cache/encoder/<hash16>/`` exists and contains the expected sample count.
3. ``src/data/manifests/token_manifest_stage3.jsonl`` exists and row count matches.
4. ``checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt`` exists.
5. (Optional, ``--dry-run``) Trainer dry-run mode runs for 10 opt-steps without
   raising and writes a step-log.

Exit 0 = ready. Exit 1 = at least one prerequisite missing.

Note: the script's intended runtime is the GPU box. Importing
``src.train.train`` requires torch and the rest of the trainer's deps. Running
on a torch-less workstation will fail at import time; that is expected.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_stage3_radio_systems.yaml")
    ap.add_argument("--manifest", default="src/data/manifests/token_manifest_stage3.jsonl")
    ap.add_argument("--cache-root", default="data/cache/encoder")
    ap.add_argument(
        "--init-ckpt",
        default="checkpoints/full_radio_stage2_systems_v2/stage2-radio-systems-polyphonic_best.pt",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Run trainer in dry-run mode for 10 opt-steps",
    )
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    # Ensure project root is on sys.path so `src.train.train` imports cleanly
    # regardless of the cwd from which this script is invoked.
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    fails: list[str] = []
    cfg = None

    # 1. Config parses and tier fields validate.
    from src.train.train import load_stage_config

    config_path = (project_root / args.config).resolve()
    if not config_path.exists():
        fails.append(f"config not found: {config_path}")
    else:
        try:
            cfg = load_stage_config(config_path)
            if not cfg.tier_grouped_sampling:
                fails.append(
                    f"config {config_path} does not have tier_grouped_sampling=true"
                )
            print(
                f"[preflight] config OK: tier_grouped_sampling=True, "
                f"b_cached={cfg.b_cached}, b_live={cfg.b_live}"
            )
        except Exception as exc:
            fails.append(f"config parse error: {exc}")

    # 2. Cache exists.
    if cfg is not None:
        cache_dir = (project_root / args.cache_root / cfg.cache_hash16).resolve()
        if not cache_dir.exists():
            fails.append(f"cache dir not found: {cache_dir}")
        else:
            metadata_path = cache_dir / "metadata.json"
            if metadata_path.exists():
                meta = json.loads(metadata_path.read_text())
                samples = meta.get("samples_processed", 0)
                print(f"[preflight] cache OK: {samples} samples at {cache_dir}")
                if samples != 215985:
                    fails.append(
                        f"cache sample count {samples} != expected 215985"
                    )
            else:
                fails.append(f"cache metadata.json not found at {metadata_path}")

    # 3. Manifest.
    manifest_path = (project_root / args.manifest).resolve()
    if not manifest_path.exists():
        fails.append(f"manifest not found: {manifest_path}")
    else:
        with manifest_path.open() as fh:
            n_rows = sum(1 for line in fh if line.strip())
        print(f"[preflight] manifest OK: {n_rows} rows at {manifest_path}")
        if n_rows != 303663:
            fails.append(
                f"manifest row count {n_rows} != expected 303663 "
                "(combined Stage 3 manifest)"
            )

    # 4. Init checkpoint.
    init_ckpt = (project_root / args.init_ckpt).resolve()
    if not init_ckpt.exists():
        fails.append(f"init checkpoint not found: {init_ckpt}")
    else:
        print(f"[preflight] init ckpt OK: {init_ckpt}")

    # 5. Optional dry-run.
    if args.dry_run and not fails:
        print("[preflight] launching trainer dry-run (10 opt-steps)...")
        import subprocess

        result = subprocess.run(
            [
                sys.executable,
                "src/train/train.py",
                "--stage-configs",
                str(args.config),
                "--mode",
                "dry-run",
                "--max-steps-per-stage",
                "10",
                "--token-manifest",
                str(args.manifest),
                "--resume-checkpoint",
                str(args.init_ckpt),
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            fails.append(f"dry-run failed: stderr=\n{result.stderr[-2000:]}")
        else:
            print("[preflight] dry-run OK")

    if fails:
        print("\n[preflight] FAIL — prerequisites missing:")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("\n[preflight] READY — all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
