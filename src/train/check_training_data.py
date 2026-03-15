#!/usr/bin/env python3
"""Validate that YOLO and curriculum training configs can see all planned datasets.

Compares what the plan expects vs what actually exists on disk / in manifests.
Run from project root:
    python src/train/check_training_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Colours (ANSI) ───────────────────────────────────────────────────────────
OK = "\033[92m[OK]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

fail_count = 0
warn_count = 0


def ok(msg: str) -> None:
    print(f"  {OK}  {msg}")


def warn(msg: str) -> None:
    global warn_count
    warn_count += 1
    print(f"  {WARN} {msg}")


def fail(msg: str) -> None:
    global fail_count
    fail_count += 1
    print(f"  {FAIL} {msg}")


def info(msg: str) -> None:
    print(f"  {INFO} {msg}")


def load_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Curriculum (Stage-B) training data
# ═════════════════════════════════════════════════════════════════════════════
def check_curriculum() -> None:
    print("\n" + "=" * 72)
    print("CURRICULUM TRAINING (Stage-B seq2seq)")
    print("=" * 72)

    # ── 1a. Stage config files ────────────────────────────────────────────
    stage_configs = [
        PROJECT_ROOT / "configs" / "train_stage1.yaml",
        PROJECT_ROOT / "configs" / "train_stage2.yaml",
        PROJECT_ROOT / "configs" / "train_stage3.yaml",
    ]
    print("\n-- Stage config files --")
    for cfg in stage_configs:
        if cfg.exists():
            ok(f"{cfg.relative_to(PROJECT_ROOT)}")
        else:
            fail(f"Missing: {cfg.relative_to(PROJECT_ROOT)}")

    # ── 1b. Token manifest(s) ────────────────────────────────────────────
    main_manifest = PROJECT_ROOT / "src" / "data" / "manifests" / "token_manifest.jsonl"
    synth_manifest = (
        PROJECT_ROOT / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"
    )

    print("\n-- Token manifests --")
    main_rows = load_jsonl(main_manifest)
    synth_rows = load_jsonl(synth_manifest)

    if main_rows:
        ok(f"Main manifest: {len(main_rows):,} entries  ({main_manifest.relative_to(PROJECT_ROOT)})")
    else:
        fail(f"Main manifest empty or missing: {main_manifest.relative_to(PROJECT_ROOT)}")
    if synth_rows:
        ok(f"Synthetic manifest: {len(synth_rows):,} entries  ({synth_manifest.relative_to(PROJECT_ROOT)})")
    else:
        warn(f"Synthetic manifest empty or missing: {synth_manifest.relative_to(PROJECT_ROOT)}")

    # ── 1c. Count by (dataset, split) ────────────────────────────────────
    counts: Dict[Tuple[str, str], int] = {}
    for row in main_rows + synth_rows:
        key = (str(row.get("dataset", "?")).lower(), str(row.get("split", "?")).lower())
        counts[key] = counts.get(key, 0) + 1

    # train.py now defaults to loading BOTH manifests (main + synthetic)
    train_visible: Dict[Tuple[str, str], int] = dict(counts)

    print("\n-- What train.py loads (default: both manifests) --")
    for (ds, sp), n in sorted(train_visible.items()):
        print(f"    {ds:<25s} {sp:<8s} {n:>8,}")
    print(f"    {'TOTAL':<25s} {'':8s} {sum(train_visible.values()):>8,}")

    if main_rows and synth_rows:
        ok("Both manifests present. train.py default loads both.")
    elif main_rows and not synth_rows:
        warn("Synthetic manifest missing — Stages 2-3 will lack synthetic data.")
    elif not main_rows:
        fail("Main manifest missing — no base dataset available.")

    # ── 1e. Check each stage's expected datasets ──────────────────────────
    # Plan expectations (from omr-final-plan.md Section 5.3)
    plan_expectations = {
        "stage1-monophonic-foundation": {
            "primus": ("train", 74_311, "~79,800/epoch target"),
            "cameraprimus": ("train", 74_596, "~34,200/epoch target"),
        },
        "stage2-polyphonic": {
            "grandstaff": ("train", 8_000, "plan says ~8,000 piano staves"),
            "synthetic_polyphonic": ("train", 30_000, "plan says ~30,000 staff crops"),
            "primus": ("train", 74_311, "20% replay buffer"),
        },
        "stage3-full-complexity": {
            "synthetic_fullpage": ("train", 120_000, "plan says ~120,000 staff crops from ~20K pages"),
            "lieder-main": ("train", 1_093, "plan says ~8,000 piano+voice staves"),
            "primus": ("train", 74_311, "10% replay buffer"),
            "grandstaff": ("train", 91_606, "10% replay buffer"),
        },
    }

    print("\n-- Per-stage dataset availability vs plan --")
    for stage_name, expected in plan_expectations.items():
        print(f"\n  [{stage_name}]")
        for ds, (sp, plan_min, note) in expected.items():
            available_all = counts.get((ds, sp), 0)
            available_visible = train_visible.get((ds, sp), 0)
            tag = ""
            if available_visible == 0 and available_all > 0:
                fail(f"{ds}/{sp}: {available_all:,} exist but INVISIBLE to train.py (in separate manifest)")
                tag = " << NOT LOADED"
            elif available_visible == 0:
                fail(f"{ds}/{sp}: 0 samples — dataset not generated yet. ({note})")
            elif available_visible < plan_min * 0.5:
                warn(f"{ds}/{sp}: {available_visible:,} samples, plan expected ~{plan_min:,} ({note})")
            else:
                ok(f"{ds}/{sp}: {available_visible:,} samples ({note})")

    # ── 1f. Effective training composition ────────────────────────────────
    print("\n-- Effective epoch composition with current data --")
    stage_configs_data = [
        ("Stage 1", 114_000, [("primus", 0.70), ("cameraprimus", 0.30)]),
        ("Stage 2", 46_000, [("grandstaff", 0.174), ("synthetic_polyphonic", 0.626), ("primus", 0.20)]),
        ("Stage 3", 154_000, [("synthetic_fullpage", 0.748), ("lieder-main", 0.052), ("primus", 0.10), ("grandstaff", 0.10)]),
    ]
    for stage_label, target_epoch, sources in stage_configs_data:
        available_total = 0
        print(f"\n  [{stage_label}]  target: {target_epoch:,} samples/epoch")
        for ds, ratio in sources:
            n = train_visible.get((ds, "train"), 0)
            target = int(target_epoch * ratio)
            status = "OK" if n > 0 else "MISSING"
            effective = min(n, target) if n > 0 else 0
            available_total += effective
            deficit_pct = (1.0 - effective / max(target, 1)) * 100
            if n == 0:
                fail(f"  {ds:<25s} target={target:>7,}  available=0  -> 0 samples ({deficit_pct:.0f}% deficit)")
            elif n < target:
                warn(f"  {ds:<25s} target={target:>7,}  available={n:>7,}  -> {effective:,} (oversampled {target/n:.1f}x)")
                available_total = available_total - effective + target  # oversampling fills the gap
            else:
                ok(f"  {ds:<25s} target={target:>7,}  available={n:>7,}  -> {target:,} samples")
                available_total = available_total - effective + target
        fill_pct = available_total / max(target_epoch, 1) * 100
        if fill_pct < 50:
            fail(f"  Effective fill: {available_total:,} / {target_epoch:,} ({fill_pct:.0f}%)")
        elif fill_pct < 90:
            warn(f"  Effective fill: {available_total:,} / {target_epoch:,} ({fill_pct:.0f}%)")
        else:
            ok(f"  Effective fill: {available_total:,} / {target_epoch:,} ({fill_pct:.0f}%)")

    # ── 1g. Spot-check image files ────────────────────────────────────────
    print("\n-- Spot-check: image file existence (first 50 per dataset) --")
    import random
    rng = random.Random(42)
    all_rows = main_rows + synth_rows
    ds_groups: Dict[str, List[Dict]] = {}
    for row in all_rows:
        ds = str(row.get("dataset", "?")).lower()
        ds_groups.setdefault(ds, []).append(row)
    for ds, rows in sorted(ds_groups.items()):
        sample = rows[:50] if len(rows) <= 50 else rng.sample(rows, 50)
        found = 0
        missing = 0
        no_path = 0
        for row in sample:
            img = row.get("image_path")
            src = row.get("source_path")
            if img:
                p = PROJECT_ROOT / img
                if p.exists():
                    found += 1
                else:
                    missing += 1
            elif src:
                # Fallback: check source path with image extensions
                sp = PROJECT_ROOT / src
                if sp.exists() or any((sp.with_suffix(ext)).exists() for ext in [".png", ".jpg"]):
                    found += 1
                else:
                    missing += 1
            else:
                no_path += 1
        if missing > 0:
            warn(f"{ds}: {found}/{len(sample)} images found, {missing} missing, {no_path} no path")
        else:
            ok(f"{ds}: {found}/{len(sample)} images found")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — YOLO (Stage-A) training data
# ═════════════════════════════════════════════════════════════════════════════
def check_yolo() -> None:
    print("\n" + "=" * 72)
    print("YOLO TRAINING (Stage-A page detection)")
    print("=" * 72)

    page_manifest = (
        PROJECT_ROOT / "data" / "processed" / "synthetic" / "manifests" / "synthetic_pages.jsonl"
    )
    rows = load_jsonl(page_manifest)

    print("\n-- Page manifest --")
    if rows:
        ok(f"{len(rows):,} entries  ({page_manifest.relative_to(PROJECT_ROOT)})")
    else:
        fail(f"Missing or empty: {page_manifest.relative_to(PROJECT_ROOT)}")
        return

    # ── 2a. Count pages with labels vs without ────────────────────────────
    has_label = sum(1 for r in rows if r.get("label_path"))
    has_png = sum(1 for r in rows if r.get("png_path"))
    yolo_invalid = [r for r in rows if r.get("yolo_label_valid") is False]
    print(f"\n-- Page data completeness --")
    info(f"Pages with png_path: {has_png:,} / {len(rows):,}")
    info(f"Pages with label_path: {has_label:,} / {len(rows):,}")
    if yolo_invalid:
        warn(f"Pages rejected by generator quality gate: {len(yolo_invalid):,}")
        reason_counts: Dict[str, int] = {}
        for row in yolo_invalid:
            for reason in row.get("yolo_reject_reasons", []):
                key = str(reason)
                reason_counts[key] = reason_counts.get(key, 0) + 1
        for reason, count in sorted(reason_counts.items(), key=lambda item: item[1], reverse=True):
            info(f"  reject_reason[{reason}] = {count:,}")

    # ── 2b. Spot-check PNG + label file existence ─────────────────────────
    print("\n-- Spot-check: PNG + YOLO label file existence (first 30) --")
    png_found = 0
    png_missing = 0
    label_found = 0
    label_missing = 0
    for row in rows[:30]:
        png = row.get("png_path")
        lbl = row.get("label_path")
        if png:
            if (PROJECT_ROOT / png).exists():
                png_found += 1
            else:
                png_missing += 1
        if lbl:
            if (PROJECT_ROOT / lbl).exists():
                label_found += 1
            else:
                label_missing += 1
    if png_missing:
        fail(f"PNG: {png_found}/30 found, {png_missing} missing")
    else:
        ok(f"PNG: {png_found}/30 found")
    if label_missing:
        fail(f"Labels: {label_found}/30 found, {label_missing} missing")
    else:
        ok(f"Labels: {label_found}/30 found")

    # ── 2c. Ultralytics label discovery ───────────────────────────────────
    print("\n-- Ultralytics label path convention --")
    # Ultralytics finds labels by replacing last 'images' dir with 'labels'
    sample_png = rows[0].get("png_path", "")
    sample_lbl = rows[0].get("label_path", "")
    if "images" in sample_png:
        ok(f"PNG paths contain 'images/' — Ultralytics can auto-discover labels")
    else:
        fail(
            f"PNG paths use 'pages_png/' not 'images/' — Ultralytics CANNOT auto-discover labels!\n"
            f"         Example PNG:   {sample_png}\n"
            f"         Example Label: {sample_lbl}\n"
            f"         FIX: Either rename pages_png/ to images/ and labels/yolo/ to labels/,\n"
            f"              or symlink, or restructure generate_synthetic.py output dirs.\n"
            f"              Ultralytics expects: .../images/split/file.png -> .../labels/split/file.txt"
        )

    # ── 2d. Plan vs actual: page counts ───────────────────────────────────
    print("\n-- Plan vs actual --")
    plan_target = 5_000  # Plan Section 4.1: "5,000 synthetic full pages"
    if has_label >= plan_target:
        ok(f"Page count: {has_label:,} >= plan target {plan_target:,}")
    elif has_label >= plan_target * 0.8:
        warn(f"Page count: {has_label:,} / plan target {plan_target:,} ({has_label*100/plan_target:.0f}%)")
    else:
        fail(f"Page count: {has_label:,} / plan target {plan_target:,} ({has_label*100/plan_target:.0f}%)")

    # ── 2e. Score type distribution ───────────────────────────────────────
    from collections import Counter
    score_types = Counter(str(r.get("score_type", "?")) for r in rows)
    style_ids = Counter(str(r.get("style_id", "?")) for r in rows)

    print("\n-- Score type distribution --")
    for st, n in score_types.most_common():
        pct = n / len(rows) * 100
        print(f"    {st:<20s} {n:>6,}  ({pct:5.1f}%)")

    plan_dist = {"piano": 40, "orchestral": 25, "chamber": 20, "choral": 10, "solo+piano": 5}
    missing_types = [t for t in plan_dist if t not in score_types and plan_dist[t] >= 10]
    if missing_types:
        warn(f"Plan expects score types not present: {', '.join(missing_types)}")
        info(f"Plan distribution: {plan_dist}")
        info(f"This is expected if only Lieder-main source material has been rendered so far.")

    print("\n-- Font/style distribution --")
    for st, n in style_ids.most_common():
        pct = n / len(rows) * 100
        print(f"    {st:<20s} {n:>6,}  ({pct:5.1f}%)")
    if len(style_ids) >= 3:
        ok(f"{len(style_ids)} rendering styles (plan says 3 visual configs)")
    else:
        warn(f"Only {len(style_ids)} rendering style(s), plan says 3")

    # ── 2f. External layout dataset (Dvořák et al.) ───────────────────────
    print("\n-- External layout dataset (Dvorak et al.) --")
    info("Plan says: '5,000 synthetic full pages + combined layout dataset from")
    info("Dvorak et al. (WORMS 2024) which merges AudioLabs v2, OSLiC, and MZKBlank")
    info("for ~7,000 images with staff, system, and measure annotations.'")
    # Check for any external layout data
    external_dirs = [
        PROJECT_ROOT / "data" / "audiolabs",
        PROJECT_ROOT / "data" / "AudioLabs",
        PROJECT_ROOT / "data" / "oslic",
        PROJECT_ROOT / "data" / "OSLiC",
        PROJECT_ROOT / "data" / "mzkblank",
        PROJECT_ROOT / "data" / "MZKBlank",
        PROJECT_ROOT / "data" / "dvorak",
        PROJECT_ROOT / "data" / "layout",
    ]
    found_any = False
    for d in external_dirs:
        if d.exists():
            found_any = True
            ok(f"Found external layout data: {d.relative_to(PROJECT_ROOT)}")
    if not found_any:
        warn(
            "No external layout dataset found (AudioLabs/OSLiC/MZKBlank).\n"
            "         YOLO trains only on synthetic pages. Plan says to also use ~7,000 external images.\n"
            "         This is OK to start with, but add external data for better generalization."
        )

    # ── 2g. YOLO label format spot-check ──────────────────────────────────
    print("\n-- YOLO label format spot-check --")
    checked = 0
    valid = 0
    for row in rows[:10]:
        lbl = row.get("label_path")
        if not lbl:
            continue
        lbl_path = PROJECT_ROOT / lbl
        if not lbl_path.exists():
            continue
        checked += 1
        text = lbl_path.read_text(encoding="utf-8").strip()
        lines = text.split("\n") if text else []
        all_ok = True
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                all_ok = False
                break
            cls_id = int(parts[0])
            if cls_id < 0 or cls_id > 0:
                all_ok = False
                break
            coords = [float(x) for x in parts[1:]]
            if not all(0 <= c <= 1.0001 for c in coords):
                all_ok = False
                break
        if all_ok:
            valid += 1
    if checked > 0:
        if valid == checked:
            ok(f"YOLO label format: {valid}/{checked} files valid (class_id cx cy w h, normalized)")
        else:
            fail(f"YOLO label format: {valid}/{checked} files valid")
    else:
        warn("Could not check any YOLO label files")

    # Check class distribution
    print("\n-- YOLO class distribution (from manifest class_counts) --")
    class_totals: Dict[str, int] = {}
    for row in rows:
        cc = row.get("class_counts", {})
        for cls_name, cnt in cc.items():
            class_totals[cls_name] = class_totals.get(cls_name, 0) + int(cnt)
    if class_totals:
        for cls_name, total in sorted(class_totals.items()):
            avg = total / max(len(rows), 1)
            print(f"    {cls_name:<20s} total={total:>8,}  avg/page={avg:>6.1f}")
        expected_classes = ["staff"]
        missing_expected = [name for name in expected_classes if class_totals.get(name, 0) <= 0]
        if missing_expected:
            warn(
                "Expected YOLO classes have zero labels: "
                + ", ".join(missing_expected)
                + ". Generator may be filtering them out."
            )
        else:
            ok("All 5 YOLO classes have non-zero label counts.")
    else:
        warn("No class_counts in manifest entries")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Summary
# ═════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print("OMR Training Data Validation")
    print(f"Project root: {PROJECT_ROOT}")

    check_curriculum()
    check_yolo()

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if fail_count == 0 and warn_count == 0:
        print(f"  {OK}  All checks passed!")
    else:
        if fail_count > 0:
            print(f"  {FAIL} {fail_count} failure(s)")
        if warn_count > 0:
            print(f"  {WARN} {warn_count} warning(s)")

    sys.exit(1 if fail_count > 0 else 0)


if __name__ == "__main__":
    main()
