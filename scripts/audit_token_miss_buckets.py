#!/usr/bin/env python3
"""Post-hoc characterisation of token_miss drops in synthetic_systems_v1.

Classifies each dropped system into one of four root-cause buckets:

  B — SVG path was used (page has ≥1 null image_path manifest row); the
      missing staff index is a "surplus box" beyond what the SVG system layout
      claimed.  _assign_staff_boxes_to_systems assigns boxes sequentially;
      any physical staff index ≥ sum(svg_staves_per_system) is left
      unmapped → no token sequence → token_miss.

  C — SVG OOB: sys_idx ≥ len(svg_measures).  Dead code in practice because
      _assign_staff_boxes_to_systems always yields sys_idx < len(systems);
      reported for completeness (expected count 0).

  D — Fallback path (no null image_path rows on this page); a specific
      staff crop was rejected by the ink/border filter so no token sequence
      was produced for that physical position.

  E — Page has zero manifest rows (all crops dropped at the
      ``if not staff_crop_entries`` early-exit); the builder never iterates
      this page, so it contributes 0 to dropped_token_miss.

Discriminator between SVG and fallback paths
--------------------------------------------
When the SVG path is used, _build_manifest_rows_for_page emits a manifest row
(with image_path = None) for every physical staff that maps to a valid SVG
system, even when that staff's crop was filtered.  The fallback path only
emits rows for surviving crops, so image_path is never None on a
fallback-path page.

Therefore: if any manifest row for a page has ``image_path == null``, the SVG
path was used for that page (→ Bucket B or C); otherwise the fallback was
used (→ Bucket D).

Counts at the system level
--------------------------
yolo_aligned_systems.process_page_systems increments dropped_token_miss once
per matched system (not once per missing staff).  This script counts at the
same granularity: one miss per system where ANY expected staff is absent from
the manifest.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Public API (imported by tests)
# ---------------------------------------------------------------------------

def load_per_staff_lookup(per_staff_manifest: Path) -> dict[str, dict]:
    """Read the per-staff JSONL manifest → dict keyed by page_id.

    Returns
    -------
    dict mapping page_id → {
        "indices": set of int staff indices (synthetic_fullpage dataset only),
        "has_null_image": bool (True if any row for this page has image_path None),
    }
    """
    by_page: dict[str, dict] = {}
    with per_staff_manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            # Only use synthetic_fullpage to avoid double-counting polyphonic rows
            if entry.get("dataset") != "synthetic_fullpage":
                continue
            pid = entry["page_id"]
            if pid not in by_page:
                by_page[pid] = {"indices": set(), "has_null_image": False}
            by_page[pid]["indices"].add(int(entry["staff_index"]))
            if entry.get("image_path") is None:
                by_page[pid]["has_null_image"] = True
    return by_page


def load_staves_sidecar(labels_systems_root: Path, page_id: str, style_id: str) -> list[int] | None:
    """Load <labels_systems_root>/<style_id>/<page_id>.staves.json.

    Returns the list of stave-counts per system, or None if the file is absent.
    """
    p = labels_systems_root / style_id / f"{page_id}.staves.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def classify_page(
    *,
    page_id: str,
    staves_per_system: list[int],
    manifest_staff_indices: set[int],
    has_null_image_row: bool,
) -> dict:
    """Classify all token_miss events for a single page.

    Parameters
    ----------
    page_id:
        Identifier for this page (used in returned diagnostics).
    staves_per_system:
        List of stave counts per oracle system (from staves.json).
    manifest_staff_indices:
        Set of physical staff indices present in the per-staff manifest for
        this page (synthetic_fullpage dataset only).
    has_null_image_row:
        True if any manifest row for this page has image_path == None.
        This is the SVG-path discriminator.

    Returns
    -------
    dict with keys:
        total_miss   — total missing (page_id, staff_index) pairs
        builder_miss — systems dropped by the builder (one per system with
                       any missing staff); 0 if page has zero manifest rows
                       (Bucket E), because the builder never iterates such pages
        bucket_B     — SVG surplus boxes
        bucket_C     — SVG OOB (dead code; always 0 from real data)
        bucket_D     — fallback + crop filtered
        bucket_E     — zero-row page (all systems; builder_miss = 0)
        rows         — list of per-miss detail dicts for JSONL output
    """
    result = {
        "total_miss": 0,
        "builder_miss": 0,
        "bucket_B": 0,
        "bucket_C": 0,
        "bucket_D": 0,
        "bucket_E": 0,
        "rows": [],
    }

    is_zero_row_page = len(manifest_staff_indices) == 0

    cumsum = 0
    for sys_idx, n in enumerate(staves_per_system):
        staff_for_sys = list(range(cumsum, cumsum + n))
        missing = [s for s in staff_for_sys if s not in manifest_staff_indices]

        if missing:
            result["total_miss"] += len(missing)

            if is_zero_row_page:
                # Bucket E: page never processed by builder
                result["bucket_E"] += 1
                bucket = "E"
                reason = "zero manifest rows for page — all crops dropped before token loop"
            elif has_null_image_row:
                # SVG path: surplus boxes (B); C is dead code, counted as B
                result["bucket_B"] += 1
                bucket = "B"
                reason = (
                    "SVG path — physical staff index(es) beyond what SVG system layout "
                    "claimed (_assign_staff_boxes_to_systems left them unmapped)"
                )
            else:
                # Fallback path: crop filtered (D)
                result["bucket_D"] += 1
                bucket = "D"
                reason = (
                    "fallback path — physical staff crop rejected by ink/border filter; "
                    "no token sequence produced for this position"
                )

            if not is_zero_row_page:
                result["builder_miss"] += 1

            for s in missing:
                result["rows"].append({
                    "page_id": page_id,
                    "sys_idx": sys_idx,
                    "missing_staff_index": s,
                    "bucket": bucket,
                    "reason_detail": reason,
                    "staves_per_system": staves_per_system,
                    "manifest_count": len(manifest_staff_indices),
                    "has_null_image_row": has_null_image_row,
                })

        cumsum += n

    return result


def _style_id_from_page_id(page_id: str) -> str | None:
    """Extract style_id from a page_id formatted as <...>__<style_id>__<page>.

    Returns None if the format doesn't match.
    """
    parts = page_id.rsplit("__", 2)
    if len(parts) >= 3:
        return parts[-2]
    return None


def compute_token_miss_rows(
    *,
    per_staff_manifest: Path,
    labels_systems_root: Path,
) -> tuple[list[dict], dict]:
    """Run the full post-hoc classifier over the entire manifest.

    Returns
    -------
    rows : list[dict]
        One JSONL-ready dict per (page_id, sys_idx, missing_staff_index).
    totals : dict
        Aggregate counts: total_miss, builder_miss, bucket_B, bucket_C,
        bucket_D, bucket_E.
    """
    per_page = load_per_staff_lookup(per_staff_manifest)

    # Also enumerate pages that have a staves.json but zero manifest rows
    zero_row_pages: dict[str, dict] = {}
    for style_dir in labels_systems_root.iterdir():
        if not style_dir.is_dir():
            continue
        style_id = style_dir.name
        for staves_path in style_dir.glob("*.staves.json"):
            page_id = staves_path.name.replace(".staves.json", "")
            if page_id not in per_page:
                zero_row_pages[page_id] = {
                    "indices": set(),
                    "has_null_image": False,
                    "style_id": style_id,
                }

    all_rows: list[dict] = []
    totals = {
        "total_miss": 0,
        "builder_miss": 0,
        "bucket_B": 0,
        "bucket_C": 0,
        "bucket_D": 0,
        "bucket_E": 0,
    }

    # Process pages with at least one manifest row
    for page_id, page_data in sorted(per_page.items()):
        style_id = _style_id_from_page_id(page_id)
        if style_id is None:
            continue
        staves = load_staves_sidecar(labels_systems_root, page_id, style_id)
        if staves is None:
            continue

        pr = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=page_data["indices"],
            has_null_image_row=page_data["has_null_image"],
        )
        for k in ("total_miss", "builder_miss", "bucket_B", "bucket_C", "bucket_D", "bucket_E"):
            totals[k] += pr[k]
        all_rows.extend(pr["rows"])

    # Process zero-row pages (Bucket E; contributes 0 to builder_miss)
    for page_id, page_data in sorted(zero_row_pages.items()):
        style_id = page_data.get("style_id") or _style_id_from_page_id(page_id)
        if style_id is None:
            continue
        staves = load_staves_sidecar(labels_systems_root, page_id, style_id)
        if staves is None:
            continue

        pr = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=set(),
            has_null_image_row=False,
        )
        for k in ("total_miss", "builder_miss", "bucket_B", "bucket_C", "bucket_D", "bucket_E"):
            totals[k] += pr[k]
        all_rows.extend(pr["rows"])

    return all_rows, totals


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-staff-manifest", type=Path, required=True,
        help="data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl",
    )
    parser.add_argument(
        "--labels-systems-root", type=Path, required=True,
        help="data/processed/synthetic_v2/labels_systems/",
    )
    parser.add_argument(
        "--output-jsonl", type=Path,
        default=Path("data/processed/synthetic_systems_v1/token_miss_breakdown.jsonl"),
        help="Per-miss JSONL output path.",
    )
    parser.add_argument(
        "--expected-total", type=int, default=1214,
        help="Expected builder_miss count from audit.json; script warns if mismatch.",
    )
    args = parser.parse_args()

    print(f"[audit_buckets] loading per-staff manifest: {args.per_staff_manifest}", flush=True)
    rows, totals = compute_token_miss_rows(
        per_staff_manifest=args.per_staff_manifest,
        labels_systems_root=args.labels_systems_root,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row) + "\n")

    print(f"\n[audit_buckets] === Token-miss bucket breakdown ===", flush=True)
    print(f"  builder_miss (should equal audit.json dropped_token_miss): {totals['builder_miss']}", flush=True)
    print(f"  Bucket B (SVG surplus box):         {totals['bucket_B']}", flush=True)
    print(f"  Bucket C (SVG OOB — dead code):     {totals['bucket_C']}", flush=True)
    print(f"  Bucket D (fallback + crop filtered): {totals['bucket_D']}", flush=True)
    print(f"  Bucket E (zero-row page, not in builder): {totals['bucket_E']}", flush=True)
    print(f"  total_miss (staff-level): {totals['total_miss']}", flush=True)
    print(f"\n[audit_buckets] JSONL written to: {args.output_jsonl}", flush=True)
    print(f"[audit_buckets] Total rows in JSONL: {len(rows)}", flush=True)

    if totals["builder_miss"] != args.expected_total:
        print(
            f"\n[audit_buckets] WARNING: builder_miss={totals['builder_miss']} "
            f"does not match expected={args.expected_total}. "
            f"Gap={totals['builder_miss'] - args.expected_total}.",
            flush=True,
        )
        return 1

    print(f"\n[audit_buckets] PASS: builder_miss matches expected {args.expected_total}.", flush=True)

    # Sample rows per bucket for quick inspection
    for bucket in ("B", "C", "D", "E"):
        bucket_rows = [r for r in rows if r["bucket"] == bucket]
        if not bucket_rows:
            continue
        print(f"\n[audit_buckets] Sample bucket {bucket} rows (up to 2):", flush=True)
        for r in bucket_rows[:2]:
            print(f"  {json.dumps({k: r[k] for k in ('page_id', 'sys_idx', 'missing_staff_index', 'bucket', 'manifest_count')})}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
