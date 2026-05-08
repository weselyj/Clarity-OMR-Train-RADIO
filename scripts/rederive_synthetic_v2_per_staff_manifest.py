"""Re-derive the per-staff token manifest for an existing synthetic_v2 corpus.

Used after fixes to `_build_manifest_rows_for_page` (e.g., using physical
staff_index rather than post-filter saved_index). Re-reads existing pages,
SVGs, and YOLO label files; re-runs ONLY the manifest-building step.

Avoids the 6h cost of full Verovio re-rendering. Existing files under
pages/, labels/, labels_systems/, and staff_crops/ are NOT modified —
those are byte-identical to the original generation run.

Per-page workflow (SVG primary path):
  1. Read per-staff YOLO labels → pixel-space staff_boxes.
  2. Run _write_staff_crops (dry-run into a temp dir) to determine which
     physical positions survive the crop filter.
  3. Map survivors back to the existing crop PNG paths.
  4. Parse SVG layout.
  5. Assign staff boxes to systems.
  6. Look up / cache token sequences via convert_source_tokens.
  7. Build token_sequences_by_phys via _extract_measure_range_from_sequence.
  8. Call _build_manifest_rows_for_page and collect the output rows.

After all pages, write the collected rows to --output-manifest as JSONL.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.data.generate_synthetic import (  # noqa: E402
    _assign_staff_boxes_to_systems,
    _build_manifest_rows_for_page,
    _extract_measure_range_from_sequence,
    _extract_system_layout_from_svg,
    _write_staff_crops,
    convert_source_tokens,
    relpath,
)


# ---------------------------------------------------------------------------
# Public helpers (importable by tests)
# ---------------------------------------------------------------------------


def build_per_source_groups(
    page_entries: List[dict],
) -> Dict[Tuple[str, str], List[dict]]:
    """Group page entries by (source_path, style_id), sorted by page_number.

    This replicates the original generate_synthetic.py processing order:
    each source × style group is a contiguous run of pages in page_number
    order, which is required for cumulative_measure_offset to be correct.

    Returns a dict mapping (source_path, style_id) → sorted list of page dicts.
    """
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for entry in page_entries:
        key = (str(entry["source_path"]), str(entry["style_id"]))
        groups[key].append(entry)

    # Sort pages within each group by page_number (ascending).
    for key in groups:
        groups[key].sort(key=lambda e: int(e["page_number"]))

    return dict(groups)


def process_page(
    *,
    page_entry: dict,
    state: dict,
    token_cache: Dict[str, Optional[List[List[str]]]],
    corpus_root: Path,
    data_root: Path,
    project_root: Path,
) -> List[dict]:
    """Process a single page and return its manifest rows.

    ``state`` is a mutable dict with keys:
      - ``"offset"`` (int): cumulative_measure_offset for this source × style group.
        Updated in-place after processing.
      - ``"tokens"`` (List[List[str]]): cached token sequences for this source.

    Returns a list of row dicts (may be empty on error or skip).
    """
    page_id: str = page_entry["page_id"]
    source_path_str: str = page_entry["source_path"]
    style_id: str = page_entry["style_id"]
    score_type: str = page_entry["score_type"]
    page_number: int = int(page_entry["page_number"])
    page_width: float = float(page_entry["page_width"])
    page_height: float = float(page_entry["page_height"])

    staff_token_sequences: Optional[List[List[str]]] = state.get("tokens")
    part_count = len(staff_token_sequences) if staff_token_sequences else 0

    if not staff_token_sequences or part_count == 0:
        # No token sequences available for this source; skip silently.
        return []

    # 1. Read SVG text.
    # Defensive guard: svg_path may be None on invalid pages.
    raw_svg_path = page_entry.get("svg_path")
    if raw_svg_path is None:
        print(f"  [SKIP] svg_path is null for {page_id}, skipping.", flush=True)
        return []
    svg_path = corpus_root / raw_svg_path
    if not svg_path.exists():
        print(f"  [WARN] SVG not found, skipping: {svg_path}", flush=True)
        return []
    svg_text = svg_path.read_text(encoding="utf-8")

    # 2. Read per-staff YOLO labels → pixel-space staff_boxes.
    # Guard: label_path is null when yolo_label_valid is False.
    raw_label_path = page_entry.get("label_path")
    if raw_label_path is None:
        print(f"  [SKIP] label_path is null for {page_id} (yolo_label_valid=false), skipping.", flush=True)
        return []
    label_path = corpus_root / raw_label_path
    if not label_path.exists():
        print(f"  [WARN] Label file not found, skipping: {label_path}", flush=True)
        return []

    staff_boxes: List[Tuple[float, float, float, float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        w_px = w * page_width
        h_px = h * page_height
        x_px = cx * page_width - w_px / 2.0
        y_px = cy * page_height - h_px / 2.0
        staff_boxes.append((x_px, y_px, w_px, h_px))

    if not staff_boxes:
        return []

    # 3. Run _write_staff_crops in a temp dir to determine survivor physical positions.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        try:
            raw_crop_entries: List[Tuple[Path, int]] = _write_staff_crops(
                svg_text=svg_text,
                staff_boxes=staff_boxes,
                source_page_width=page_width,
                source_page_height=page_height,
                output_dir=tmp_path,
                page_basename=page_id,
                max_crops=None,
            )
        except Exception as exc:
            print(f"  [WARN] _write_staff_crops failed for {page_id}: {exc}", flush=True)
            return []

    # 4. Map survivors back to existing crop file paths.
    #    raw_crop_entries is a list of (temp_path, phys_idx) in saved_index order (1-indexed).
    #    The existing crop at saved_index i+1 (0-indexed i) is:
    #    staff_crops/<style>/<page_id>__staff{i+1:02d}.png
    crops_dir = corpus_root / "staff_crops" / style_id
    staff_crop_entries: List[Tuple[Path, int]] = []
    for saved_enum, (_tmp_path, phys_idx) in enumerate(raw_crop_entries):
        existing_crop = crops_dir / f"{page_id}__staff{saved_enum + 1:02d}.png"
        staff_crop_entries.append((existing_crop, phys_idx))

    # 5. Parse SVG layout.
    svg_layout = _extract_system_layout_from_svg(svg_text)
    if not svg_layout:
        # No layout found; skip this page (cannot build token sequences without it).
        return []

    svg_measures = [s.measure_count for s in svg_layout]

    # 6. Assign staff boxes to systems.
    staff_to_system = _assign_staff_boxes_to_systems(staff_boxes, svg_layout)

    # 7. Build token_sequences_by_phys for all physical positions.
    cumulative_measure_offset: int = state["offset"]
    token_sequences_by_phys: Dict[int, List[str]] = {}

    for phys_idx in range(len(staff_boxes)):
        mapping = staff_to_system.get(phys_idx)
        if mapping is None:
            continue
        sys_idx, pos_in_system = mapping
        if sys_idx >= len(svg_measures):
            continue
        pi = pos_in_system % part_count
        if pi >= part_count:
            continue
        m_start = cumulative_measure_offset + sum(svg_measures[:sys_idx])
        m_end = m_start + svg_measures[sys_idx]
        token_sequences_by_phys[phys_idx] = _extract_measure_range_from_sequence(
            staff_token_sequences[pi], m_start, m_end
        )

    # 8. Compute dataset_variants.
    dataset_variants: List[Tuple[str, str]] = [("synthetic_fullpage", "")]
    if score_type in {"piano", "chamber", "solo_instrument_with_piano"}:
        dataset_variants.append(("synthetic_polyphonic", "__poly"))

    # 9. Build manifest rows.
    new_rows = _build_manifest_rows_for_page(
        page_basename=page_id,
        staff_crop_entries=staff_crop_entries,
        total_physical_staves=len(staff_boxes),
        token_sequences_by_phys=token_sequences_by_phys,
        page_number=page_number,
        style_id=style_id,
        score_type=score_type,
        source_relpath=source_path_str,
        project_root=project_root,
        dataset_variants=dataset_variants,
    )

    # 10. Advance offset only when crops survived — mirrors the original
    #     generate_synthetic.py behavior where the offset is updated inside
    #     the `if staff_crop_entries:` branch. Pages where all crops fail the
    #     ink filter must not shift the measure-range window for subsequent pages.
    if raw_crop_entries:
        state["offset"] += sum(svg_measures)

    return new_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=Path("data/processed/synthetic_v2"),
        help="Root of the synthetic_v2 corpus (contains pages/, labels/, staff_crops/).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Repo data root; source_path entries in the pages manifest are relative to this.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        required=True,
        help="Path to write the new per-staff manifest JSONL.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="Optional limit for smoke testing (process only the first N pages).",
    )
    args = parser.parse_args()

    corpus_root = args.corpus_root.resolve()
    data_root = args.data_root.resolve()
    project_root = REPO
    output_path = args.output_manifest.resolve()

    pages_manifest = corpus_root / "manifests" / "synthetic_pages.jsonl"
    if not pages_manifest.exists():
        print(f"ERROR: pages manifest not found at {pages_manifest}", flush=True)
        return 1

    # Load all page entries.
    page_entries: List[dict] = []
    with pages_manifest.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            page_entries.append(json.loads(line))

    if args.max_pages is not None:
        page_entries = page_entries[: args.max_pages]

    total_pages = len(page_entries)
    print(
        f"Loaded {total_pages} page entries from {pages_manifest}",
        flush=True,
    )

    # Group pages by (source_path, style_id) in page_number order.
    groups = build_per_source_groups(page_entries)
    print(f"Processing {len(groups)} source×style groups.", flush=True)

    # Per-source token cache (keyed by source_path string).
    token_cache: Dict[str, Optional[List[List[str]]]] = {}

    all_rows: List[dict] = []
    n_processed = 0
    n_skipped_source = 0
    n_skipped_page = 0
    t0 = time.time()

    for (source_path_str, style_id), group_pages in groups.items():
        # Populate token cache for this source.
        if source_path_str not in token_cache:
            abs_source = (data_root / source_path_str).resolve()
            try:
                token_cache[source_path_str] = convert_source_tokens(abs_source)
            except Exception as exc:
                print(
                    f"  [WARN] convert_source_tokens failed for {source_path_str}: {exc}",
                    flush=True,
                )
                token_cache[source_path_str] = None

        staff_token_sequences = token_cache[source_path_str]
        if not staff_token_sequences:
            n_skipped_source += len(group_pages)
            continue

        # Initialise per-source×style state with offset 0.
        state: dict = {"offset": 0, "tokens": staff_token_sequences}

        for page_entry in group_pages:
            page_id = page_entry["page_id"]
            rows = process_page(
                page_entry=page_entry,
                state=state,
                token_cache=token_cache,
                corpus_root=corpus_root,
                data_root=data_root,
                project_root=project_root,
            )
            if rows:
                all_rows.extend(rows)
                n_processed += 1
            else:
                n_skipped_page += 1

            total_done = n_processed + n_skipped_page
            if total_done % 500 == 0 and total_done > 0:
                elapsed = time.time() - t0
                print(
                    f"  [{total_done}/{total_pages}] pages processed, "
                    f"last_page={page_id}, "
                    f"rows_so_far={len(all_rows)}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )

    # Write output manifest.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out_fh:
        for row in all_rows:
            out_fh.write(json.dumps(row) + "\n")

    elapsed = time.time() - t0
    print(
        f"\nDone in {elapsed:.1f}s. "
        f"{n_processed} pages produced {len(all_rows)} manifest rows. "
        f"Skipped: source_error={n_skipped_source}, page_skip={n_skipped_page}.",
        flush=True,
    )
    print(f"Output written to: {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
