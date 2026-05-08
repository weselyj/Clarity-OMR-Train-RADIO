#!/usr/bin/env python3
"""Audit a per-staff manifest for alignment correctness.

Checks (no labels-root):
- For each page, are staff_index values contiguous starting at 0?
- Distribution of entries-per-page.

Checks (with labels-root):
- Does each page's manifest entry count match the label file's line count?
  Mismatch indicates post-filter renumbering hiding dropped staves.

Dataset filter (--dataset):
- The production synthetic_v2 manifest carries both 'synthetic_fullpage' and
  'synthetic_polyphonic' variants per (page, staff_index) for piano score types.
  This causes a page with 3 physical staves to produce 6 entries with indices
  [0, 0, 1, 1, 2, 2], which trips the duplicate-index check spuriously.
- Pass --dataset <value> to restrict loading to entries whose 'dataset' field
  matches the given value before grouping by page_id.  When omitted, all entries
  are loaded (backward-compatible).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _load_manifest_indices(
    manifest_path: Path,
    dataset_filter: str | None = None,
) -> dict[str, list[int]]:
    """Load per-page staff_index lists from *manifest_path*.

    Args:
        manifest_path: Path to the JSONL manifest file.
        dataset_filter: When not ``None``, only entries whose ``dataset`` field
            equals this value are included.  Entries without a ``dataset`` field
            are excluded when a filter is active.  When ``None`` (the default),
            all entries are included regardless of ``dataset``.
    """
    by_page: dict[str, list[int]] = defaultdict(list)
    with manifest_path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if dataset_filter is not None and entry.get("dataset") != dataset_filter:
                    continue
                by_page[entry["page_id"]].append(int(entry["staff_index"]))
            except (json.JSONDecodeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"Bad manifest line {line_num} in {manifest_path}: {line!r}"
                ) from exc
    return by_page


def _count_label_lines(labels_root: Path, page_id: str) -> int | None:
    candidates = list(labels_root.rglob(f"{page_id}.txt"))
    if not candidates:
        return None
    label_file = candidates[0]
    return sum(1 for line in label_file.read_text().splitlines() if line.strip())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels-root", type=Path, default=None,
                        help="If given, compare per-page manifest entry count to label file line count.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=20,
                        help="How many drifted pages to include in the report sample.")
    parser.add_argument(
        "--dataset",
        default=None,
        help=(
            "When set, only manifest entries whose 'dataset' field equals this value "
            "are loaded. Useful for manifests that carry multiple dataset variants per "
            "(page, staff_index) — e.g. synthetic_v2 has both 'synthetic_fullpage' and "
            "'synthetic_polyphonic' for piano pages, which would otherwise trigger the "
            "duplicate-index check spuriously."
        ),
    )
    args = parser.parse_args()

    by_page = _load_manifest_indices(args.manifest, dataset_filter=args.dataset)
    pages_total = len(by_page)
    pages_with_gap = 0
    pages_non_contiguous = 0
    pages_with_duplicate_indices = 0
    sample_drift: list[dict] = []

    for page_id, indices in by_page.items():
        # Check for duplicate indices before deduplicating
        if len(indices) != len(set(indices)):
            pages_with_duplicate_indices += 1

        unique_sorted = sorted(set(indices))
        expected = list(range(len(unique_sorted)))
        if unique_sorted != expected:
            pages_non_contiguous += 1
            if max(unique_sorted) + 1 != len(unique_sorted):
                pages_with_gap += 1
            if len(sample_drift) < args.sample_size:
                sample_drift.append({"page_id": page_id, "staff_indices": unique_sorted})

    entries_per_page = Counter(len(v) for v in by_page.values())

    label_mismatch_pages = 0
    sample_label_mismatch: list[dict] = []
    if args.labels_root is not None:
        for page_id, indices in by_page.items():
            # manifest_entry_count is raw count; manifest_unique_count is deduplicated count.
            # Compare unique count to label line count for post-filter renumbering detection.
            manifest_entry_count = len(indices)
            manifest_unique_count = len(set(indices))
            label_count = _count_label_lines(args.labels_root, page_id)
            if label_count is not None and label_count != manifest_unique_count:
                label_mismatch_pages += 1
                if len(sample_label_mismatch) < args.sample_size:
                    sample_label_mismatch.append({
                        "page_id": page_id,
                        "manifest_entry_count": manifest_entry_count,
                        "manifest_unique_count": manifest_unique_count,
                        "label_count": label_count,
                    })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "manifest": str(args.manifest),
        "dataset_filter": args.dataset,
        "labels_root": str(args.labels_root) if args.labels_root else None,
        "pages_total": pages_total,
        "pages_with_duplicate_indices": pages_with_duplicate_indices,
        "pages_with_index_gap": pages_with_gap,
        "pages_with_non_contiguous_indices": pages_non_contiguous,
        "pages_with_label_count_mismatch": label_mismatch_pages,
        "entries_per_page": {str(k): v for k, v in sorted(entries_per_page.items())},
        "sample_drifted_pages": sample_drift,
        "sample_label_mismatch_pages": sample_label_mismatch,
    }, indent=2))
    print(f"[audit] manifest={args.manifest} pages={pages_total} "
          f"non_contiguous={pages_non_contiguous} label_mismatch={label_mismatch_pages}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
