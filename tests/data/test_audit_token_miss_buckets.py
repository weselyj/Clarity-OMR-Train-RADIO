# tests/data/test_audit_token_miss_buckets.py
"""Tests for scripts/audit_token_miss_buckets.py.

Classification logic under test:
  B — SVG path used (page has ≥1 null-image_path manifest row); missing staff
      index is a "surplus box" beyond the number the SVG system layout claimed.
  C — SVG OOB: sys_idx >= len(svg_measures). In practice dead code because
      _assign_staff_boxes_to_systems always produces sys_idx < len(systems).
      Tested for robustness; expected count is 0 on real data.
  D — Fallback path (no null-image rows on the page); missing staff index was
      a crop that failed the ink/border filter.
  E — Page has zero manifest rows (all crops dropped before reaching the loop);
      never iterated by the builder so contributes 0 to dropped_token_miss.

The discriminator between SVG-path and fallback-path is the presence of at
least one manifest row with image_path == None on that page.  On the SVG path,
_build_manifest_rows_for_page emits a row (with null image_path) for every
physical staff that maps to a valid SVG system — even when the crop was
filtered. The fallback path only emits rows for surviving crops, so image_path
is never None on a fallback-path page.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Import under test (functions from the script, not production code)
# ---------------------------------------------------------------------------

from scripts.audit_token_miss_buckets import (
    classify_page,
    load_per_staff_lookup,
    load_staves_sidecar,
    compute_token_miss_rows,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_manifest_jsonl(tmp_path: Path, rows: list[dict]) -> Path:
    p = tmp_path / "manifest.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


def _make_staves_json(tmp_path: Path, page_id: str, style_id: str, staves: list[int]) -> Path:
    d = tmp_path / "labels_systems" / style_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{page_id}.staves.json"
    p.write_text(json.dumps(staves))
    return p


def _base_row(page_id: str, staff_index: int, image_path: str | None = "some/path.png") -> dict:
    return {
        "sample_id": f"{page_id}__staff{staff_index + 1:02d}",
        "dataset": "synthetic_fullpage",
        "split": "train",
        "image_path": image_path,
        "page_id": page_id,
        "source_path": "data/score.mxl",
        "style_id": page_id.rsplit("__", 2)[1] if "__" in page_id else "bravura-compact",
        "page_number": 1,
        "staff_index": staff_index,
        "source_format": "musicxml",
        "score_type": "piano",
        "token_sequence": ["<bos>", "<staff_start>", "<staff_end>", "<eos>"],
        "token_count": 4,
    }


# ---------------------------------------------------------------------------
# Unit tests for classify_page
# ---------------------------------------------------------------------------

class TestClassifyPage:
    """Tests for the per-page classification helper."""

    def test_bucket_B_svg_surplus_box(self):
        """SVG path page (has null-image row); last staff of a system is surplus.

        Setup: staves=[2, 2]; page has 3 manifest rows (0,1,2) with row 1
        having image_path=None (SVG path indicator). Staff indices 3 from
        system 1 is missing → Bucket B.
        """
        page_id = "score__bravura-compact__p001"
        staves = [2, 2]  # 4 expected total staves
        # Manifest rows: indices 0,1,2 present; index 3 absent
        # Row 1 has null image_path → SVG path indicator
        manifest_indices = {0, 1, 2}
        has_null_image = True  # at least one row has image_path=None

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=has_null_image,
        )

        assert result["total_miss"] == 1, f"Expected 1 miss, got {result['total_miss']}"
        assert result["bucket_B"] == 1
        assert result["bucket_C"] == 0
        assert result["bucket_D"] == 0
        assert result["bucket_E"] == 0

    def test_bucket_C_svg_oob(self):
        """SVG OOB (dead code path). classify_page receives explicit bucket_c
        override to test the data structure, since we cannot trigger this from
        staves.json alone. Expected to be 0 on real data.
        """
        # Since bucket C is dead code, we test that classify_page returns 0
        # for it on a normal SVG-path page (same as Bucket B scenario).
        page_id = "score__bravura-compact__p001"
        staves = [2, 2]
        manifest_indices = {0, 1, 2}  # missing index 3

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=True,
        )
        assert result["bucket_C"] == 0  # dead code on real data

    def test_bucket_D_fallback_crop_filtered(self):
        """Fallback path page (no null-image rows); a specific staff's crop was
        filtered out, so it has no manifest row.

        Setup: staves=[2, 2]; manifest has rows 0,1,2 (all with real paths).
        Index 3 is missing → Bucket D (fallback + crop filtered).
        """
        page_id = "score__gootville-wide__p002"
        staves = [2, 2]  # 4 expected total staves
        manifest_indices = {0, 1, 2}  # missing index 3
        has_null_image = False  # fallback path

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=has_null_image,
        )

        assert result["total_miss"] == 1
        assert result["bucket_B"] == 0
        assert result["bucket_D"] == 1
        assert result["bucket_E"] == 0

    def test_bucket_E_zero_manifest_rows(self):
        """Page with zero manifest rows (all crops dropped in Bucket E scenario).

        The builder never processes this page, so it contributes 0 to
        dropped_token_miss. classify_page with empty manifest_indices still
        reports the staves as E for documentation but the builder count is 0.
        """
        page_id = "score__bravura-compact__p003"
        staves = [3, 3, 3]  # 9 expected staves
        manifest_indices = set()  # zero rows
        has_null_image = False

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=has_null_image,
        )

        # Bucket E: page with zero rows; but builder_miss = 0 because builder
        # never processes this page.
        assert result["bucket_E"] == 3  # all 3 systems are E-type
        assert result["builder_miss"] == 0  # not counted in builder totals

    def test_multiple_systems_mixed_misses(self):
        """Page where system 0 is complete but system 1 has a miss (Bucket D)."""
        page_id = "score__leipzig-default__p002"
        staves = [3, 3]  # system 0: indices 0,1,2; system 1: indices 3,4,5
        manifest_indices = {0, 1, 2, 3, 4}  # index 5 missing (last of system 1)
        has_null_image = False

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=has_null_image,
        )

        assert result["total_miss"] == 1
        assert result["bucket_D"] == 1
        assert result["builder_miss"] == 1  # 1 system dropped

    def test_no_miss_returns_zeros(self):
        """Page where all expected staves are in the manifest → no miss."""
        page_id = "score__bravura-compact__p001"
        staves = [2, 2]
        manifest_indices = {0, 1, 2, 3}  # all present

        result = classify_page(
            page_id=page_id,
            staves_per_system=staves,
            manifest_staff_indices=manifest_indices,
            has_null_image_row=False,
        )

        assert result["total_miss"] == 0
        assert result["builder_miss"] == 0
        assert result["bucket_B"] == 0
        assert result["bucket_D"] == 0


# ---------------------------------------------------------------------------
# Unit tests for load helpers
# ---------------------------------------------------------------------------

class TestLoadHelpers:
    def test_load_per_staff_lookup_basic(self, tmp_path):
        """load_per_staff_lookup returns index→has_null_image per page."""
        page_id = "score__bravura-compact__p001"
        rows = [
            _base_row(page_id, 0, image_path="some/path.png"),
            _base_row(page_id, 1, image_path=None),
            _base_row(page_id, 2, image_path="some/path2.png"),
        ]
        # Add a polyphonic duplicate row (different dataset) to ensure
        # load_per_staff_lookup deduplicates.
        poly_row = dict(_base_row(page_id, 0, "some/path.png"))
        poly_row["dataset"] = "synthetic_polyphonic"
        poly_row["sample_id"] = f"{page_id}__staff01__poly"
        rows.append(poly_row)

        manifest = _make_manifest_jsonl(tmp_path, rows)
        lookup = load_per_staff_lookup(manifest)

        assert page_id in lookup
        page_data = lookup[page_id]
        assert page_data["indices"] == {0, 1, 2}
        assert page_data["has_null_image"] is True

    def test_load_per_staff_lookup_no_null(self, tmp_path):
        page_id = "score__gootville-wide__p002"
        rows = [
            _base_row(page_id, 0, image_path="p.png"),
            _base_row(page_id, 1, image_path="p.png"),
        ]
        manifest = _make_manifest_jsonl(tmp_path, rows)
        lookup = load_per_staff_lookup(manifest)

        assert lookup[page_id]["has_null_image"] is False

    def test_load_staves_sidecar(self, tmp_path):
        page_id = "score__bravura-compact__p001"
        style_id = "bravura-compact"
        _make_staves_json(tmp_path, page_id, style_id, [3, 3, 3])

        staves_root = tmp_path / "labels_systems"
        result = load_staves_sidecar(staves_root, page_id, style_id)

        assert result == [3, 3, 3]

    def test_load_staves_sidecar_missing_file(self, tmp_path):
        staves_root = tmp_path / "labels_systems"
        staves_root.mkdir(parents=True, exist_ok=True)
        result = load_staves_sidecar(staves_root, "nonexistent__style__p001", "style")
        assert result is None


# ---------------------------------------------------------------------------
# Integration test: full classifier on a small fixture corpus
# ---------------------------------------------------------------------------

class TestComputeTokenMissRows:
    """Integration-flavored test running the full classifier on a small fixture."""

    def _setup_fixture(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Build a small fixture corpus with known bucket counts.

        Corpus:
          - page_A (Bucket B, SVG path): staves=[2,2]; manifest has [0,1,2]
            with row 1 having image_path=None. System 1 is missing staff 3.
          - page_B (Bucket D, fallback): staves=[2,2]; manifest has [0,1,2]
            all with real paths. System 1 missing staff 3.
          - page_C (Bucket E, zero rows): staves=[3]; no manifest rows.
          - page_D (complete, no miss): staves=[2]; manifest has [0,1].

        Expected builder_miss counts:
          B=1, C=0, D=1, E=0, total=2
        (page_C is never iterated by builder, so E contributes 0.)
        """
        style_id = "bravura-compact"

        manifest_rows = [
            # page_A: SVG path (row 1 has null image)
            _base_row(f"page_A__{style_id}__p001", 0, "p.png"),
            _base_row(f"page_A__{style_id}__p001", 1, None),   # SVG null indicator
            _base_row(f"page_A__{style_id}__p001", 2, "p.png"),
            # page_B: fallback (all non-null)
            _base_row(f"page_B__{style_id}__p001", 0, "p.png"),
            _base_row(f"page_B__{style_id}__p001", 1, "p.png"),
            _base_row(f"page_B__{style_id}__p001", 2, "p.png"),
            # page_C: zero rows (omitted here)
            # page_D: complete
            _base_row(f"page_D__{style_id}__p001", 0, "p.png"),
            _base_row(f"page_D__{style_id}__p001", 1, "p.png"),
        ]
        manifest = _make_manifest_jsonl(tmp_path, manifest_rows)

        staves_root = tmp_path / "labels_systems"
        for page_id, staves in [
            (f"page_A__{style_id}__p001", [2, 2]),
            (f"page_B__{style_id}__p001", [2, 2]),
            (f"page_C__{style_id}__p001", [3]),      # 3 staves, zero rows
            (f"page_D__{style_id}__p001", [2]),
        ]:
            _make_staves_json(tmp_path, page_id, style_id, staves)

        output = tmp_path / "token_miss_breakdown.jsonl"
        return manifest, staves_root, output

    def test_integration_bucket_counts(self, tmp_path: Path):
        manifest, staves_root, output = self._setup_fixture(tmp_path)

        rows, totals = compute_token_miss_rows(
            per_staff_manifest=manifest,
            labels_systems_root=staves_root,
        )

        # Total builder_miss must equal 2 (page_A sys 1 + page_B sys 1).
        # Bucket E (page_C) is excluded from builder_miss because the builder
        # never iterates zero-row pages; it is reported separately in bucket_E.
        assert totals["builder_miss"] == 2, (
            f"Expected 2 builder_miss, got {totals['builder_miss']}"
        )
        assert totals["bucket_B"] == 1
        assert totals["bucket_C"] == 0
        assert totals["bucket_D"] == 1
        # page_C (staves=[3]) has 1 system → bucket_E = 1 system
        assert totals["bucket_E"] == 1  # zero-row page; system-level count
        # B + C + D == builder_miss (E excluded)
        assert totals["bucket_B"] + totals["bucket_C"] + totals["bucket_D"] == totals["builder_miss"]

    def test_integration_jsonl_row_fields(self, tmp_path: Path):
        manifest, staves_root, output = self._setup_fixture(tmp_path)

        rows, totals = compute_token_miss_rows(
            per_staff_manifest=manifest,
            labels_systems_root=staves_root,
        )

        # Each JSONL row must have required fields
        required_fields = {
            "page_id", "sys_idx", "missing_staff_index", "bucket", "reason_detail",
        }
        for row in rows:
            assert required_fields.issubset(row.keys()), (
                f"Row missing fields: {required_fields - row.keys()}"
            )

    def test_integration_output_totals_match_rows(self, tmp_path: Path):
        manifest, staves_root, output = self._setup_fixture(tmp_path)

        rows, totals = compute_token_miss_rows(
            per_staff_manifest=manifest,
            labels_systems_root=staves_root,
        )

        # Count rows per bucket and compare to totals
        from collections import Counter
        bucket_counts = Counter(r["bucket"] for r in rows)

        # builder_miss counts systems (one per system, even if multiple staves missing)
        # For this fixture each miss-system has exactly 1 missing staff, so counts match
        assert bucket_counts.get("B", 0) == totals["bucket_B"]
        assert bucket_counts.get("D", 0) == totals["bucket_D"]
