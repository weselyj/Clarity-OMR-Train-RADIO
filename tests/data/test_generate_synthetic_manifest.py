"""Producer-side test: per-staff manifest must use pre-filter physical staff_index.

This file directly exercises the contract that `generate_synthetic.py` writes
manifest entries keyed by physical staff position (not post-filter enumerate
of survivors). We invoke an internal helper that returns the manifest rows
for a synthetic crop_entries fixture, isolated from Verovio rendering.
"""
from __future__ import annotations

from pathlib import Path


def test_manifest_rows_use_physical_position_for_staff_index() -> None:
    """staff_crop_entries with physical positions [0,1,2,3,5,6] (filter dropped 4)
    must produce manifest rows with staff_index in [0,1,2,3,5,6], NOT [0..5].
    Filtered positions (4, 7, 8) get rows with image_path=None and the right tokens."""
    from src.data.generate_synthetic import _build_manifest_rows_for_page

    # 9 physical staves; survivors at positions 0,1,2,3,5,6
    staff_crop_entries = [
        (Path(f"crop_{p}.png"), p) for p in [0, 1, 2, 3, 5, 6]
    ]
    total_physical_staves = 9
    # One token sequence per physical position (artificial)
    token_sequences_by_phys = {
        p: ["<bos>", "<staff_start>", f"phys-{p}", "<staff_end>", "<eos>"]
        for p in range(total_physical_staves)
    }

    rows = _build_manifest_rows_for_page(
        page_basename="page_X",
        staff_crop_entries=staff_crop_entries,
        total_physical_staves=total_physical_staves,
        token_sequences_by_phys=token_sequences_by_phys,
        page_number=1,
        style_id="bravura-compact",
        score_type="vocal",
        source_relpath="data/example.mxl",
        project_root=Path("/tmp"),
        dataset_variants=[("synthetic_fullpage", "")],
    )

    # Every physical position with a token sequence gets a row.
    assert len(rows) == total_physical_staves
    by_idx = {r["staff_index"]: r for r in rows}
    assert sorted(by_idx) == list(range(total_physical_staves))

    # Survivors have image_path; filtered (4, 7, 8) have None.
    for p in [0, 1, 2, 3, 5, 6]:
        assert by_idx[p]["image_path"] is not None
    for p in [4, 7, 8]:
        assert by_idx[p]["image_path"] is None

    # Tokens always derived from the per-physical-position lookup, regardless of crop.
    for p in range(total_physical_staves):
        assert f"phys-{p}" in by_idx[p]["token_sequence"]


def test_manifest_rows_skip_positions_without_tokens() -> None:
    """If a physical position has no token sequence in the lookup
    (e.g., the renderer's measure mapping skipped it), no row is emitted."""
    from src.data.generate_synthetic import _build_manifest_rows_for_page

    staff_crop_entries = [(Path(f"crop_{p}.png"), p) for p in [0, 1, 2]]
    # Only positions 0 and 2 have tokens — position 1's token sequence is missing.
    token_sequences_by_phys = {
        0: ["<bos>", "<staff_start>", "x", "<staff_end>", "<eos>"],
        2: ["<bos>", "<staff_start>", "z", "<staff_end>", "<eos>"],
    }

    rows = _build_manifest_rows_for_page(
        page_basename="p",
        staff_crop_entries=staff_crop_entries,
        total_physical_staves=3,
        token_sequences_by_phys=token_sequences_by_phys,
        page_number=1,
        style_id="x",
        score_type="piano",
        source_relpath="src",
        project_root=Path("/tmp"),
        dataset_variants=[("synthetic_fullpage", "")],
    )

    # Only positions 0 and 2 produced rows; position 1 was silently skipped.
    by_idx = {r["staff_index"]: r for r in rows}
    assert sorted(by_idx) == [0, 2]


def test_manifest_rows_emit_per_dataset_variant() -> None:
    """For piano/chamber score_types, manifest writes both 'synthetic_fullpage'
    and 'synthetic_polyphonic' variants per physical position."""
    from src.data.generate_synthetic import _build_manifest_rows_for_page

    staff_crop_entries = [(Path("crop_0.png"), 0), (Path("crop_1.png"), 1)]
    token_sequences_by_phys = {
        i: ["<bos>", "<staff_start>", f"x{i}", "<staff_end>", "<eos>"]
        for i in range(2)
    }

    rows = _build_manifest_rows_for_page(
        page_basename="p",
        staff_crop_entries=staff_crop_entries,
        total_physical_staves=2,
        token_sequences_by_phys=token_sequences_by_phys,
        page_number=1,
        style_id="x",
        score_type="piano",
        source_relpath="src",
        project_root=Path("/tmp"),
        dataset_variants=[("synthetic_fullpage", ""), ("synthetic_polyphonic", "__poly")],
    )

    # 2 physical positions × 2 dataset variants = 4 rows
    assert len(rows) == 4
    datasets = {r["dataset"] for r in rows}
    assert datasets == {"synthetic_fullpage", "synthetic_polyphonic"}
    sample_ids = {r["sample_id"] for r in rows}
    assert "p__staff01" in sample_ids
    assert "p__staff01__poly" in sample_ids
    assert "p__staff02" in sample_ids
    assert "p__staff02__poly" in sample_ids


def test_manifest_rows_no_drop_equivalence_regression_guard() -> None:
    """Regression guard: when zero crops are filtered, every emitted row has a
    non-None image_path and staff_index matches its enumerate position. Pre-fix
    code produced this same shape; post-fix must too. If a future refactor
    breaks the no-drop equivalence, this test fails."""
    from src.data.generate_synthetic import _build_manifest_rows_for_page

    n = 5
    staff_crop_entries = [(Path(f"crop_{i}.png"), i) for i in range(n)]
    token_sequences_by_phys = {
        i: ["<bos>", "<staff_start>", f"phys-{i}", "<staff_end>", "<eos>"]
        for i in range(n)
    }

    rows = _build_manifest_rows_for_page(
        page_basename="page",
        staff_crop_entries=staff_crop_entries,
        total_physical_staves=n,
        token_sequences_by_phys=token_sequences_by_phys,
        page_number=1,
        style_id="x",
        score_type="vocal",
        source_relpath="src",
        project_root=Path("/tmp"),
        dataset_variants=[("synthetic_fullpage", "")],
    )

    # Equivalence properties of a no-drop page:
    #   1. one row per physical position (no drops, no extras)
    assert len(rows) == n
    #   2. every row carries a non-None image_path (no filter rejections)
    assert all(r["image_path"] is not None for r in rows), (
        "no-drop page must produce non-None image_path for every row; "
        "if this fails the producer regressed past the alignment fix"
    )
    #   3. staff_index matches the enumerate position (which equals physical index here)
    for i, r in enumerate(rows):
        assert r["staff_index"] == i
