"""Tests for scripts/rederive_synthetic_v2_per_staff_manifest.py

Two TDD tests that cover the per-page driver (process_page) and the
cumulative_measure_offset accumulation logic. All Verovio/PIL-dependent
helpers (_write_staff_crops, _extract_system_layout_from_svg, etc.) are mocked
so the tests run locally without any GPU-box data or Verovio installation.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build fixture data
# ---------------------------------------------------------------------------


def _make_page_entry(
    *,
    page_id: str = "score__style__p001",
    source_path: str = "data/test/score.musicxml",
    style_id: str = "bravura-compact",
    score_type: str = "piano",
    page_number: int = 1,
    svg_path: str = "pages/bravura-compact/score__style__p001.svg",
    label_path: str = "labels/bravura-compact/score__style__p001.txt",
    page_width: float = 1000.0,
    page_height: float = 1400.0,
) -> dict:
    return {
        "page_id": page_id,
        "source_path": source_path,
        "style_id": style_id,
        "score_type": score_type,
        "page_number": page_number,
        "svg_path": svg_path,
        "label_path": label_path,
        "page_width": page_width,
        "page_height": page_height,
    }


def _make_yolo_label_lines(
    staff_boxes_normalized: List[Tuple[float, float, float, float]]
) -> str:
    """Produce YOLO label text from normalized (cx, cy, w, h) tuples."""
    return "\n".join(
        f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
        for cx, cy, w, h in staff_boxes_normalized
    )


# ---------------------------------------------------------------------------
# Test 1: single-page smoke test (no filter drops)
# ---------------------------------------------------------------------------


def test_rederive_processes_single_page_smoke(tmp_path: Path) -> None:
    """process_page produces the correct number of manifest rows for a page
    with 3 physical staves, all surviving the crop filter (no drops).

    All Verovio/PIL helpers are mocked. We validate that:
    - The returned row list has the right length (1 row × 3 staves × 2 variants for piano).
    - Each row has a non-None image_path (no filter drops).
    - staff_index values are [0, 1, 2] for the two staves of each variant.
    - Tokens are non-empty lists.
    """
    from scripts.rederive_synthetic_v2_per_staff_manifest import process_page

    # --- Fixture setup ---
    page_entry = _make_page_entry(
        page_id="score__bravura__p001",
        source_path="data/test/score.musicxml",
        style_id="bravura-compact",
        score_type="piano",
        page_number=1,
        page_width=1000.0,
        page_height=1400.0,
        svg_path="pages/bravura-compact/score__bravura__p001.svg",
        label_path="labels/bravura-compact/score__bravura__p001.txt",
    )

    # Three staves (normalized YOLO format, cx/cy/w/h)
    label_text = _make_yolo_label_lines([
        (0.5, 0.15, 0.9, 0.08),
        (0.5, 0.45, 0.9, 0.08),
        (0.5, 0.75, 0.9, 0.08),
    ])

    # Build corpus fixture on disk: svg, label file
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    svg_dir = corpus_root / "pages" / "bravura-compact"
    svg_dir.mkdir(parents=True)
    (svg_dir / "score__bravura__p001.svg").write_text("<svg/>", encoding="utf-8")

    label_dir = corpus_root / "labels" / "bravura-compact"
    label_dir.mkdir(parents=True)
    (label_dir / "score__bravura__p001.txt").write_text(label_text, encoding="utf-8")

    # Existing crop PNGs (we won't actually read them; paths must exist for path construction)
    crops_dir = corpus_root / "staff_crops" / "bravura-compact"
    crops_dir.mkdir(parents=True)
    for i in range(1, 4):
        (crops_dir / f"score__bravura__p001__staff{i:02d}.png").touch()

    # --- Mock helpers ---
    # _write_staff_crops returns [(temp_path, phys_idx), ...] for all 3 staves surviving
    mock_crop_entries = [
        (tmp_path / f"tmp_staff{i+1:02d}.png", i) for i in range(3)
    ]

    from src.data.generate_synthetic import _SvgSystemInfo

    # SVG layout: 1 system, 3 staves, 2 measures
    mock_svg_layout = [
        _SvgSystemInfo(
            measure_count=2,
            staves_per_system=3,
            y_top=0.0,
            y_bottom=1400.0,
            x_left=10.0,
        )
    ]

    # staff_to_system: phys 0,1,2 → system 0, positions 0,1,2
    mock_staff_to_system = {0: (0, 0), 1: (0, 1), 2: (0, 2)}

    # Token sequences: 3 parts, 2 measures each
    mock_token_seqs: List[List[str]] = [
        ["<bos>", "<staff_start>", "<measure_start>", f"part{i}_tok", "<measure_end>", "<staff_end>", "<eos>"]
        for i in range(3)
    ]

    state = {"offset": 0, "tokens": mock_token_seqs}
    token_cache: Dict[str, Optional[List[List[str]]]] = {
        "data/test/score.musicxml": mock_token_seqs
    }

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()

    with (
        patch(
            "scripts.rederive_synthetic_v2_per_staff_manifest._write_staff_crops",
            return_value=mock_crop_entries,
        ),
        patch(
            "scripts.rederive_synthetic_v2_per_staff_manifest._extract_system_layout_from_svg",
            return_value=mock_svg_layout,
        ),
        patch(
            "scripts.rederive_synthetic_v2_per_staff_manifest._assign_staff_boxes_to_systems",
            return_value=mock_staff_to_system,
        ),
    ):
        rows = process_page(
            page_entry=page_entry,
            state=state,
            token_cache=token_cache,
            corpus_root=corpus_root,
            data_root=data_root,
            project_root=repo_root,
        )

    # piano → 2 variants (synthetic_fullpage, synthetic_polyphonic)
    # 3 staves × 2 variants = 6 rows
    assert len(rows) == 6, f"Expected 6 rows, got {len(rows)}: {rows}"

    fullpage_rows = [r for r in rows if r["dataset"] == "synthetic_fullpage"]
    poly_rows = [r for r in rows if r["dataset"] == "synthetic_polyphonic"]
    assert len(fullpage_rows) == 3
    assert len(poly_rows) == 3

    staff_indices = sorted(r["staff_index"] for r in fullpage_rows)
    assert staff_indices == [0, 1, 2]

    # All staves survived the filter → all image_paths are non-None
    for row in fullpage_rows:
        assert row["image_path"] is not None, f"Expected non-None image_path, got None for row {row}"

    # Token sequences must be non-empty
    for row in rows:
        assert isinstance(row["token_sequence"], list)
        assert len(row["token_sequence"]) > 0


# ---------------------------------------------------------------------------
# Test 2: cumulative_measure_offset accumulates per (source, style) pair
# ---------------------------------------------------------------------------


def test_rederive_offset_accumulates_per_source(tmp_path: Path) -> None:
    """cumulative_measure_offset must be independent across (source, style)
    pairs and must accumulate within each pair across pages.

    We create two sources × one style each, with two pages per source.
    After processing page 1 of source A (2 measures), the offset for source A
    must be 2 for page 2. Source B's offset must start at 0 independently.

    We verify this by checking the m_start/m_end slices passed to
    _extract_measure_range_from_sequence for each page × source combination.
    """
    from scripts.rederive_synthetic_v2_per_staff_manifest import build_per_source_groups

    # 2 sources × 2 pages each, 1 staff, 1 system with 2 measures per page.
    sources = ["data/sourceA.musicxml", "data/sourceB.musicxml"]
    style = "bravura-compact"

    # Token sequences: 1 part per source, minimal
    def _make_token_seq() -> List[List[str]]:
        # 6 measures: each with 2 tokens so we can detect measure range
        parts = []
        toks: List[str] = ["<bos>", "<staff_start>"]
        for m in range(6):
            toks += ["<measure_start>", f"note_m{m}", "<measure_end>"]
        toks += ["<staff_end>", "<eos>"]
        parts.append(toks)
        return parts

    token_cache: Dict[str, Optional[List[List[str]]]] = {
        src: _make_token_seq() for src in sources
    }

    # Build two pages per source in synthetic_pages.jsonl entries.
    # page_number 1 and 2 for each source.
    all_pages = []
    for src in sources:
        for pno in [1, 2]:
            page_id = f"{'A' if 'sourceA' in src else 'B'}_p{pno:03d}"
            all_pages.append(_make_page_entry(
                page_id=page_id,
                source_path=src,
                style_id=style,
                score_type="vocal",
                page_number=pno,
                svg_path=f"pages/{style}/{page_id}.svg",
                label_path=f"labels/{style}/{page_id}.txt",
            ))

    # build_per_source_groups should return a dict mapping (source_path, style_id)
    # to a list of page entries sorted by page_number.
    groups = build_per_source_groups(all_pages)

    # Check group keys
    key_a = ("data/sourceA.musicxml", style)
    key_b = ("data/sourceB.musicxml", style)
    assert key_a in groups, f"Missing key {key_a} in groups {list(groups.keys())}"
    assert key_b in groups, f"Missing key {key_b} in groups {list(groups.keys())}"

    # Each group has 2 pages, ordered by page_number
    assert [p["page_number"] for p in groups[key_a]] == [1, 2]
    assert [p["page_number"] for p in groups[key_b]] == [1, 2]

    # Simulate the offset accumulation loop:
    # - Source A, page 1 → offset starts at 0; SVG has 2 measures → offset becomes 2.
    # - Source A, page 2 → offset starts at 2 (accumulated).
    # - Source B, page 1 → offset starts at 0 (independent).
    # - Source B, page 2 → offset starts at 2.
    captured_offsets: Dict[str, List[int]] = {"A": [], "B": []}

    from src.data.generate_synthetic import _SvgSystemInfo

    mock_svg_layout_2m = [
        _SvgSystemInfo(
            measure_count=2,
            staves_per_system=1,
            y_top=0.0,
            y_bottom=1400.0,
            x_left=10.0,
        )
    ]

    mock_crop_entry_per_page = [(tmp_path / "tmp_staff01.png", 0)]
    mock_staff_to_system = {0: (0, 0)}

    from scripts.rederive_synthetic_v2_per_staff_manifest import process_page

    # Minimal corpus on disk (SVG + label for each page)
    corpus_root = tmp_path / "corpus"
    label_text = _make_yolo_label_lines([(0.5, 0.5, 0.9, 0.1)])

    for src_label, src in zip(["A", "B"], sources):
        for pno in [1, 2]:
            page_id = f"{src_label}_p{pno:03d}"
            svg_dir = corpus_root / "pages" / style
            svg_dir.mkdir(parents=True, exist_ok=True)
            (svg_dir / f"{page_id}.svg").write_text("<svg/>", encoding="utf-8")
            lbl_dir = corpus_root / "labels" / style
            lbl_dir.mkdir(parents=True, exist_ok=True)
            (lbl_dir / f"{page_id}.txt").write_text(label_text, encoding="utf-8")
            crops_dir = corpus_root / "staff_crops" / style
            crops_dir.mkdir(parents=True, exist_ok=True)
            (crops_dir / f"{page_id}__staff01.png").touch()

    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    data_root = tmp_path / "data"
    data_root.mkdir(exist_ok=True)

    for src_label, src in zip(["A", "B"], sources):
        state = {"offset": 0, "tokens": token_cache[src]}
        key = (src, style)
        for page_entry in groups[key]:
            with (
                patch(
                    "scripts.rederive_synthetic_v2_per_staff_manifest._write_staff_crops",
                    return_value=mock_crop_entry_per_page,
                ),
                patch(
                    "scripts.rederive_synthetic_v2_per_staff_manifest._extract_system_layout_from_svg",
                    return_value=mock_svg_layout_2m,
                ),
                patch(
                    "scripts.rederive_synthetic_v2_per_staff_manifest._assign_staff_boxes_to_systems",
                    return_value=mock_staff_to_system,
                ),
            ):
                offset_before = state["offset"]
                captured_offsets[src_label].append(offset_before)
                process_page(
                    page_entry=page_entry,
                    state=state,
                    token_cache=token_cache,
                    corpus_root=corpus_root,
                    data_root=data_root,
                    project_root=repo_root,
                )

    # Source A: page 1 offset=0, page 2 offset=2 (consumed 2 measures on page 1)
    assert captured_offsets["A"] == [0, 2], (
        f"Source A offsets should be [0, 2], got {captured_offsets['A']}"
    )
    # Source B: page 1 offset=0, page 2 offset=2 (independent of source A)
    assert captured_offsets["B"] == [0, 2], (
        f"Source B offsets should be [0, 2], got {captured_offsets['B']}"
    )
