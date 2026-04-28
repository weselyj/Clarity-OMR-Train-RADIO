"""Unit test for src/data/multi_dpi.py — rasterize SVG at multiple DPIs."""
from pathlib import Path

import pytest
from PIL import Image


@pytest.fixture
def sample_svg(tmp_path: Path) -> Path:
    """A minimal valid SVG with declared 8.5x11 inch viewport."""
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="8.5in" height="11in" viewBox="0 0 850 1100">
  <rect x="0" y="0" width="850" height="1100" fill="white"/>
  <line x1="100" y1="200" x2="750" y2="200" stroke="black" stroke-width="1"/>
</svg>
"""
    p = tmp_path / "sample.svg"
    p.write_text(svg)
    return p


def test_rasterize_at_94_dpi(sample_svg: Path, tmp_path: Path):
    from src.data.multi_dpi import rasterize_svg

    out = tmp_path / "out_94.png"
    rasterize_svg(sample_svg, out, dpi=94)

    img = Image.open(out)
    # 8.5 in x 94 DPI = 799px (allow +-2 px for rasterizer rounding)
    assert 797 <= img.width <= 801
    assert 1032 <= img.height <= 1036


def test_rasterize_at_300_dpi(sample_svg: Path, tmp_path: Path):
    from src.data.multi_dpi import rasterize_svg

    out = tmp_path / "out_300.png"
    rasterize_svg(sample_svg, out, dpi=300)

    img = Image.open(out)
    # 8.5 in x 300 DPI = 2550px
    assert 2548 <= img.width <= 2552
    assert 3298 <= img.height <= 3302


def test_rasterize_preserves_aspect_ratio(sample_svg: Path, tmp_path: Path):
    from src.data.multi_dpi import rasterize_svg

    out_94 = tmp_path / "94.png"
    out_300 = tmp_path / "300.png"
    rasterize_svg(sample_svg, out_94, dpi=94)
    rasterize_svg(sample_svg, out_300, dpi=300)

    img94 = Image.open(out_94)
    img300 = Image.open(out_300)
    aspect_94 = img94.width / img94.height
    aspect_300 = img300.width / img300.height
    assert abs(aspect_94 - aspect_300) < 0.001
