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


@pytest.fixture
def sample_svg_bytes() -> bytes:
    """The same minimal SVG as raw bytes (no file on disk)."""
    svg = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="8.5in" height="11in" viewBox="0 0 850 1100">
  <rect x="0" y="0" width="850" height="1100" fill="white"/>
  <line x1="100" y1="200" x2="750" y2="200" stroke="black" stroke-width="1"/>
</svg>
"""
    return svg.encode("utf-8")


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


# ---------------------------------------------------------------------------
# New tests for bytes-based helpers
# ---------------------------------------------------------------------------

def test_rasterize_svg_bytes_writes_file(sample_svg_bytes: bytes, tmp_path: Path):
    """rasterize_svg_bytes writes a PNG at the right dimensions."""
    from src.data.multi_dpi import rasterize_svg_bytes

    out = tmp_path / "out_bytes_150.png"
    rasterize_svg_bytes(sample_svg_bytes, out, dpi=150)

    assert out.exists(), "Output PNG was not created"
    img = Image.open(out)
    # 8.5 in x 150 DPI = 1275 px wide, 11 in x 150 = 1650 px tall
    assert 1273 <= img.width <= 1277
    assert 1648 <= img.height <= 1652


def test_rasterize_svg_bytes_matches_file_variant(sample_svg: Path, sample_svg_bytes: bytes, tmp_path: Path):
    """rasterize_svg_bytes produces images of the same size as rasterize_svg."""
    from src.data.multi_dpi import rasterize_svg, rasterize_svg_bytes

    out_file = tmp_path / "from_file.png"
    out_bytes = tmp_path / "from_bytes.png"
    rasterize_svg(sample_svg, out_file, dpi=150)
    rasterize_svg_bytes(sample_svg_bytes, out_bytes, dpi=150)

    img_file = Image.open(out_file)
    img_bytes = Image.open(out_bytes)
    assert img_file.size == img_bytes.size


def test_rasterize_svg_bytes_to_png_bytes_returns_valid_png(sample_svg_bytes: bytes):
    """rasterize_svg_bytes_to_png_bytes returns valid PNG bytes at the right size."""
    from io import BytesIO
    from src.data.multi_dpi import rasterize_svg_bytes_to_png_bytes

    png_data = rasterize_svg_bytes_to_png_bytes(sample_svg_bytes, dpi=96)

    assert isinstance(png_data, bytes)
    assert len(png_data) > 0
    # Must parse as a valid image
    img = Image.open(BytesIO(png_data))
    # 8.5 in x 96 DPI = 816 px wide, 11 in x 96 = 1056 px tall
    assert 814 <= img.width <= 818
    assert 1054 <= img.height <= 1058
