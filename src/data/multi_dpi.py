"""Rasterize Verovio-produced SVGs to PNGs at arbitrary DPIs.

The existing generate_synthetic.py uses cairosvg to convert SVG -> PNG once at a
fixed DPI.  On this Windows build machine cairosvg requires libcairo-2.dll which
is not present as a 64-bit native DLL; ImageMagick (installed via Chocolatey) is
the working rasteriser and is used here instead.  The public interface is
identical to what cairosvg would expose, so callers are unaffected.

If cairosvg becomes available (e.g. after installing GTK3 64-bit runtime), the
cairosvg path can be restored:

    scale = dpi / 96.0
    cairosvg.svg2png(url=str(svg_path), write_to=str(out_path), scale=scale)
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# ImageMagick 7 "magick" binary — installed to a fixed path by Chocolatey.
# We also try shutil.which() so the tests work if magick is on PATH.
_MAGICK_DEFAULT = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"

# ---------------------------------------------------------------------------
# SVG pre-processing: stroke injection
# ---------------------------------------------------------------------------

_PATH_TAG_RE = re.compile(rb"<path\b[^>]*?>")


def _ensure_stroke(svg_bytes: bytes) -> bytes:
    """Add stroke="black" to <path> elements that don't already have a stroke attribute.

    Verovio's SVGs encode staff lines as <path d=".." stroke-width="13" /> and
    rely on a CSS rule (`#<id> path {stroke:currentColor}`) to color them.
    ImageMagick's rsvg delegate ignores the rule, so paths render unstroked
    and staff lines disappear. We inject the stroke explicitly to work around it.
    """
    def fix(match: re.Match) -> bytes:
        tag = match.group(0)
        if b"stroke=" in tag:
            return tag
        # Insert stroke="black" right after "<path"
        return tag[:5] + b' stroke="black"' + tag[5:]
    return _PATH_TAG_RE.sub(fix, svg_bytes)


def _magick_exe() -> str:
    on_path = shutil.which("magick")
    if on_path:
        return on_path
    return _MAGICK_DEFAULT


def rasterize_svg(svg_path: Path, out_path: Path, dpi: int) -> None:
    """Render *svg_path* to *out_path* at the requested DPI.

    Uses ImageMagick's librsvg/cairo delegate for pixel-accurate rendering.
    The output dimensions will be approximately ``width_in * dpi`` x
    ``height_in * dpi`` (±1 px for sub-pixel rounding).
    """
    # Read, patch, and delegate to the bytes-based helper so stroke injection
    # is applied consistently regardless of which entry point is used.
    svg_bytes = Path(svg_path).read_bytes()
    rasterize_svg_bytes(svg_bytes, out_path, dpi)


def rasterize_svg_bytes(svg_bytes: bytes, out_path: Path, dpi: int) -> None:
    """Render SVG content supplied as *svg_bytes* to *out_path* at the requested DPI.

    Mirrors ``rasterize_svg`` but accepts raw bytes instead of a file path.
    The bytes are written to a temporary file so ImageMagick can detect the SVG
    format via the file header (piping via stdin does not reliably set the input
    format on all ImageMagick builds).

    Applies ``_ensure_stroke`` before rendering so that Verovio staff-line paths
    (which carry stroke-width but no stroke attribute) are visible in the output.
    """
    svg_bytes = _ensure_stroke(svg_bytes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(svg_bytes)
    try:
        cmd = [
            _magick_exe(),
            "-density", str(dpi),
            str(tmp_path),
            "-background", "white",
            "-alpha", "remove",
            str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ImageMagick failed (exit {result.returncode}):\n"
                f"  cmd: {cmd}\n"
                f"  stderr: {result.stderr.strip()}"
            )
    finally:
        tmp_path.unlink(missing_ok=True)


def rasterize_svg_bytes_to_png_bytes(svg_bytes: bytes, dpi: int = 96) -> bytes:
    """Render SVG content supplied as *svg_bytes* and return raw PNG bytes.

    Drop-in replacement for::

        cairosvg.svg2png(bytestring=svg_bytes, background_color="white")

    The PNG is written to a temporary file and read back so the caller gets a
    plain ``bytes`` object without any open file handle.

    Applies ``_ensure_stroke`` (via ``rasterize_svg_bytes``) so that Verovio
    staff-line paths are rendered with visible strokes.
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        rasterize_svg_bytes(svg_bytes, tmp_path, dpi)
        return tmp_path.read_bytes()
    finally:
        tmp_path.unlink(missing_ok=True)
