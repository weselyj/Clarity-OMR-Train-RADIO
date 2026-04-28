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

import shutil
import subprocess
from pathlib import Path

# ImageMagick 7 "magick" binary — installed to a fixed path by Chocolatey.
# We also try shutil.which() so the tests work if magick is on PATH.
_MAGICK_DEFAULT = r"C:\Program Files\ImageMagick-7.1.2-Q16-HDRI\magick.exe"


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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        _magick_exe(),
        "-density", str(dpi),
        str(svg_path),
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
