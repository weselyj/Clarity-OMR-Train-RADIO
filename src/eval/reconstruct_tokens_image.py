#!/usr/bin/env python3
"""Reconstruct diagnostic images from token sequences and compare with references."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_RENDER_OPTIONS = {
    "font": "Bravura",
    "pageWidth": 2200,
    "pageHeight": 3000,
    "scale": 40,
    "breaks": "auto",
    "svgBoundingBoxes": True,
    "svgViewBox": True,
}


def _resolve_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
            if not isinstance(row, dict):
                continue
            rows.append(row)
    return rows


def _slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def _normalize_token_sequence(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(token) for token in value]
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(token) for token in parsed]
    raise ValueError("Token sequence must be a JSON list of token strings.")


def _build_side_by_side(
    *,
    crop_path: Path,
    recon_path: Path,
    output_path: Path,
    label_left: str = "page crop",
    label_right: str = "token reconstruction",
) -> Optional[Path]:
    """Create a simple side-by-side comparison: [page crop | reconstruction]."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    crop_img = Image.open(crop_path).convert("RGB")
    recon_img = Image.open(recon_path).convert("RGB")

    # Scale reconstruction to match crop height
    target_h = crop_img.height
    r_w = max(1, int(recon_img.width * target_h / max(recon_img.height, 1)))
    recon_scaled = recon_img.resize((r_w, target_h), Image.Resampling.LANCZOS)

    gap = 10
    label_h = 20
    canvas_w = crop_img.width + gap + r_w
    canvas_h = label_h + target_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    draw.text((2, 1), label_left, fill=(0, 0, 0))
    draw.text((crop_img.width + gap + 2, 1), label_right, fill=(100, 100, 100))

    canvas.paste(crop_img, (0, label_h))
    canvas.paste(recon_scaled, (crop_img.width + gap, label_h))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _render_single_staff_png(
    token_sequence: Sequence[str],
    output_png_path: Path,
    *,
    render_options: Dict[str, object],
    part_id: str = "staff",
) -> bool:
    """Render a single staff's token sequence to PNG via music21 + Verovio.

    Uses ``breaks: "none"`` with a very wide page so that all measures
    appear on a single horizontal system – matching how staff crops look
    on the rendered page.  If Verovio still produces multiple pages (e.g.
    when content is extremely long), all pages are stitched left-to-right.

    Returns True on success, False on failure.
    """
    try:
        import cairosvg
        import verovio
        from music21 import stream

        from src.pipeline.export_musicxml import append_tokens_to_part
    except ImportError:
        return False

    try:
        score = stream.Score(id=f"{part_id}_score")
        part = stream.Part(id=part_id)
        append_tokens_to_part(part, list(token_sequence))
        score.append(part)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".musicxml", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        score.write("musicxml", fp=str(tmp_path))

        toolkit = verovio.toolkit()
        toolkit.loadFile(str(tmp_path))
        # Force a single horizontal system: keep original pageWidth so
        # note spacing matches the page crop, but use "none" breaks so
        # Verovio extends beyond the page boundary rather than wrapping.
        # A very tall page prevents any page-break.
        opts = dict(render_options)
        opts["pageHeight"] = 60000
        opts["breaks"] = "none"
        toolkit.setOptions(opts)
        toolkit.redoLayout()

        page_count = int(toolkit.getPageCount())
        if page_count < 1:
            tmp_path.unlink(missing_ok=True)
            return False

        output_png_path.parent.mkdir(parents=True, exist_ok=True)

        if page_count == 1:
            svg_text = toolkit.renderToSVG(1)
            cairosvg.svg2png(
                bytestring=svg_text.encode("utf-8"),
                write_to=str(output_png_path),
                background_color="white",
            )
        else:
            # Stitch all pages left-to-right (should be rare with the
            # large dimensions, but keep as a safety net).
            from PIL import Image
            import io

            page_images = []
            for pg in range(1, page_count + 1):
                svg_text = toolkit.renderToSVG(pg)
                png_bytes = cairosvg.svg2png(
                    bytestring=svg_text.encode("utf-8"),
                    background_color="white",
                )
                page_images.append(Image.open(io.BytesIO(png_bytes)).convert("RGB"))

            total_w = sum(img.width for img in page_images)
            max_h = max(img.height for img in page_images)
            stitched = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))
            x_off = 0
            for img in page_images:
                stitched.paste(img, (x_off, 0))
                x_off += img.width
            stitched.save(output_png_path)

        tmp_path.unlink(missing_ok=True)
        return True
    except Exception:
        return False


def _build_staff_comparison_grid(
    staff_rows: Sequence[Dict[str, object]],
    *,
    project_root: Path,
    output_dir: Path,
    prefix: str,
    render_options: Dict[str, object],
) -> Optional[Path]:
    """Build a comparison grid: each row = [page crop | token reconstruction].

    Returns the path to the saved grid image, or None on failure.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None

    panels: List[Tuple["Image.Image", "Image.Image", int]] = []  # (crop, recon, staff_idx)
    recon_dir = output_dir / f"{prefix}_staff_recons"
    recon_dir.mkdir(parents=True, exist_ok=True)

    for row in staff_rows:
        staff_idx = int(row.get("staff_index", 0) or 0)
        # Load original staff crop from the page
        crop_path = _resolve_path(project_root, str(row.get("image_path", "")))
        if crop_path is None or not crop_path.exists():
            continue
        crop_img = Image.open(crop_path).convert("RGB")

        # Render token reconstruction
        tokens = _normalize_token_sequence(row.get("token_sequence", []))
        recon_path = recon_dir / f"staff{staff_idx:02d}_recon.png"
        ok = _render_single_staff_png(
            tokens, recon_path, render_options=render_options, part_id=f"s{staff_idx}",
        )
        if not ok or not recon_path.exists():
            # If rendering failed, create a placeholder
            recon_img = Image.new("RGB", crop_img.size, color=(240, 240, 240))
        else:
            recon_img = Image.open(recon_path).convert("RGB")

        panels.append((crop_img, recon_img, staff_idx))

    if not panels:
        return None

    # Normalise widths: scale all panels to a common width
    target_w = max(max(c.width, r.width) for c, r, _ in panels)
    gap = 10  # horizontal gap between crop and reconstruction
    label_h = 20  # height for label text
    row_gap = 6  # vertical gap between rows

    rows: List["Image.Image"] = []
    for crop_img, recon_img, staff_idx in panels:
        # Scale crop to target width, preserve aspect
        c_h = max(1, int(crop_img.height * target_w / max(crop_img.width, 1)))
        crop_scaled = crop_img.resize((target_w, c_h), Image.Resampling.LANCZOS)
        # Scale reconstruction to same height as crop
        r_w = max(1, int(recon_img.width * c_h / max(recon_img.height, 1)))
        recon_scaled = recon_img.resize((r_w, c_h), Image.Resampling.LANCZOS)

        row_w = target_w + gap + r_w
        row_h = label_h + c_h
        row_img = Image.new("RGB", (row_w, row_h), color=(255, 255, 255))
        # Draw label
        draw = ImageDraw.Draw(row_img)
        draw.text((2, 1), f"Staff {staff_idx} — page crop", fill=(0, 0, 0))
        draw.text((target_w + gap + 2, 1), "token reconstruction", fill=(100, 100, 100))
        # Paste images
        row_img.paste(crop_scaled, (0, label_h))
        row_img.paste(recon_scaled, (target_w + gap, label_h))
        rows.append(row_img)

    # Stack rows vertically
    total_w = max(r.width for r in rows)
    total_h = sum(r.height for r in rows) + row_gap * (len(rows) - 1)
    grid = Image.new("RGB", (total_w, total_h), color=(255, 255, 255))
    y_offset = 0
    for row_img in rows:
        grid.paste(row_img, (0, y_offset))
        y_offset += row_img.height + row_gap

    grid_path = output_dir / f"{prefix}_staff_grid.png"
    grid.save(grid_path)
    return grid_path


def _resolve_render_options(
    *,
    page_manifest_entry: Optional[Dict[str, object]],
    font: Optional[str],
    page_width: Optional[int],
    page_height: Optional[int],
    scale: Optional[int],
) -> Dict[str, object]:
    options = dict(DEFAULT_RENDER_OPTIONS)
    if page_manifest_entry is not None and isinstance(page_manifest_entry.get("render_options"), dict):
        options.update(page_manifest_entry["render_options"])  # type: ignore[index]
    if font:
        options["font"] = font
    if page_width:
        options["pageWidth"] = int(page_width)
    if page_height:
        options["pageHeight"] = int(page_height)
    if scale:
        options["scale"] = int(scale)
    return options


def _find_page_manifest_entry(page_rows: Sequence[Dict[str, object]], page_id: str) -> Optional[Dict[str, object]]:
    for row in page_rows:
        if str(row.get("page_id", "")) == page_id:
            return row
    return None


def _dedupe_page_entries(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    deduped: List[Dict[str, object]] = []
    seen: set[Tuple[object, object, object]] = set()
    sorted_rows = sorted(
        rows,
        key=lambda item: (
            int(item.get("staff_index", 0) or 0),
            str(item.get("sample_id", "")),
        ),
    )
    for row in sorted_rows:
        key = (
            row.get("staff_index"),
            row.get("image_path"),
            json.dumps(row.get("token_sequence", []), ensure_ascii=False),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Reconstruct image(s) from token sequence(s) and compare with reference crop/page."
    )
    parser.add_argument("--project-root", type=Path, default=project_root, help="Repository root path.")
    parser.add_argument(
        "--token-manifest",
        type=Path,
        default=project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl",
        help="Token manifest JSONL path.",
    )
    parser.add_argument(
        "--page-manifest",
        type=Path,
        default=project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_pages.jsonl",
        help="Synthetic page manifest JSONL path (used for page-level reference and render options).",
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--sample-id", type=str, help="Single token-manifest sample_id to reconstruct.")
    target_group.add_argument("--page-id", type=str, help="Reconstruct from all token entries for this page_id.")
    target_group.add_argument(
        "--tokens-json",
        type=str,
        help="Either a JSON token-list string or path to a .json file containing a token list.",
    )
    target_group.add_argument(
        "--random-samples",
        type=int,
        metavar="N",
        help="Pick N random pages and render per-staff comparison grids for each.",
    )
    parser.add_argument(
        "--reference-image",
        type=str,
        default=None,
        help="Optional explicit reference image path (crop or page PNG).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "src" / "eval" / "reconstruct_debug",
        help="Output directory for reconstructed artifacts.",
    )
    parser.add_argument("--output-prefix", type=str, default=None, help="Optional output filename prefix.")
    parser.add_argument("--font", type=str, default=None, help="Optional Verovio font override.")
    parser.add_argument("--page-width", type=int, default=None, help="Optional Verovio pageWidth override.")
    parser.add_argument("--page-height", type=int, default=None, help="Optional Verovio pageHeight override.")
    parser.add_argument("--scale", type=int, default=None, help="Optional Verovio scale override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    token_manifest = args.token_manifest.resolve()
    page_manifest = args.page_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    token_sequences: List[List[str]] = []
    reference_image_path: Optional[Path] = _resolve_path(project_root, args.reference_image)
    page_entry: Optional[Dict[str, object]] = None
    source_info: Dict[str, object] = {}
    page_staff_rows: Optional[List[Dict[str, object]]] = None

    # --random-samples: pick N random pages and render a grid for each
    if args.random_samples:
        import random

        n = args.random_samples
        token_rows = _load_jsonl(token_manifest)
        page_manifest_rows = _load_jsonl(page_manifest) if page_manifest.exists() else []

        # Collect unique page_ids
        page_ids = sorted({str(row.get("page_id", "")) for row in token_rows if row.get("page_id")})
        if not page_ids:
            raise ValueError("No page_ids found in token manifest.")
        chosen = random.sample(page_ids, min(n, len(page_ids)))

        render_options = _resolve_render_options(
            page_manifest_entry=None,
            font=args.font,
            page_width=args.page_width,
            page_height=args.page_height,
            scale=args.scale,
        )

        grids: List[str] = []
        for i, page_id in enumerate(chosen, 1):
            rows_for_page = _dedupe_page_entries(
                [r for r in token_rows if str(r.get("page_id", "")) == page_id]
            )
            if not rows_for_page:
                continue
            # Use page-specific render options if available
            pe = _find_page_manifest_entry(page_manifest_rows, page_id)
            opts = _resolve_render_options(
                page_manifest_entry=pe,
                font=args.font,
                page_width=args.page_width,
                page_height=args.page_height,
                scale=args.scale,
            )
            prefix = _slugify(page_id)
            grid_path = _build_staff_comparison_grid(
                rows_for_page,
                project_root=project_root,
                output_dir=output_dir,
                prefix=prefix,
                render_options=opts,
            )
            status = "ok" if grid_path else "FAILED"
            print(f"[{i}/{len(chosen)}] {page_id}  {status}")
            if grid_path:
                grids.append(str(grid_path))

        summary_path = output_dir / "random_samples_summary.json"
        summary_path.write_text(
            json.dumps({"samples": len(chosen), "grids": grids}, indent=2),
            encoding="utf-8",
        )
        print(f"\n{len(grids)} grids written to {output_dir}")
        return

    if args.tokens_json:
        token_json_arg = args.tokens_json
        candidate_path = _resolve_path(project_root, token_json_arg)
        if candidate_path is not None and candidate_path.exists():
            payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        else:
            payload = json.loads(token_json_arg)
        token_sequences = [_normalize_token_sequence(payload)]
        source_info = {"mode": "tokens_json"}
    else:
        token_rows = _load_jsonl(token_manifest)
        if args.sample_id:
            match = next((row for row in token_rows if str(row.get("sample_id", "")) == args.sample_id), None)
            if match is None:
                raise ValueError(f"sample_id not found in token manifest: {args.sample_id}")
            token_sequences = [_normalize_token_sequence(match.get("token_sequence", []))]
            source_info = {
                "mode": "sample_id",
                "sample_id": args.sample_id,
                "dataset": match.get("dataset"),
                "page_id": match.get("page_id"),
            }
            if page_manifest.exists():
                page_manifest_rows = _load_jsonl(page_manifest)
                match_page_id = str(match.get("page_id", ""))
                if match_page_id:
                    page_entry = _find_page_manifest_entry(page_manifest_rows, match_page_id)
            if reference_image_path is None:
                reference_image_path = _resolve_path(project_root, str(match.get("image_path", "")))
        elif args.page_id:
            page_rows = [row for row in token_rows if str(row.get("page_id", "")) == args.page_id]
            if not page_rows:
                raise ValueError(f"page_id not found in token manifest: {args.page_id}")
            page_rows = _dedupe_page_entries(page_rows)
            token_sequences = [_normalize_token_sequence(row.get("token_sequence", [])) for row in page_rows]
            source_info = {
                "mode": "page_id",
                "page_id": args.page_id,
                "staff_sequences": len(token_sequences),
                "datasets": sorted({str(row.get("dataset", "")) for row in page_rows}),
            }
            if page_manifest.exists():
                page_manifest_rows = _load_jsonl(page_manifest)
                page_entry = _find_page_manifest_entry(page_manifest_rows, args.page_id)
                if reference_image_path is None and page_entry is not None:
                    reference_image_path = _resolve_path(project_root, str(page_entry.get("png_path", "")))
            # Store rows for per-staff grid comparison
            page_staff_rows = page_rows
        else:
            raise ValueError("Either --sample-id, --page-id, --tokens-json, or --random-samples must be provided.")

    if not token_sequences:
        raise ValueError("No token sequence available for reconstruction.")

    if args.output_prefix:
        prefix = _slugify(args.output_prefix)
    elif args.sample_id:
        prefix = _slugify(args.sample_id)
    elif args.page_id:
        prefix = _slugify(args.page_id)
    else:
        prefix = "tokens"

    render_options = _resolve_render_options(
        page_manifest_entry=page_entry,
        font=args.font,
        page_width=args.page_width,
        page_height=args.page_height,
        scale=args.scale,
    )

    # --page-id mode: per-staff comparison grid (page crop vs token reconstruction)
    if page_staff_rows is not None:
        grid_path = _build_staff_comparison_grid(
            page_staff_rows,
            project_root=project_root,
            output_dir=output_dir,
            prefix=prefix,
            render_options=render_options,
        )
        summary: Dict[str, object] = {
            "source": source_info,
            "token_manifest": str(token_manifest),
            "page_manifest": str(page_manifest) if page_manifest.exists() else None,
            "token_sequence_count": len(token_sequences),
            "staff_grid": str(grid_path) if grid_path is not None else None,
            "reference_image": str(reference_image_path) if reference_image_path is not None else None,
            "render_options": render_options,
        }
        summary_path = output_dir / f"{prefix}_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    # --sample-id / --tokens-json mode: single-staff reconstruction
    # Render just the staff tokens as a compact single-staff image,
    # then compare side-by-side with the actual page crop.
    recon_png_path = output_dir / f"{prefix}_reconstruction.png"
    _render_single_staff_png(
        token_sequences[0],
        recon_png_path,
        render_options=render_options,
        part_id=prefix,
    )

    comparison_path = None
    if reference_image_path is not None and reference_image_path.exists() and recon_png_path.exists():
        comparison_path = _build_side_by_side(
            crop_path=reference_image_path,
            recon_path=recon_png_path,
            output_path=output_dir / f"{prefix}_comparison.png",
            label_left="page crop",
            label_right="token reconstruction",
        )

    summary = {
        "source": source_info,
        "token_manifest": str(token_manifest),
        "page_manifest": str(page_manifest) if page_manifest.exists() else None,
        "token_sequence_count": len(token_sequences),
        "reconstruction_image": str(recon_png_path) if recon_png_path.exists() else None,
        "reference_image": str(reference_image_path) if reference_image_path is not None else None,
        "comparison_image": str(comparison_path) if comparison_path is not None else None,
        "render_options": render_options,
    }

    summary_path = output_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

