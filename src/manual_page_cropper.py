#!/usr/bin/env python3
"""Interactive multi-page bar crop editor for PDF page images."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CropRect:
    left: int
    top: int
    right: int
    bottom: int

    def clamp(self, width: int, height: int) -> "CropRect":
        left = max(0, min(self.left, width - 1))
        top = max(0, min(self.top, height - 1))
        right = max(left + 1, min(self.right, width))
        bottom = max(top + 1, min(self.bottom, height))
        return CropRect(left=left, top=top, right=right, bottom=bottom)

    def contains(self, x: int, y: int) -> bool:
        return self.left <= x <= self.right and self.top <= y <= self.bottom

    def to_dict(self) -> Dict[str, int]:
        return {
            "left": int(self.left),
            "top": int(self.top),
            "right": int(self.right),
            "bottom": int(self.bottom),
        }


def _bar_reading_order_key(rect: CropRect) -> Tuple[int, int, int, int]:
    return (rect.top, rect.left, rect.bottom, rect.right)


class _ManualCropEditor:
    def __init__(self, page_images: Sequence[Path]) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("tkinter is required for manual page cropping.") from exc
        try:
            from PIL import Image, ImageTk
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required for manual page cropping.") from exc

        self._tk = tk
        self._ttk = ttk
        self._image_lib = Image
        self._image_tk = ImageTk
        self.page_images = [Path(path) for path in page_images]
        if not self.page_images:
            raise ValueError("At least one page image is required for manual cropping.")

        self.root = tk.Tk()
        self.root.title("PDF to MusicXML Manual Bar Crop")
        self.root.geometry("1280x900")
        self.root.minsize(900, 700)

        self.current_index = 0
        self.current_image = None
        self.current_photo = None
        self.current_scale = 1.0
        self.current_offset = (0.0, 0.0)
        self.drag_start: Optional[Tuple[int, int]] = None
        self.preview_rect: Optional[CropRect] = None
        self.crop_rects: List[List[CropRect]] = [[] for _ in self.page_images]
        self.selected_rect_index: Optional[int] = None
        self.cancelled = True

        self.status_var = tk.StringVar()
        self.help_var = tk.StringVar(
            value=(
                "Drag to add a bar. Right-click a bar to select it. "
                "Delete removes the selected bar, Enter saves."
            )
        )

        self._build_ui()
        self._bind_events()
        self._load_page(0)

    def _build_ui(self) -> None:
        top = self._ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        self._ttk.Label(top, textvariable=self.status_var).pack(side="left")
        self._ttk.Label(top, textvariable=self.help_var).pack(side="right")

        self.canvas = self._tk.Canvas(self.root, bg="#202020", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        controls = self._ttk.Frame(self.root, padding=(10, 0, 10, 10))
        controls.pack(fill="x")
        self._ttk.Button(controls, text="Prev", command=self.prev_page).pack(side="left")
        self._ttk.Button(controls, text="Next", command=self.next_page).pack(side="left", padx=(6, 0))
        self._ttk.Button(controls, text="Remove Selected Bar", command=self.remove_selected).pack(side="left", padx=(12, 0))
        self._ttk.Button(controls, text="Clear Page Bars", command=self.reset_current).pack(side="left", padx=(6, 0))
        self._ttk.Button(controls, text="Copy Prev", command=self.copy_previous).pack(side="left", padx=(6, 0))
        self._ttk.Button(controls, text="Apply To Remaining", command=self.apply_to_remaining).pack(side="left", padx=(6, 0))
        self._ttk.Button(controls, text="Save Bars", command=self.save_and_close).pack(side="right")
        self._ttk.Button(controls, text="Cancel", command=self.cancel_and_close).pack(side="right", padx=(0, 6))

    def _bind_events(self) -> None:
        self.root.protocol("WM_DELETE_WINDOW", self.cancel_and_close)
        self.root.bind("<Left>", lambda _event: self.prev_page())
        self.root.bind("<Right>", lambda _event: self.next_page())
        self.root.bind("<Return>", lambda _event: self.save_and_close())
        self.root.bind("<Delete>", lambda _event: self.remove_selected())
        self.root.bind("<BackSpace>", lambda _event: self.remove_selected())
        self.canvas.bind("<Configure>", lambda _event: self._redraw())
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)
        self.canvas.bind("<ButtonRelease-1>", self._on_drag_end)
        self.canvas.bind("<ButtonPress-3>", self._on_right_click)

    def _load_page(self, index: int) -> None:
        self.current_index = max(0, min(index, len(self.page_images) - 1))
        with self._image_lib.open(self.page_images[self.current_index]) as image_obj:
            self.current_image = image_obj.convert("RGB")
        self.preview_rect = None
        self.selected_rect_index = None
        self._update_status()
        self._redraw()

    def _current_page_rects(self) -> List[CropRect]:
        return self.crop_rects[self.current_index]

    def _sort_current_page_rects(self) -> None:
        rects = self._current_page_rects()
        rects.sort(key=_bar_reading_order_key)

    def _update_status(self) -> None:
        path = self.page_images[self.current_index]
        count = len(self._current_page_rects())
        selected = (
            f", selected #{self.selected_rect_index + 1}"
            if self.selected_rect_index is not None and self.selected_rect_index < count
            else ""
        )
        crop_label = "full page" if count == 0 else f"{count} bar(s){selected}"
        self.status_var.set(f"Page {self.current_index + 1}/{len(self.page_images)}: {path.name} ({crop_label})")

    def _fit_image(self, width: int, height: int) -> Tuple[int, int, float, float, float]:
        if self.current_image is None:
            return 1, 1, 1.0, 0.0, 0.0
        canvas_w = max(1, width)
        canvas_h = max(1, height)
        scale = min(canvas_w / self.current_image.width, canvas_h / self.current_image.height)
        scale = max(1e-6, scale)
        disp_w = max(1, int(round(self.current_image.width * scale)))
        disp_h = max(1, int(round(self.current_image.height * scale)))
        offset_x = (canvas_w - disp_w) / 2.0
        offset_y = (canvas_h - disp_h) / 2.0
        return disp_w, disp_h, scale, offset_x, offset_y

    def _draw_crop_rect(self, rect: CropRect, index: int, *, selected: bool, preview: bool = False) -> None:
        x0, y0 = self._image_to_canvas(rect.left, rect.top)
        x1, y1 = self._image_to_canvas(rect.right, rect.bottom)
        outline = "#ffb000" if selected else "#4ecdc4"
        width = 3 if selected else 2
        if preview:
            outline = "#ff6b6b"
            width = 2
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=width)
        label_text = "new" if preview else f"bar {index + 1}"
        self.canvas.create_text(
            x0 + 10,
            y0 + 10,
            text=label_text,
            anchor="nw",
            fill="#ffffff",
            font=("Segoe UI", 11, "bold"),
        )

    def _redraw(self) -> None:
        if self.current_image is None:
            return
        canvas_w = max(1, self.canvas.winfo_width())
        canvas_h = max(1, self.canvas.winfo_height())
        disp_w, disp_h, scale, offset_x, offset_y = self._fit_image(canvas_w, canvas_h)
        resized = self.current_image.resize((disp_w, disp_h), self._image_lib.Resampling.BILINEAR)
        self.current_photo = self._image_tk.PhotoImage(resized)
        self.current_scale = scale
        self.current_offset = (offset_x, offset_y)

        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, anchor="nw", image=self.current_photo)

        for idx, rect in enumerate(self._current_page_rects()):
            self._draw_crop_rect(rect, idx, selected=(idx == self.selected_rect_index))
        if self.preview_rect is not None:
            self._draw_crop_rect(self.preview_rect, len(self._current_page_rects()), selected=False, preview=True)

        self._update_status()

    def _canvas_to_image(self, x: float, y: float) -> Tuple[int, int]:
        if self.current_image is None:
            return (0, 0)
        offset_x, offset_y = self.current_offset
        image_x = int(round((x - offset_x) / self.current_scale))
        image_y = int(round((y - offset_y) / self.current_scale))
        image_x = max(0, min(image_x, self.current_image.width))
        image_y = max(0, min(image_y, self.current_image.height))
        return (image_x, image_y)

    def _image_to_canvas(self, x: int, y: int) -> Tuple[float, float]:
        offset_x, offset_y = self.current_offset
        return (offset_x + (x * self.current_scale), offset_y + (y * self.current_scale))

    def _rect_from_points(self, start: Tuple[int, int], end: Tuple[int, int]) -> Optional[CropRect]:
        if self.current_image is None:
            return None
        left = min(start[0], end[0])
        top = min(start[1], end[1])
        right = max(start[0], end[0])
        bottom = max(start[1], end[1])
        if (right - left) < 4 or (bottom - top) < 4:
            return None
        return CropRect(left, top, right, bottom).clamp(self.current_image.width, self.current_image.height)

    def _find_rect_at_point(self, point: Tuple[int, int]) -> Optional[int]:
        rects = self._current_page_rects()
        for idx in range(len(rects) - 1, -1, -1):
            if rects[idx].contains(point[0], point[1]):
                return idx
        return None

    def _on_drag_start(self, event) -> None:
        self.drag_start = self._canvas_to_image(event.x, event.y)
        self.preview_rect = None

    def _on_drag_move(self, event) -> None:
        if self.drag_start is None:
            return
        end = self._canvas_to_image(event.x, event.y)
        self.preview_rect = self._rect_from_points(self.drag_start, end)
        self._redraw()

    def _on_drag_end(self, event) -> None:
        if self.drag_start is None:
            return
        end = self._canvas_to_image(event.x, event.y)
        rect = self._rect_from_points(self.drag_start, end)
        if rect is not None:
            self._current_page_rects().append(rect)
            self._sort_current_page_rects()
            self.selected_rect_index = self._current_page_rects().index(rect)
        self.drag_start = None
        self.preview_rect = None
        self._redraw()

    def _on_right_click(self, event) -> None:
        point = self._canvas_to_image(event.x, event.y)
        self.selected_rect_index = self._find_rect_at_point(point)
        self._redraw()

    def prev_page(self) -> None:
        if self.current_index > 0:
            self._load_page(self.current_index - 1)

    def next_page(self) -> None:
        if self.current_index < (len(self.page_images) - 1):
            self._load_page(self.current_index + 1)

    def remove_selected(self) -> None:
        rects = self._current_page_rects()
        if not rects:
            return
        if self.selected_rect_index is None or self.selected_rect_index >= len(rects):
            self.selected_rect_index = len(rects) - 1
        rects.pop(self.selected_rect_index)
        if not rects:
            self.selected_rect_index = None
        else:
            self.selected_rect_index = min(self.selected_rect_index, len(rects) - 1)
        self._redraw()

    def reset_current(self) -> None:
        self.crop_rects[self.current_index] = []
        self.selected_rect_index = None
        self.preview_rect = None
        self._redraw()

    def copy_previous(self) -> None:
        if self.current_index <= 0 or self.current_image is None:
            return
        previous = self.crop_rects[self.current_index - 1]
        self.crop_rects[self.current_index] = [
            rect.clamp(self.current_image.width, self.current_image.height) for rect in previous
        ]
        self._sort_current_page_rects()
        self.selected_rect_index = len(self.crop_rects[self.current_index]) - 1 if previous else None
        self._redraw()

    def apply_to_remaining(self) -> None:
        current = list(self._current_page_rects())
        for index in range(self.current_index + 1, len(self.page_images)):
            with self._image_lib.open(self.page_images[index]) as image_obj:
                self.crop_rects[index] = [
                    rect.clamp(image_obj.width, image_obj.height) for rect in current
                ]
                self.crop_rects[index].sort(key=_bar_reading_order_key)
        self._redraw()

    def save_and_close(self) -> None:
        self.cancelled = False
        self.root.destroy()

    def cancel_and_close(self) -> None:
        self.cancelled = True
        self.root.destroy()

    def run(self) -> List[List[CropRect]]:
        self.root.mainloop()
        if self.cancelled:
            raise RuntimeError("Manual page cropping was cancelled.")
        return [list(rects) for rects in self.crop_rects]


def crop_pages_with_editor(
    page_images: Sequence[Path],
    output_dir: Path,
    *,
    metadata_path: Optional[Path] = None,
) -> List[Path]:
    """Open a manual bar window, then export cropped page images."""
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    page_crop_rects = _ManualCropEditor(page_images).run()

    exported_pages: List[Path] = []
    metadata_rows: List[Dict[str, object]] = []
    for page_index, page_path in enumerate(page_images):
        base_name = Path(page_path).stem
        suffix = Path(page_path).suffix
        with Image.open(page_path) as image_obj:
            rects = sorted(page_crop_rects[page_index], key=_bar_reading_order_key)
            if not rects:
                output_path = output_dir / f"{base_name}{suffix}"
                image_obj.copy().save(output_path)
                exported_pages.append(output_path)
                metadata_rows.append(
                    {
                        "page_index": page_index,
                        "bar_index": None,
                        "source_path": str(Path(page_path).resolve()),
                        "output_path": str(output_path.resolve()),
                        "bar_bbox": None,
                    }
                )
                continue

            for bar_index, rect in enumerate(rects):
                applied_rect = rect.clamp(image_obj.width, image_obj.height)
                output_path = output_dir / f"{base_name}__bar{bar_index + 1:02d}{suffix}"
                image_obj.crop(
                    (applied_rect.left, applied_rect.top, applied_rect.right, applied_rect.bottom)
                ).save(output_path)
                exported_pages.append(output_path)
                metadata_rows.append(
                    {
                        "page_index": page_index,
                        "bar_index": bar_index,
                        "source_path": str(Path(page_path).resolve()),
                        "output_path": str(output_path.resolve()),
                        "bar_bbox": applied_rect.to_dict(),
                    }
                )

    if metadata_path is not None:
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(metadata_rows, indent=2), encoding="utf-8")

    return exported_pages
