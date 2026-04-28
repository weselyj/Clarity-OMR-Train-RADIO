"""Visual sanity check: render bbox overlays on 5 random images.

Output: data/processed/omr_layout_real/sanity_check/<filename>.png with red boxes.

Run after convert_omr_layout.py has completed. Eyeball the output images: each
red box should wrap a single staff (one horizontal strip), not a whole system.
"""
import random
from pathlib import Path

from PIL import Image, ImageDraw

IMAGES = Path("data/processed/omr_layout_real/images")
LABELS = Path("data/processed/omr_layout_real/labels")
OUT = Path("data/processed/omr_layout_real/sanity_check")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(42)

    candidates = sorted(IMAGES.iterdir())
    if not candidates:
        print("No images found in", IMAGES)
        return

    sample = random.sample(candidates, min(5, len(candidates)))
    written = 0
    for img_path in sample:
        lbl_path = LABELS / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"  WARNING: no label for {img_path.name}, skipping")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        lines = lbl_path.read_text().strip().splitlines()
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                continue
            _, xc, yc, w, h = map(float, parts)
            x1 = (xc - w / 2) * img.width
            y1 = (yc - h / 2) * img.height
            x2 = (xc + w / 2) * img.width
            y2 = (yc + h / 2) * img.height
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        out_path = OUT / img_path.name
        img.save(out_path)
        print(f"  {img_path.name}: {len(lines)} staves -> {out_path}")
        written += 1

    print(f"\nWrote {written} sanity-check images to {OUT}")


if __name__ == "__main__":
    main()
