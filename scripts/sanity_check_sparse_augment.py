"""Visual sanity check: 3 random sparse-augment images with bbox overlays."""
import random
from pathlib import Path
from PIL import Image, ImageDraw

# Use 94 DPI for the spot-check (faster to load)
IMAGES = Path("data/processed/sparse_augment/images/dpi94")
LABELS = Path("data/processed/sparse_augment/labels")
OUT = Path("data/processed/sparse_augment/sanity_check")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    candidates = []
    for style_dir in IMAGES.iterdir() if IMAGES.exists() else []:
        candidates.extend(sorted(style_dir.glob("*.png")))
    if not candidates:
        print("No images found.")
        return
    sample = random.sample(candidates, min(3, len(candidates)))
    for img_path in sample:
        # The label is named the same as the image but lives under labels/<style>/
        style = img_path.parent.name
        lbl_path = LABELS / style / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            print(f"SKIP {img_path.name}: no label at {lbl_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for line in lbl_path.read_text().strip().splitlines():
            _, xc, yc, w, h = map(float, line.split())
            x1 = (xc - w / 2) * img.width
            y1 = (yc - h / 2) * img.height
            x2 = (xc + w / 2) * img.width
            y2 = (yc + h / 2) * img.height
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        img.save(OUT / img_path.name)
    print(f"Wrote {len(sample)} sanity-check images to {OUT}")


if __name__ == "__main__":
    main()
