"""Apply sparse_augment transformations across a sample of the lieder corpus.

Produces three variants per source MXL:
- {stem}_strip.musicxml  -- strip_silent_intro
- {stem}_mbrest.musicxml -- add_multibar_rest on the "Voice" part
- {stem}_thin.musicxml   -- thin_vocal_part on the "Voice" part

Skips silently if a part is missing or any other transformation error occurs;
prints a SKIP line so causes can be audited.

Output: data/processed/sparse_augment/mxl/*.musicxml
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.data.sparse_augment import (
    strip_silent_intro,
    add_multibar_rest,
    thin_vocal_part,
)

SOURCE = Path("data/openscore_lieder/scores")
OUT_MXL = Path("data/processed/sparse_augment/mxl")
SAMPLE_SIZE = 200  # cap input scores; transformations are deterministic so
                   # we don't need a random seed


def main() -> None:
    OUT_MXL.mkdir(parents=True, exist_ok=True)

    src_files = sorted(SOURCE.rglob("*.mxl"))[:SAMPLE_SIZE]
    print(f"Processing {len(src_files)} source MXLs")

    written = 0
    skipped = 0
    for src in src_files:
        # Each variant produces ONE output MXL. Errors per variant are independent.
        variants = [
            (lambda s, o: strip_silent_intro(s, o, intro_measures=4), "_strip"),
            (lambda s, o: add_multibar_rest(o, s, "Voice", n_measures=8), "_mbrest"),
            (lambda s, o: thin_vocal_part(s, o, "Voice", keep_n_notes=1), "_thin"),
        ]
        for fn, suffix in variants:
            out = OUT_MXL / f"{src.stem}{suffix}.musicxml"
            if out.exists():
                continue
            try:
                fn(src, out)
                written += 1
            except Exception as e:
                skipped += 1
                print(f"SKIP {src.stem}{suffix}: {type(e).__name__}: {e}")

    print(f"Wrote {written} augmented MXLs; skipped {skipped}")


if __name__ == "__main__":
    main()
