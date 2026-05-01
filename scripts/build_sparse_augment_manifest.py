"""Build a manifest JSONL for the augmented MXLs in sparse_augment/mxl/.

Output: src/data/manifests/sparse_augment_manifest.jsonl
Each line: {"musicxml_path": "data/processed/sparse_augment/mxl/<file>.musicxml"}
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MXL_DIR = REPO / "data" / "processed" / "sparse_augment" / "mxl"
OUT = REPO / "src" / "data" / "manifests" / "sparse_augment_manifest.jsonl"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(MXL_DIR.glob("*.musicxml"))
    with OUT.open("w") as f:
        for mxl in files:
            rel = mxl.relative_to(REPO).as_posix()
            f.write(json.dumps({"musicxml_path": rel}) + "\n")
    print(f"Wrote {len(files)} entries to {OUT}")


if __name__ == "__main__":
    main()
