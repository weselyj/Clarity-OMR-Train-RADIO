"""Build the 20-sample overfit-smoke manifest for Stage 3 v4 Phase 0b.

Selection rules:
  - 2-staff grand-staff systems only.
  - token_count <= max_token_count (default 256) — keeps loss tight.
  - From `grandstaff_systems` train split.
  - Deterministic: sort by source_path lexicographically, take first N.

The point of this manifest is to function as a pre-flight check that the trainer
can overfit a small clean set. It is NOT used in main training.

Usage:
  python scripts/data/build_overfit_smoke_manifest.py \\
      --src-manifest data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl \\
      --output data/manifests/overfit_smoke_v4.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = REPO_ROOT / "data/processed/grandstaff_systems/manifests/synthetic_token_manifest.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "data/manifests/overfit_smoke_v4.jsonl"


def _staff_count(entry: dict) -> int | None:
    """Read staff count from either the plan-style or grandstaff-native field."""
    val = entry.get("staff_count")
    if val is None:
        val = entry.get("staves_in_system")
    return val


def _tokens(entry: dict) -> list:
    """Read the token list from either the plan-style or grandstaff-native field."""
    toks = entry.get("tokens")
    if toks is None:
        toks = entry.get("token_sequence", [])
    return toks


def _sort_key(entry: dict) -> str:
    """Deterministic lex sort key. Prefer source_path, fall back to sample_id then image_path."""
    for k in ("source_path", "sample_id", "image_path"):
        v = entry.get(k)
        if isinstance(v, str):
            return v
    # Last resort: JSON of the entry so the sort is at least defined.
    return json.dumps(entry, sort_keys=True)


def build_manifest(
    src_manifest: Path,
    output: Path,
    n: int = 20,
    max_token_count: int = 256,
) -> None:
    candidates = []
    with src_manifest.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("split") != "train":
                continue
            if _staff_count(entry) != 2:
                continue
            tok_count = entry.get("token_count")
            if tok_count is None:
                tok_count = len(_tokens(entry))
            if tok_count > max_token_count:
                continue
            # Normalize: ensure token_count is set on the emitted entry so
            # downstream consumers (and tests) can rely on it.
            entry["token_count"] = tok_count
            # Normalize staff_count too (tests expect this field name).
            if "staff_count" not in entry:
                entry["staff_count"] = _staff_count(entry)
            candidates.append(entry)

    candidates.sort(key=_sort_key)
    selected = candidates[:n]

    if len(selected) < n:
        raise RuntimeError(
            f"Only {len(selected)} candidates after filtering (wanted {n}). "
            f"Check src_manifest and filter criteria."
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for entry in selected:
            f.write(json.dumps(entry) + "\n")
    print(f"Wrote {len(selected)} entries to {output}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-manifest", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--max-token-count", type=int, default=256)
    args = ap.parse_args()
    build_manifest(args.src_manifest, args.output, args.n, args.max_token_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
