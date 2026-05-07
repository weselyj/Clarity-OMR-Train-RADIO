#!/usr/bin/env python3
"""Build the combined Stage 3 token manifest by concatenating four source manifests."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _stream_manifest(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-systems-manifest", type=Path, required=True)
    parser.add_argument("--grandstaff-systems-manifest", type=Path, required=True)
    parser.add_argument("--primus-systems-manifest", type=Path, required=True)
    parser.add_argument("--cameraprimus-systems-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    sources = [
        ("synthetic_systems", args.synthetic_systems_manifest),
        ("grandstaff_systems", args.grandstaff_systems_manifest),
        ("primus_systems", args.primus_systems_manifest),
        ("cameraprimus_systems", args.cameraprimus_systems_manifest),
    ]

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    per_dataset = Counter()
    per_split = Counter()
    total = 0
    with args.output_manifest.open("w", encoding="utf-8") as fout:
        for expected_dataset, src_path in sources:
            for entry in _stream_manifest(src_path):
                ds = entry.get("dataset")
                if ds != expected_dataset:
                    raise ValueError(
                        f"manifest {src_path} has entry with dataset={ds!r}, "
                        f"expected {expected_dataset!r}"
                    )
                fout.write(json.dumps(entry) + "\n")
                per_dataset[ds] += 1
                per_split[entry.get("split", "train")] += 1
                total += 1

    audit = {
        "total_entries": total,
        "per_dataset": dict(per_dataset),
        "per_split": dict(per_split),
        "sources": [str(p) for _, p in sources],
        "output": str(args.output_manifest),
    }
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.write_text(json.dumps(audit, indent=2))
    print(f"[combined] wrote {total} entries; per_dataset={dict(per_dataset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
