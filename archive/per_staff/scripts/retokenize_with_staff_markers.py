#!/usr/bin/env python3
"""Generic single-staff retokenizer.

Filters a manifest to a single source dataset, prepends `<staff_idx_0>` to the
content of each entry, rewrites `dataset` and `staves_in_system` fields. Used
to produce primus_systems and cameraprimus_systems from the existing combined
or per-dataset manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _retokenize_entry(entry: dict, target_dataset: str) -> dict:
    seq = entry["token_sequence"]
    if not (len(seq) >= 4 and seq[0] == "<bos>" and seq[1] == "<staff_start>"
            and seq[-2] == "<staff_end>" and seq[-1] == "<eos>"):
        raise ValueError(
            f"malformed token_sequence for entry {entry.get('sample_id', '?')}: "
            f"first 2 = {seq[:2]}, last 2 = {seq[-2:]}"
        )
    new_seq = ["<bos>", "<staff_start>", "<staff_idx_0>"] + seq[2:]  # insert marker after staff_start
    new_entry = dict(entry)
    new_entry["dataset"] = target_dataset
    new_entry["staves_in_system"] = 1
    new_entry["token_sequence"] = new_seq
    new_entry["token_count"] = len(new_seq)
    return new_entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--source-dataset", required=True,
                        help="Filter input to entries with this `dataset` field (e.g., 'primus').")
    parser.add_argument("--target-dataset", required=True,
                        help="Set the output `dataset` field to this (e.g., 'primus_systems').")
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_out = 0
    with args.input_manifest.open("r", encoding="utf-8") as fin, \
         args.output_manifest.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            n_in += 1
            if entry.get("dataset") != args.source_dataset:
                continue
            new_entry = _retokenize_entry(entry, args.target_dataset)
            fout.write(json.dumps(new_entry) + "\n")
            n_out += 1

    print(f"[retokenize] read {n_in} entries; wrote {n_out} for source={args.source_dataset} → target={args.target_dataset}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
