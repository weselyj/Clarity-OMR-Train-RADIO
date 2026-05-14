#!/usr/bin/env python3
"""Build the combined Stage 3 token manifest by concatenating four (or five) source manifests.

Schema normalisation
--------------------
Different builder scripts emit different field sets.  To guarantee a consistent
schema in the combined manifest, every entry is passed through ``_normalise``
before writing.  Missing fields are filled with ``None`` (or a type-appropriate
sentinel) so that any downstream tool that accesses a field without ``.get()``
doesn't raise ``KeyError``.

Canonical field set (all entries in the combined manifest carry these keys):

  Core identity / routing
    sample_id       – unique entry identifier
    dataset         – corpus name (e.g. "synthetic_systems", "grandstaff_systems")
    split           – "train" | "val" | "test"

  Image reference (at least one is set; the other may be None)
    image_path      – relative path to the pre-cropped system image (raster)
    source_path     – relative path to the source document (.mxl, .krn, …)

  Token payload
    token_sequence  – list of string tokens
    token_count     – len(token_sequence); filled from sequence if absent
    staves_in_system – integer stave count for the system

  synthetic_systems–style provenance (None for corpora that don't have them)
    page_id         – identifier of the source page
    system_index    – index of this system on its page (0-based)
    staff_indices   – list of per-page staff indices in this system
    style_id        – engraving style (e.g. "bravura-compact")
    page_number     – 1-based page number within the score
    source_format   – originating score format ("musicxml", "mscz", …)
    score_type      – instrumentation type ("piano", "orchestral", …)

  grandstaff_systems–style provenance (None for corpora that don't have them)
    group_id        – piece-level group identifier (prevents val/train leakage)
    modality        – data modality ("image+notation", …)
    variant         – image variant ("clean" | "distorted")
    krn_path        – relative path to the .krn source file
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

# ---------------------------------------------------------------------------
# Canonical field defaults.  Any field absent in a source entry is inserted
# with the corresponding default value.  ``None`` is used for optional string /
# object fields; a type-correct sentinel is used otherwise so that arithmetic
# on the value doesn't crash (e.g. ``0`` for integers, ``[]`` for lists).
# ---------------------------------------------------------------------------
_CANONICAL_DEFAULTS: dict = {
    # Core
    "sample_id": None,
    "dataset": None,
    "split": "train",
    # Image references
    "image_path": None,
    "source_path": None,
    # Token payload
    "token_sequence": [],
    "token_count": None,    # filled dynamically from token_sequence if absent
    "staves_in_system": None,
    # synthetic_systems provenance
    "page_id": None,
    "system_index": None,
    "staff_indices": None,
    "style_id": None,
    "page_number": None,
    "source_format": None,
    "score_type": None,
    # grandstaff_systems provenance
    "group_id": None,
    "modality": None,
    "variant": None,
    "krn_path": None,
}


def _normalise(entry: dict) -> dict:
    """Return a copy of *entry* with all canonical fields present.

    Missing fields are filled from ``_CANONICAL_DEFAULTS``.  ``token_count`` is
    derived from ``token_sequence`` when absent (or when it is ``None``).
    """
    out = dict(entry)
    for field, default in _CANONICAL_DEFAULTS.items():
        if field not in out:
            out[field] = default
    # token_count: always derive from sequence when missing or None
    if out.get("token_count") is None:
        seq = out.get("token_sequence") or []
        out["token_count"] = len(seq)
    return out


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
    parser.add_argument("--scanned-grandstaff-systems-manifest", type=Path,
                        required=False, default=None,
                        help="Optional 5th source: scanned_grandstaff_systems manifest. "
                             "When omitted the corpus is silently skipped.")
    parser.add_argument("--primus-systems-manifest", type=Path, required=True)
    parser.add_argument("--cameraprimus-systems-manifest", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    args = parser.parse_args()

    sources = [
        ("synthetic_systems", args.synthetic_systems_manifest),
        ("grandstaff_systems", args.grandstaff_systems_manifest),
    ]
    if args.scanned_grandstaff_systems_manifest is not None:
        sources.append(("scanned_grandstaff_systems", args.scanned_grandstaff_systems_manifest))
    sources += [
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
                normalised = _normalise(entry)
                fout.write(json.dumps(normalised) + "\n")
                per_dataset[ds] += 1
                per_split[normalised.get("split", "train")] += 1
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
