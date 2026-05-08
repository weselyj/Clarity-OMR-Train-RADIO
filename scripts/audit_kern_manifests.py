#!/usr/bin/env python3
"""Audit existing manifests for kern-conversion damage from the pre-fix converter.

Read-only. Produces a JSON report listing per-corpus damage statistics. Does NOT re-derive any manifest.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.convert_tokens import convert_kern_file


_MINOR_THRESHOLD = 0.05  # <=5% edit distance / len(new) is "minor"


def _edit_distance(a: List[str], b: List[str]) -> int:
    # Token-level Levenshtein (small enough for tokens; OK at O(n*m)).
    if a == b:
        return 0
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def classify_diff(new_tokens: List[str], old_tokens: List[str]) -> str:
    if new_tokens == old_tokens:
        return "unchanged"
    dist = _edit_distance(new_tokens, old_tokens)
    denom = max(1, len(new_tokens))
    if dist / denom <= _MINOR_THRESHOLD:
        return "changed_minor"
    return "changed_major"


def audit_manifest(manifest_path: Path, *, project_root: Path, sample_seed: int = 42) -> Dict[str, object]:
    """Audit a single manifest. Returns a per-corpus report dict."""
    rng = random.Random(sample_seed)
    total_entries = 0
    entries_with_krn = 0
    spine_hist: Counter = Counter()
    tag_counts: Counter = Counter()
    sample_diffs: List[Dict[str, object]] = []

    sample_pool: List[Dict[str, object]] = []

    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            total_entries += 1
            krn_rel = entry.get("krn_path")
            if not krn_rel or not str(krn_rel).endswith(".krn"):
                continue
            entries_with_krn += 1
            krn_path = (project_root / krn_rel).resolve()
            if not krn_path.exists():
                continue
            new_tokens = convert_kern_file(krn_path)
            spine_count = max(1, new_tokens.count("<staff_start>")) if new_tokens else 0
            spine_hist[str(spine_count) if spine_count < 3 else "3+"] += 1
            old_tokens = list(entry.get("token_sequence") or [])
            tag = classify_diff(new_tokens, old_tokens)
            tag_counts[tag] += 1
            if tag != "unchanged":
                sample_pool.append(
                    {
                        "sample_id": entry.get("sample_id"),
                        "krn_path": krn_rel,
                        "tag": tag,
                        "new_len": len(new_tokens),
                        "old_len": len(old_tokens),
                    }
                )

    rng.shuffle(sample_pool)
    sample_diffs = sample_pool[:10]

    return {
        "manifest_path": str(manifest_path),
        "total_entries": total_entries,
        "entries_with_krn_source": entries_with_krn,
        "spine_count_histogram": dict(spine_hist),
        "tag_counts": {
            "unchanged": tag_counts.get("unchanged", 0),
            "changed_minor": tag_counts.get("changed_minor", 0),
            "changed_major": tag_counts.get("changed_major", 0),
        },
        "sample_diffs": sample_diffs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("audit/kern_conversion_damage.json"))
    args = parser.parse_args()

    manifests = list(args.processed_root.rglob("*.jsonl"))
    report: Dict[str, object] = {
        "audit_run_at": datetime.now(timezone.utc).isoformat(),
        "corpora": {},
    }
    for manifest in manifests:
        try:
            corpus = manifest.parent.parent.name  # data/processed/<corpus>/manifests/<file>
        except Exception:
            corpus = manifest.stem
        per_manifest = audit_manifest(manifest, project_root=args.project_root)
        report["corpora"].setdefault(corpus, []).append(per_manifest)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[audit] wrote report to {args.output}")
    print(f"[audit] manifests audited: {len(manifests)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
