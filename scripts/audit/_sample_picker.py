"""Pick a stratified sample of training entries for the audit.

The audit needs ~20 training samples spread across all 4 corpora so the
parity / round-trip experiments aren't biased toward whichever corpus
happens to come first in the manifest. Deterministic given a seed so the
audit is reproducible.
"""
from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Dict, List


def pick_audit_samples(
    manifest_path: Path,
    *,
    n_per_corpus: int = 5,
    seed: int = 42,
) -> List[Dict]:
    """Pick `n_per_corpus` train-split entries from each corpus.

    Returns entries in deterministic order (corpus-name asc, then
    sample_id asc among the picked set) so downstream scripts that
    iterate `samples` in order produce stable output.
    """
    rng = random.Random(seed)
    by_corpus: Dict[str, List[Dict]] = {}
    with manifest_path.open(encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("split") != "train":
                continue
            corpus = entry.get("dataset", "unknown")
            by_corpus.setdefault(corpus, []).append(entry)

    picked: List[Dict] = []
    for corpus in sorted(by_corpus):
        pool = by_corpus[corpus]
        k = min(n_per_corpus, len(pool))
        chosen = rng.sample(pool, k) if k > 0 else []
        chosen.sort(key=lambda e: e["sample_id"])
        picked.extend(chosen)
    return picked


if __name__ == "__main__":  # pragma: no cover
    import sys
    manifest = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    for entry in pick_audit_samples(manifest, n_per_corpus=n, seed=seed):
        print(f"{entry['dataset']:<20} {entry['sample_id']}")
