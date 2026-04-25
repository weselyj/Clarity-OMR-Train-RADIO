"""Defines the OpenScore Lieder held-out 10% eval split.

Deterministic: seed is fixed. Same split used by:
- eval/run_baseline_reproduction.py (DaViT checkpoint)
- eval/run_lieder_eval.py (RADIO checkpoint)
- All training stages (must NOT include eval pieces in train data)

Layout verified 2026-04-25:
  data/openscore_lieder/scores/<Composer>/<Opus>/<Song>/<id>.mxl
All 1,462 files are .mxl; no .musicxml files exist in the corpus.

Hash stability note: split_hash() uses p.name (filename only), making it
cross-platform stable regardless of Windows vs Linux path separators.
"""
import hashlib
import random
from pathlib import Path
from typing import List

# Scores are nested under data/openscore_lieder/scores/<Composer>/...
# rglob picks them up regardless of nesting depth.
LIEDER_ROOT = Path("data/openscore_lieder")
EVAL_FRACTION = 0.10
SEED = 20260425  # YYYYMMDD


def list_all_pieces() -> List[Path]:
    """Return sorted list of all .mxl/.musicxml files in the Lieder corpus."""
    files = sorted(LIEDER_ROOT.rglob("*.mxl")) + sorted(
        LIEDER_ROOT.rglob("*.musicxml")
    )
    if not files:
        raise RuntimeError(
            f"no Lieder pieces found at {LIEDER_ROOT} — did Task 2 finish?"
        )
    return files


def get_eval_pieces() -> List[Path]:
    """Return the deterministic 10% eval split."""
    all_pieces = list_all_pieces()
    rng = random.Random(SEED)
    n_eval = int(len(all_pieces) * EVAL_FRACTION)
    return sorted(rng.sample(all_pieces, n_eval))


def get_train_pieces() -> List[Path]:
    """Return the train split (everything not in eval)."""
    eval_set = set(get_eval_pieces())
    return [p for p in list_all_pieces() if p not in eval_set]


def split_hash() -> str:
    """A fingerprint of the eval split — log this with every eval run.

    Uses p.name (filename stem + extension only) so the hash is identical
    on Windows and Linux regardless of path-separator differences.
    """
    h = hashlib.sha256()
    for p in get_eval_pieces():
        h.update(p.name.encode())
    return h.hexdigest()[:16]


if __name__ == "__main__":
    eval_p = get_eval_pieces()
    train_p = get_train_pieces()
    print(f"Total: {len(eval_p) + len(train_p)}")
    print(f"Eval split:  {len(eval_p)} pieces")
    print(f"Train split: {len(train_p)} pieces")
    print(f"Split hash:  {split_hash()}")
    print("\nFirst 5 eval pieces:")
    for p in eval_p[:5]:
        print(f"  {p}")
