"""Re-run audit but capture per-file kind sets for breakdown analysis."""
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
from src.data.kern_validation import compare_via_music21

GRANDSTAFF_ROOT = Path("data/grandstaff")
krn_paths = sorted(GRANDSTAFF_ROOT.rglob("*.krn"))
print(f"auditing {len(krn_paths):,} files", file=sys.stderr)

per_file_kinds = []  # list of (path, frozenset of kinds)
files_failed = 0
for i, kp in enumerate(krn_paths):
    try:
        r = compare_via_music21(kp)
        kinds = frozenset(d.kind for d in r.divergences)
        per_file_kinds.append((str(kp), kinds))
    except Exception:
        files_failed += 1
        per_file_kinds.append((str(kp), frozenset(["__failed_to_compare__"])))
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(krn_paths)}", file=sys.stderr)

# Breakdown
total = len(per_file_kinds)
passing = sum(1 for _, k in per_file_kinds if not k)
print(f"\ntotal: {total:,}")
print(f"passing: {passing:,} ({passing/total*100:.1f}%)")
print(f"failed_to_compare: {files_failed:,}")

# Group failing files by their kind-set
failing = [(p, k) for p, k in per_file_kinds if k]
combo_counts = Counter(k for _, k in failing)

print(f"\nfailing files broken down by their unique combination of divergence kinds:")
print(f"{'count':>8s}  {'pct':>6s}  kinds")
for combo, count in combo_counts.most_common(20):
    pct = count / total * 100
    label = ", ".join(sorted(combo))
    print(f"{count:>8,}  {pct:>5.2f}%  {label}")

# Single-kind vs multi-kind
single = sum(1 for _, k in failing if len(k) == 1)
multi = sum(1 for _, k in failing if len(k) > 1)
print(f"\nfailing files with single-kind divergence: {single:,}")
print(f"failing files with multi-kind divergence:  {multi:,}")

# Save per-file kinds for further analysis
with open("audit/per_file_kinds_v5.json", "w") as f:
    json.dump([{"path": p, "kinds": sorted(k)} for p, k in per_file_kinds if k],
              f, indent=2)
print(f"\nsaved per_file kinds to audit/per_file_kinds_v5.json")
