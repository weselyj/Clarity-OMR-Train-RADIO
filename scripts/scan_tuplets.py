"""Scan all .krn files for unusual tuplet ratios appearing in music21's parse."""
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, ".")
import music21

GRANDSTAFF_ROOT = Path("data/grandstaff")
krn_paths = sorted(GRANDSTAFF_ROOT.rglob("*.krn"))
print(f"scanning {len(krn_paths):,} files for tuplet ratios", file=sys.stderr)

ratio_counts = Counter()
files_with_unusual = set()

for i, kp in enumerate(krn_paths):
    try:
        s = music21.converter.parse(str(kp), format="humdrum")
        for elem in s.recurse().notesAndRests:
            tuplets = list(getattr(elem.duration, "tuplets", []) or [])
            if not tuplets:
                continue
            t = tuplets[0]
            actual = int(getattr(t, "numberNotesActual", 0) or 0)
            normal = int(getattr(t, "numberNotesNormal", 0) or 0)
            if actual == 0 or normal == 0:
                continue
            ratio_counts[(actual, normal)] += 1
            # Mark "unusual" = anything other than 3:2, 5:4, 6:4, 7:4
            if (actual, normal) not in {(3, 2), (5, 4), (6, 4), (7, 4)}:
                files_with_unusual.add(str(kp))
    except Exception:
        pass
    if (i + 1) % 5000 == 0:
        print(f"  {i+1}/{len(krn_paths)}", file=sys.stderr)

print()
print(f"--- Tuplet ratio distribution across corpus ---")
for (a, n), count in sorted(ratio_counts.items(), key=lambda x: -x[1]):
    label = f"{a}:{n}"
    print(f"  {label:>8s}  {count:>10,} occurrences")

print()
print(f"files with at least one 'unusual' (non-3:2/5:4/6:4/7:4) tuplet: {len(files_with_unusual):,}")
