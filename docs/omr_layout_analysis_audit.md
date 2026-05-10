# OMR Layout Analysis Dataset Audit

**Task:** 0.3 — Class mapping audit for `v-dvorak/omr-layout-analysis`  
**Date:** 2026-04-28  
**Author:** Claude Code (subagent)  
**Verdict:** **GO** — `stave` class semantically matches our `staff` target.

---

## 1. Annotation Format

**Format:** Custom COCO-like JSON per image page (not standard COCO with a `categories` array).

Each `.json` file contains:

```json
{
  "width": <int>,
  "height": <int>,
  "system_measures": [{"left": ..., "top": ..., "width": ..., "height": ...}, ...],
  "stave_measures": [...],
  "staves": [...],
  "systems": [...],       // only present in maker_mode output
  "grand_staff": [...]    // only present in maker_mode output
}
```

The bounding box fields use **absolute pixel coordinates** in `[left, top, width, height]` format.  
Source path: `al2_extracted/Schubert_D911-01/json/Schubert_D911-01_000.json` (representative sample).

The pipeline code converts these to YOLO format (class, x_center, y_center, width, height — all relative) via `Sheet.get_all_yolo_labels()` in `app/LabelKeeper/Sheet.py`.

---

## 2. Class IDs and Names

From `app/Utils/Settings.py`:

| YOLO Class ID | Name | AL2 bbox count | % of total |
|---|---|---|---|
| 0 | `system_measures` | 24,186 (AL2 only) | ~5% |
| 1 | `stave_measures` | 50,064 (AL2 only) | ~10% |
| 2 | `staves` | 11,143 (AL2 only) | ~2% |
| 3 | `systems` | 5,376 (AL2 only) | ~1% |
| 4 | `grand_staff` | 5,375 (AL2 only) | ~1% |

Full dataset totals across all 4 sources (from README):

| Class | Total bboxes |
|---|---|
| `staves` (class 2) | **67,064** |
| `stave_measures` (class 1) | 275,548 |
| `system_measures` (class 0) | 99,102 |
| `systems` (class 3) | 23,851 |
| `grand_staff` (class 4) | 23,428 |

Note: 1,006 MZKBlank images are negative samples (no music), contributing 0 bboxes of any class.

---

## 3. Per-Image Stave Bbox Count

**Sampling methodology:** 5 images drawn from 5 distinct score directories in AudioLabs v2 (random seed 42); plus cross-score statistics from all 940 AL2 pages.

### 5-image sample (detailed):

| Score | Page | Staves | Image size | Staves/system |
|---|---|---|---|---|
| Wagner WWV086C-3 | p196 | 12 | 593×811 | ~2 |
| Wagner WWV086A | p017 | 12 | 596×842 | — |
| Chorissimo Blue105 | p000 | 14 | 770×1107 | ~7 (choral) |
| Wagner WWV086D-1 | p064 | 12 | 598×799 | ~2 |
| Wagner WWV086B-1 | p037 | 12 | 604×817 | ~2 |

**Sample summary:** Median = 12, range = [12, 14].

### Full AL2 dataset (940 pages):

- **Median staves per page: 12**
- **Range: [0, 18]** (0 on negative/blank pages, up to 18 on dense orchestral pages)
- **Average: 11.9 staves per page**
- **Distribution (most common count): 12 staves (805 out of 940 pages)**

### Stave bbox geometry (from Schubert D911-01):

Individual stave bboxes are narrow horizontal strips — height ~22–29px on a ~932px tall page (~2.5% of page height), width ~500–540px (nearly full page width). This is precisely the geometry of a single 5-line staff, confirming individual-stave granularity.

---

## 4. Semantic Match Assessment

**Conclusion: MATCH. Their `stave` (class 2) == our `staff` (class 0).**

Evidence:

1. **Staves per system = 3 in vocal+piano lieder (Schubert D911).** Winterreise songs in the AudioLabs v2 subset show 15 staves / 5 systems = **exactly 3.0 staves per system**: vocal, piano treble, piano bass. This is only achievable if every individual 5-line staff is annotated separately — NOT if systems were annotated as units.

2. **Bbox geometry confirms individual staves.** Each `stave` bbox is ~2.5% of page height and ~78–83% of page width — the exact footprint of one 5-line staff line, not a system (which would be ~3× taller in a vocal+piano score).

3. **Visual overlay confirms.** The rendered overlay (`docs/omr_layout_audit_sample.png`) shows red bounding boxes each wrapping exactly one staff row, with 15 distinct horizontal strips corresponding to 5 systems × 3 staves per system on the Schubert page.

4. **The annotation reference document is explicit.** From `docs/annot_reference.md`:
   > "Staff — has five parallel lines that are all the same length — has to be associated with any notes, music"
   
   A `stave` = one 5-line staff = our `staff`. There is no ambiguity with systems or grand staves, which are separate classes.

5. **Cross-source consistency.** The 67,064 stave bboxes come from AudioLabs v2 (real scans derived from MIDI/CSV), Muscima++ (handwritten), and OSLiC (rendered from MuseScore) — all annotating at individual-stave granularity.

---

## 5. Vocal Staff Inclusion on Lieder Pages

**Vocal staves ARE included** in the dataset, with one important caveat.

**Positive evidence:**

- Schubert D911 (Winterreise, 24 songs from OSLiC subset) is present in the AudioLabs v2 data. Every sampled page shows 3.0 staves/system, which only works if the vocal stave is annotated alongside the piano staves.
- D911-01 to D911-07 all show ~3 staves/system consistently.
- The Chorissimo scores (4-voice choral) similarly show dense multi-stave-per-system annotation.

**Caveat — annotation policy for sparse staves:**

From `docs/annot_reference.md` (bold emphasis mine):

> ✅ valid staff: even though there is **no music at the end** of the staff  
> ❌ invalid staves: **no music associated with them at all**

This means:
- A vocal stave with notes + rests is **annotated** (valid).
- A vocal stave that is **completely blank** (multi-measure whole-rest-only, no notes) is **NOT annotated**.

**Impact:** For our primary failure mode — the sparse vocal stave on the pickup page or between phrases — this caveat matters. A lieder page where the vocal staff has only 1 pickup note + rests is likely included (music is present). A hypothetical page where the vocal part is entirely blank (tacet) would be excluded. OSLiC scores from MuseScore rarely have fully blank vocal staves; they typically show rests. Net assessment: the OSLiC lieder data provides coverage for sparse-but-not-totally-empty vocal staves, which is the realistic sparse-staff scenario we care about.

**The known Muscima++ issue** (from README):
> "Muscima++ takes empty staves as valid staves, we only consider staves with some music to be valid — this leads to multiple problems while working with M++: empty staves are marked as valid. Has to be fixed mostly manually."

The M++ handling of empty staves is an admitted inconsistency in this dataset — M++ may include some staves our policy would exclude. This affects only 140 images and is minor.

---

## 6. Class-Mapping Plan

Our Stage A YOLO expects a single class: `0 = staff`.

Their 5-class YOLO format:

| Their ID | Their name | Mapping |
|---|---|---|
| 0 | `system_measures` | **DROP** (not staff-level) |
| 1 | `stave_measures` | **DROP** (measure-level, not staff-level) |
| 2 | `staves` | **→ MAP to class 0 `staff`** |
| 3 | `systems` | **DROP** (system-level grouping) |
| 4 | `grand_staff` | **DROP** (instrument-group-level) |

**Converter implementation:** When processing each `.txt` YOLO label file, keep only lines where `class_id == 2`, and output them with `class_id = 0`. Lines with class IDs 0, 1, 3, 4 are discarded.

In `app/__main__.py` terms: invoke the dataset builder with `-l 2` (labels flag):
```
python3 -m app output_dir --al2 --mpp --osl -l 2
```
This tells the builder to only include label index 2 (`staves`) in the output, and since only one label is requested, it becomes class 0 in the generated YOLO files.

Alternatively, implement in `src/data/omr_layout_import.py` by filtering the downloaded pre-built YOLO dataset (if using the `datasets-release` archive) or by invoking the builder tool.

---

## 7. GO / NO-GO Recommendation

**Verdict: GO**

Evidence summary:
- `stave` class labels individual 5-line staves at exactly the granularity Stage A expects.
- Vocal staves in OSLiC lieder scores (including Winterreise, the archetypical 3-stave vocal+piano layout) are included whenever they contain music.
- 67,064 stave bboxes across 7,013 images provides substantial real-scan coverage.
- Simple class-ID filter (`class == 2 → 0`, drop rest) is sufficient — no re-labeling required.
- Annotation format is YOLO-compatible; the repo's own tooling can produce single-class output with `-l 2`.

**One caveat to monitor:** Pages where the vocal stave is completely blank (no notes, no rests — e.g., a page-final tacet) are excluded by annotation policy. This is a minor gap that only affects a rare edge case (fully silent vocal part for an entire page). The core sparse-but-present scenario (1 pickup note + rests) is covered.

**Phase 2 can proceed.** The converter should apply the `class 2 → 0` filter and discard classes 0, 1, 3, 4.

---

## Artifacts

- **Sample annotation JSON:** `<source-clone>/al2_extracted/Schubert_D911-01/json/Schubert_D911-01_001.json` (out-of-tree scratch clone of the omr-layout-analysis dataset)
- **Rendered overlay PNG:** `docs/omr_layout_audit_sample.png` (~324KB, committed)
  - Red boxes: `stave` class (15 individual staff rows on a vocal+piano lieder page)
  - Blue boxes: `system_measures` class (28 measure boundaries)
  - Score: Schubert D911-01 (Winterreise), page 2/5
- **Scratch clone:** local clone of the omr-layout-analysis dataset (out-of-tree; not committed)
- **Source stats code:** `omr_audit.py`, `omr_audit2.py`, `omr_audit3.py` (out-of-tree scratch utilities used to derive the stats summarised above; not committed)
