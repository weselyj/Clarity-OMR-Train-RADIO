# Bottom-Quartile Lieder Failure-Mode Cluster (Stage 3 v3 best.pt)

**Date:** 2026-05-13
**Input:** `eval/results/lieder_stage3_v3_best_scores.csv` — 33 pieces with onset_f1 < 0.1

## Cluster counts

| Cluster | Count | % of bottom-quartile |
|---|---:|---:|
| phantom-staff-residual | 25 | 75.8% |
| other | 7 | 21.2% |
| bass-clef-misread | 4 | 12.1% |
| key-time-sig-residual | 3 | 9.1% |

## Examples per cluster

### phantom-staff-residual
- `lc4978396`
- `lc4999332`
- `lc4985985`
- `lc5051353`
- `lc5093434`

### other
- `lc4945954`
- `lc5001880`
- `lc5598547`
- `lc6211903`
- `lc6260379`

### bass-clef-misread
- `lc4985985`
- `lc5077482`
- `lc6686901`
- `lc6812631`

### key-time-sig-residual
- `lc5032950`
- `lc5705502`
- `lc6302395`

## Gate check

`bass-clef-misread` = 4/33 = 12.1% of bottom-quartile.

**GATE FAIL:** <30% -> pause and re-scope before Phase 1.