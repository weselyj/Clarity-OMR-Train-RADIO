# Real-Scan Degradation Calibration

**Date:** 2026-05-13
**Inputs:** 9 scans from `/tmp/real_scans_calibration`

## Per-scan measurements

| File | rotation (°) | noise σ | blur var | JPEG q |
|---|---:|---:|---:|---:|
| `Receipt_-_Primary_Chord_Progression-p01.png` | 1.00 | 0.0671 | 852.8 | n/a |
| `Receipt_-_Restoration_Hardware_1-p01.png` | -0.00 | 0.0718 | 973.4 | n/a |
| `Scanned_20260514-0334-p01.png` | -0.00 | 0.0730 | 977.4 | n/a |
| `Scanned_20260514-0335-p01.png` | -0.00 | 0.0657 | 839.4 | n/a |
| `Scanned_20260514-0335_1-p01.png` | -0.00 | 0.0728 | 1005.6 | n/a |
| `Scanned_20260514-0335_2-p01.png` | -0.00 | 0.0725 | 967.4 | n/a |
| `Warranty_Deed-p01.png` | 2.00 | 0.0651 | 787.7 | n/a |
| `bethlehem.jpg` | -0.00 | 0.0777 | 1717.6 | 29 |
| `timemachine-p01.png` | -0.00 | 0.0745 | 1030.5 | n/a |

## Aggregate

- Rotation range: [-0.00, 2.00], median 0.00
- Noise σ range: [0.0651, 0.0777], median 0.0725
- Blur var range: [787.7, 1717.6], median 973.4
- JPEG quality range: [29, 29], median 29

## Recommended degradation pipeline parameters

Set `src/data/scan_degradation.py` defaults to span the observed distribution:
- Rotation: ±2.3° uniform
- Noise σ: uniform [0.065, 0.098]
- Blur kernel σ: scaled so output blur_laplacian_var matches [788, 1718]
- JPEG quality: uniform [50, 60]