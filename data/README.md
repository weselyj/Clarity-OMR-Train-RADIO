# Datasets

Downloaded 2026-04-25. None of the actual data files are tracked (gitignored).

| dir | source | version/commit | count | status |
|---|---|---|---|---|
| primus/ | https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz | 2018 Applied Sciences release | 87,678 incipits | downloaded + extracted 2026-04-25 |
| camera_primus/ | https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz | archive | 87,678 incipits | downloaded + extracted 2026-04-25 |
| grandstaff/ | https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz | archive | 53,883 grand-staff samples | downloaded + extracted 2026-04-25 |
| openscore_lieder/ | https://github.com/OpenScore/Lieder | 6b2dc542ce2e8aa4b78c8ee62103b210efc07015 | 1,462 .mxl files | cloned 2026-04-25 |
| synthetic_full_page/ | regenerated via src/data/generate_synthetic.py | seed=42 | ~20,000 pages (~120K staff crops) | TASK 13 |

## analyze_data.py output (2026-04-25)

Run: `python analyze_data.py` (with path patched to local repo — see note below)

| dir | files | size | extensions (top 5) |
|---|---|---|---|
| camera_primus/ | 526,069 | 4.9 GB | .agnostic: 87678, .mei: 87678, .png: 87678, .semantic: 87678, .jpg: 87678 |
| grandstaff/ | 269,592 | 2.0 GB | .jpg: 107769, .gm: 54050, .bekrn: 53889, .krn: 53883, .tgz: 1 |
| openscore_lieder/ | 8,737 | 956.9 MB | .md: 1894, .mscz: 1462, .mxl: 1462, .txt: 1356, .mscx: 1352 |
| primus/ | 876,781 | 1000.2 MB | .mei: 175356, .mid: 175356, .png: 175356, .pae: 175356, .agnostic: 87678 |
| synthetic_full_page/ | 0 | 0 B | not yet generated — see Task 13 |

Note: `analyze_data.py` has a hardcoded path to the upstream developer's machine (`C:\Users\clq\...`). It was patched locally for this run by substituting the correct repo path. The script should be updated to accept a `--root` argument.

## Notes

- **PrIMuS**: Direct download, no registration required. Compressed archive 261 MB (extracts to ~1 GB). URL: https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz
- **Camera-PrIMuS**: Direct download, no registration required. Compressed archive 2.26 GB (extracts to ~4.9 GB). URL: https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz
- **GrandStaff**: Direct download from MultiScore project (https://sites.google.com/view/multiscore-project/datasets). Compressed archive 908 MB (extracts to ~2 GB). URL: https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz. NOTE: upstream README cited github.com/multiscore/GrandStaff which 404s; the real download URL is at grfia.dlsi.ua.es.
- **OpenScore Lieder**: Git cloned. Pinned commit: 6b2dc542ce2e8aa4b78c8ee62103b210efc07015
- **Synthetic Full-Page**: Generated in Task 13 (~4h CPU) via src/data/generate_synthetic.py --output data/synthetic_full_page/ --num-pages 20000

## Directory naming

This fork uses underscores (camera_primus, openscore_lieder) rather than the upstream hyphens (camera-primus, openscore-lieder) for Python identifier compatibility.

## Re-download from scratch

```bash
git clone https://github.com/OpenScore/Lieder.git data/openscore_lieder
git -C data/openscore_lieder checkout 6b2dc542ce2e8aa4b78c8ee62103b210efc07015
curl -L -o data/primus/primusCalvoRizoAppliedSciences2018.tgz https://grfia.dlsi.ua.es/primus/packages/primusCalvoRizoAppliedSciences2018.tgz
curl -L -o data/camera_primus/CameraPrIMuS.tgz https://grfia.dlsi.ua.es/primus/packages/CameraPrIMuS.tgz
curl -L -o data/grandstaff/grandstaff.tgz https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz
```
