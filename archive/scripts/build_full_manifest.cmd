@echo off
setlocal
cd /d "%USERPROFILE%\Clarity-OMR-Train-RADIO"
call venv\Scripts\activate.bat
echo === [%date% %time%] STAGE 1: src.data.index (full corpus) ===
python -u -m src.data.index ^
  --data-root data ^
  --split-config configs/splits.yaml ^
  --output-manifest src/data/manifests/canonical_manifest_full.jsonl ^
  --output-summary src/data/manifests/canonical_summary_full.json
if errorlevel 1 (
  echo === [%date% %time%] STAGE 1 FAILED, aborting ===
  exit /b 1
)
echo === [%date% %time%] STAGE 1 OK ===
echo === [%date% %time%] STAGE 2: src.data.convert_tokens (primus + cameraprimus + grandstaff) ===
python -u -m src.data.convert_tokens ^
  --input-manifest src/data/manifests/canonical_manifest_full.jsonl ^
  --output-manifest src/data/manifests/token_manifest_full.jsonl ^
  --output-summary src/data/manifests/token_summary_full.json ^
  --datasets primus,cameraprimus,grandstaff
set rc=%errorlevel%
echo === [%date% %time%] DONE exit=%rc% ===
exit /b %rc%
