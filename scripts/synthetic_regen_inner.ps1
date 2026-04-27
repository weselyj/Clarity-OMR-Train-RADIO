# Inner launcher for synthetic_full_page regeneration.
# Generates ~20,000 synthetic full-page renders from openscore_lieder
# sources in 4 verovio engraving styles. CPU-bound, several hours.
#
# Cairo dependency note (from data/README.md):
# cairocffi (used by cairosvg, used by the synthetic generator) needs
# libcairo-2.dll on PATH. The darktable installer bundles a working copy;
# we prepend its bin directory so the generator can find it.
#
# Detached via the matching launch script using WMI Create.

$ErrorActionPreference = "Continue"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$venv = Join-Path $repo "venv-cu132\Scripts\python.exe"
$out  = Join-Path $repo "data\synthetic_full_page"
$log  = Join-Path $repo "logs\synthetic_regen.log"
$err  = Join-Path $repo "logs\synthetic_regen.err"
$pidf = Join-Path $repo "logs\synthetic_regen.pid"

# Prepend darktable's cairo to PATH for this process.
$cairoBin = "C:\Program Files\darktable\bin"
$env:PATH = $cairoBin + ";" + $env:PATH

Set-Location $repo
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
$PID | Out-File -FilePath $pidf -Encoding ascii -NoNewline

# Pre-flight: verify cairo loads in this process before kicking off the
# multi-hour render. Catches a missing libcairo-2.dll early.
& $venv -c "import cairocffi; print(f'cairocffi {cairocffi.__version__} OK; using cairo {cairocffi.cairo_version_string()}')" 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

$pyArgs = @(
    "-m", "src.data.generate_synthetic",
    "--mode", "render",
    "--output-dir", $out,
    "--workers", "8",
    "--seed", "42"
    # No --max-scores or --max-pages-per-score: render the full set.
)

& $venv @pyArgs *>>$log 2>>$err
"=== EXIT_CODE=$LASTEXITCODE ===" | Out-File -FilePath $log -Append -Encoding utf8
