# Recreate the venv-cu132 nightly torch + CUDA 13.2 environment.
#
# Idempotent: if venv-cu132 already exists, the script appends to it.
# Tested with Python 3.13.13, NVIDIA driver 596.21, CUDA Version 13.2.
#
# Background: PyTorch nightly cu132's torch._load_dll_libraries does not
# detect the nvidia-cu13 / nvidia-cudnn-cu13 wheel layouts under
# nvidia/cu13/bin/x86_64 and nvidia/cudnn/bin, so `import torch` fails
# with WinError 126 on shm.dll. We work around this by dropping a
# sitecustomize.py into the venv that adds those dirs to the DLL search
# path at Python startup. Source of the fix:
# scripts/cu132_venv_sitecustomize.py.

$ErrorActionPreference = "Stop"
$repo = Join-Path $env:USERPROFILE "Clarity-OMR-Train-RADIO"
$venv = Join-Path $repo "venv-cu132"
$py   = Join-Path $venv "Scripts\python.exe"
$pip  = "$py -m pip"

if (-not (Test-Path $venv)) {
    Write-Host "[setup_venv_cu132] creating $venv"
    py -3.13 -m venv $venv
}

Write-Host "[setup_venv_cu132] upgrading pip + setuptools + wheel"
& $py -m pip install --upgrade pip setuptools wheel

Write-Host "[setup_venv_cu132] installing nightly torch + torchvision (cu132)"
& $py -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

Write-Host "[setup_venv_cu132] installing nvidia-cudnn-cu13 (also pulls cu13 cublas + nvrtc)"
& $py -m pip install nvidia-cudnn-cu13

Write-Host "[setup_venv_cu132] installing project requirements"
& $py -m pip install -r (Join-Path $repo "requirements.txt")

Write-Host "[setup_venv_cu132] installing pytest"
& $py -m pip install pytest

Write-Host "[setup_venv_cu132] dropping sitecustomize.py"
$siteCustomizeSrc = Join-Path $repo "scripts\cu132_venv_sitecustomize.py"
$siteCustomizeDst = Join-Path $venv "Lib\site-packages\sitecustomize.py"
Copy-Item $siteCustomizeSrc $siteCustomizeDst -Force

Write-Host "[setup_venv_cu132] verifying"
& $py -c "import torch; print('torch:', torch.__version__); print('cuda:', torch.version.cuda); print('arch:', torch.cuda.get_arch_list()); x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16); print('bf16 matmul:', float((x @ x).sum()))"

Write-Host "[setup_venv_cu132] done."
