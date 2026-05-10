#!/usr/bin/env bash
# Recreate the venv-cu132 nightly torch + CUDA 13.2 environment (Linux).
#
# Idempotent: if venv-cu132 already exists, the script appends to it.
# Tested on Ubuntu 24.04+ with NVIDIA driver 596+, CUDA Version 13.2.
#
# The Windows-only DLL workaround (scripts/cu132_venv_sitecustomize.py)
# does not apply here.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv="$repo/venv-cu132"
py="$venv/bin/python"

if [ ! -d "$venv" ]; then
    echo "[setup_venv_cu132] creating $venv"
    python3 -m venv "$venv"
fi

echo "[setup_venv_cu132] upgrading pip + setuptools + wheel"
"$py" -m pip install --upgrade pip setuptools wheel

echo "[setup_venv_cu132] installing nightly torch + torchvision (cu132)"
"$py" -m pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132

echo "[setup_venv_cu132] installing nvidia-cudnn-cu13"
"$py" -m pip install nvidia-cudnn-cu13

echo "[setup_venv_cu132] installing project requirements"
"$py" -m pip install -r "$repo/requirements.txt"

echo "[setup_venv_cu132] installing pytest"
"$py" -m pip install pytest

echo "[setup_venv_cu132] verifying CUDA"
"$py" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"

echo "[setup_venv_cu132] done. Activate with: source $venv/bin/activate"
