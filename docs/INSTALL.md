# Installation

This project requires a CUDA-capable GPU. See [HARDWARE.md](HARDWARE.md).

## 1. Clone

```bash
git clone https://github.com/weselyj/Clarity-OMR-Train-RADIO.git
cd Clarity-OMR-Train-RADIO
```

## 2. Create the cu132 environment

### Windows (PowerShell)

```powershell
.\scripts\setup_venv_cu132.ps1
```

This creates `venv-cu132/`, installs PyTorch nightly cu132 + cuDNN 9.21.01 +
project requirements, and drops a `sitecustomize.py` to resolve the cu132
DLL search path. Idempotent — re-run to refresh.

Activate:

```powershell
.\venv-cu132\Scripts\Activate.ps1
```

### Linux (bash)

```bash
./scripts/setup_venv_cu132.sh
```

This creates `venv-cu132/`, installs PyTorch nightly cu132 + cuDNN 13 +
project requirements. The Windows DLL workaround does not apply on Linux.

Activate:

```bash
source venv-cu132/bin/activate
```

### Manual install (any platform)

```bash
python -m venv venv-cu132
source venv-cu132/bin/activate  # or .\venv-cu132\Scripts\Activate.ps1
pip install --upgrade pip setuptools wheel
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
pip install nvidia-cudnn-cu13
pip install -r requirements.txt
pip install pytest
```

## 3. Verify

```bash
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('CUDA OK:', torch.cuda.get_device_name(0))"
```

Expected: `CUDA OK: <your GPU name>`.

## 4. Run the test suite

```bash
pytest tests/data -q
```

Expected: pure-Python data-layer tests pass. CUDA-required tests under
`tests/inference/`, `tests/pipeline/`, `tests/cli/`, `tests/models/`,
`tests/train/`, and `eval/tests/` skip cleanly on a CPU-only environment
with reason
`"CUDA required (this project requires a CUDA-capable GPU; see docs/HARDWARE.md)"`.

To run the full suite (requires CUDA):

```bash
pytest -q
```

## Rollback (cu128)

If cu132 nightly is unstable, the previous cu128 environment can be restored:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

This produces `venv/` (the rollback path), distinct from `venv-cu132/`
(production).
