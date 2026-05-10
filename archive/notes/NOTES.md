# Environment Notes

## Python Version

Built against Python 3.13 instead of plan's 3.12 - 3.12 installer unavailable on this Windows host. Some packages may require workarounds; document each as encountered.

## Clone Method

Cloned via HTTPS (not SSH git@github.com) - SSH key for GitHub not configured on this host. HTTPS with gh credential helper works correctly.

## PyTorch

PyTorch (venv/): torch 2.11.0+cu128 / torchvision 0.26.0+cu128 — rollback only
PyTorch (venv-cu132/): torch 2.13.0.dev20260426+cu132, cuDNN 9.21.01 — production default

CUDA verified: RTX 5090 detected, torch.cuda.is_available() returns True.
