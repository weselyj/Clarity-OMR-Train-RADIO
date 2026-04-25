# Environment Notes

## Python Version

Built against Python 3.13 instead of plan's 3.12 - 3.12 installer unavailable on this Windows host. Some packages may require workarounds; document each as encountered.

## Clone Method

Cloned via HTTPS (not SSH git@github.com) - SSH key for GitHub not configured on this host. HTTPS with gh credential helper works correctly.

## PyTorch

Installed: torch 2.11.0+cu128 / torchvision 0.26.0+cu128 (Python 3.13 wheels available from pytorch.org/whl/cu128). CUDA verified: RTX 5090 detected, torch.cuda.is_available() returns True.
