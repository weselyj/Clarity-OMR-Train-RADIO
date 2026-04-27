"""Auto-add NVIDIA cu13 + cudnn bin dirs to the DLL search path.

PyTorch nightly cu132's torch._load_dll_libraries does not detect the
nvidia-cu13 / nvidia-cudnn-cu13 wheel layouts (nvidia/cu13/bin/x86_64
and nvidia/cudnn/bin), so import torch fails with WinError 126 on
shm.dll. This file runs at Python startup (via the sitecustomize hook)
and adds the relevant directories before any user code imports torch.

Lives in the cu132 venv only. Safe no-op if the nvidia packages aren't
installed.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform != "win32":
    pass  # sitecustomize is venv-scoped; non-Windows is moot here
else:
    _site_packages = Path(__file__).resolve().parent
    _candidates = [
        _site_packages / "nvidia" / "cudnn" / "bin",
        _site_packages / "nvidia" / "cu13" / "bin" / "x86_64",
    ]
    for d in _candidates:
        if d.is_dir():
            try:
                os.add_dll_directory(str(d))
            except (OSError, FileNotFoundError):
                pass
