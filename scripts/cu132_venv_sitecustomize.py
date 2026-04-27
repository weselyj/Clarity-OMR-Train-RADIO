"""Auto-add NVIDIA cu13 + cudnn bin dirs to the DLL search path.

PyTorch nightly cu132's torch._load_dll_libraries does not detect the
nvidia-cu13 / nvidia-cudnn-cu13 wheel layouts (nvidia/cu13/bin/x86_64
and nvidia/cudnn/bin). Without help, `import torch` fails with WinError
126 on shm.dll; and even after that's worked around, runtime calls into
cudnn (`cudnnGetVersion`) fail with "Invalid handle. Cannot load symbol
cudnnGetVersion" during the first conv pass.

Two workarounds applied at Python startup:
  1. os.add_dll_directory(...)  — fixes Python-managed loads (covers
     torch's lazy DLL loader and the import-time shm.dll path).
  2. Prepend to os.environ["PATH"]  — fixes runtime LoadLibrary calls
     in libtorch C++ that don't go through the Python search-path
     manager (covers cudnnGetVersion at the first conv).

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
    _path_prefix = []
    for d in _candidates:
        if d.is_dir():
            _ds = str(d)
            try:
                os.add_dll_directory(_ds)
            except (OSError, FileNotFoundError):
                pass
            _path_prefix.append(_ds)
    if _path_prefix:
        os.environ["PATH"] = os.pathsep.join(_path_prefix) + os.pathsep + os.environ.get("PATH", "")
