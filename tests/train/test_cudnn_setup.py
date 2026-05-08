"""Verify the trainer enables cuDNN benchmark mode and sets TF32 matmul precision
when CUDA is available. These are free wins on static-shape workloads."""
import pytest


def test_apply_cuda_perf_toggles_enables_cudnn_benchmark(monkeypatch):
    """_apply_cuda_perf_toggles must set torch.backends.cudnn.benchmark = True
    and matmul precision = 'high' when CUDA is available."""
    import torch

    # Force the CUDA branch even on CPU-only test runners by monkeypatching the
    # availability check. We don't actually run training; we just call the helper
    # and assert the toggles fire.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    # Reset to default before exercising
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    from src.train import train as train_mod
    train_mod._apply_cuda_perf_toggles()

    assert torch.backends.cudnn.benchmark is True
    assert torch.get_float32_matmul_precision() == "high"


def test_apply_cuda_perf_toggles_no_op_without_cuda(monkeypatch):
    """No-op (and no exception) when CUDA isn't available."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")

    from src.train import train as train_mod
    train_mod._apply_cuda_perf_toggles()

    # Should be unchanged
    assert torch.backends.cudnn.benchmark is False
    assert torch.get_float32_matmul_precision() == "highest"
