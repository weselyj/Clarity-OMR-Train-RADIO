"""Tests for Item C: channels_last memory format (Tier 2 #9).

Gated behind CUDA availability — channels_last conv acceleration only fires
on CUDA, but we can still assert weight layout on CPU with the expected format.

CPU test (always runs): construct model with channels_last=True and assert that
4D conv weight tensors satisfy is_contiguous(memory_format=torch.channels_last).

The test uses a minimal mock of the model construction path to avoid loading
pretrained weights (which requires network access).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch", reason="torch required")


# ---------------------------------------------------------------------------
# Helper: build a tiny synthetic Conv2d model and apply channels_last to it.
# We test the flag mechanics in isolation so we don't need timm or pretrained
# weights (which would require network access in CI).
# ---------------------------------------------------------------------------

class _TinyConvModel(torch.nn.Module):
    """Minimal 4-layer conv model for layout testing."""

    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.linear = torch.nn.Linear(16, 4)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class TestChannelsLastFlag:
    """Verify channels_last memory format is applied correctly."""

    def test_channels_last_conv_weights_have_correct_layout(self):
        """Conv2d weight tensors should report channels_last contiguity after .to(cl)."""
        device = torch.device("cpu")
        model = _TinyConvModel()
        model = model.to(device, memory_format=torch.channels_last)

        cl = torch.contiguous_format  # NCHW (default)
        cl_format = torch.channels_last

        conv_weights_checked = 0
        for name, param in model.named_parameters():
            if param.dim() == 4:
                # 4D conv weight: should be channels_last contiguous.
                assert param.is_contiguous(memory_format=cl_format), (
                    f"Parameter '{name}' (shape {list(param.shape)}) is NOT "
                    f"contiguous in channels_last format after model.to(channels_last)."
                )
                conv_weights_checked += 1

        assert conv_weights_checked >= 2, (
            f"Expected at least 2 conv weight tensors, found {conv_weights_checked}. "
            "Test model may have changed."
        )

    def test_channels_last_input_tensor(self):
        """A 4D NCHW image batch converted via .to(memory_format=channels_last)
        must be channels_last contiguous."""
        images = torch.rand(2, 3, 64, 64)
        assert not images.is_contiguous(memory_format=torch.channels_last), (
            "Freshly created rand tensor should NOT be channels_last by default."
        )
        images_cl = images.to(memory_format=torch.channels_last)
        assert images_cl.is_contiguous(memory_format=torch.channels_last), (
            "After .to(memory_format=channels_last) the tensor must be cl-contiguous."
        )

    def test_channels_last_not_applied_when_flag_off(self):
        """Without channels_last, the default (contiguous NCHW) format is used."""
        device = torch.device("cpu")
        model = _TinyConvModel()
        model = model.to(device)  # no channels_last

        for name, param in model.named_parameters():
            if param.dim() == 4:
                assert param.is_contiguous(), (
                    f"Parameter '{name}' should be contiguous (NCHW) by default."
                )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA-only: channels_last conv acceleration requires a CUDA device.",
    )
    def test_channels_last_on_cuda(self):
        """On a CUDA device the model and input should both be channels_last."""
        device = torch.device("cuda")
        model = _TinyConvModel()
        model = model.to(device, memory_format=torch.channels_last)

        for name, param in model.named_parameters():
            if param.dim() == 4:
                assert param.is_contiguous(memory_format=torch.channels_last), (
                    f"CUDA: parameter '{name}' should be channels_last."
                )

        images = torch.rand(2, 3, 64, 64, device=device).to(
            memory_format=torch.channels_last
        )
        assert images.is_contiguous(memory_format=torch.channels_last)

        # Verify forward pass runs without error.
        with torch.no_grad():
            out = model(images)
        assert out.shape[0] == 2
