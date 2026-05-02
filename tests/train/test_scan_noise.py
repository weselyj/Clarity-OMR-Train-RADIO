"""Unit tests for the scan-noise augmentation warmup ramp.

The pipeline lives in src/train/scan_noise.py and is imported (via patching)
by scripts/train_yolo.py. We test the pure helpers (no albumentations import
needed for these tests).
"""
from __future__ import annotations

import pytest

from src.train.scan_noise import (
    BASE_NOISE_PROBABILITIES,
    intensity_for_step,
    scaled_probabilities,
)


class TestIntensityForStep:
    def test_no_warmup_full_intensity(self):
        """Without warmup, intensity is always 1.0."""
        assert intensity_for_step(0, warmup_steps=0) == 1.0
        assert intensity_for_step(1000, warmup_steps=0) == 1.0
        assert intensity_for_step(-5, warmup_steps=0) == 1.0  # negative steps still 1.0

    def test_warmup_ramps_linearly(self):
        """With warmup_steps=2000, intensity ramps from 0 to 1 over 2000 steps."""
        assert intensity_for_step(0, warmup_steps=2000) == pytest.approx(0.0)
        assert intensity_for_step(500, warmup_steps=2000) == pytest.approx(0.25)
        assert intensity_for_step(1000, warmup_steps=2000) == pytest.approx(0.5)
        assert intensity_for_step(1500, warmup_steps=2000) == pytest.approx(0.75)
        assert intensity_for_step(2000, warmup_steps=2000) == pytest.approx(1.0)

    def test_warmup_capped_at_one(self):
        """After warmup, intensity stays at 1.0."""
        assert intensity_for_step(3000, warmup_steps=2000) == pytest.approx(1.0)
        assert intensity_for_step(100_000, warmup_steps=2000) == pytest.approx(1.0)

    def test_negative_step_clamped_to_zero(self):
        """Negative step shouldn't produce negative intensity."""
        assert intensity_for_step(-100, warmup_steps=2000) == pytest.approx(0.0)


class TestScaledProbabilities:
    def test_full_intensity_returns_base_probs(self):
        """At intensity=1.0, scaled probabilities equal base probabilities."""
        result = scaled_probabilities(intensity=1.0)
        for key, value in BASE_NOISE_PROBABILITIES.items():
            assert result[key] == pytest.approx(value)

    def test_zero_intensity_zeroes_all_probs(self):
        """At intensity=0.0, all scaled probabilities are 0."""
        result = scaled_probabilities(intensity=0.0)
        for value in result.values():
            assert value == 0.0

    def test_half_intensity_halves_all_probs(self):
        """At intensity=0.5, all probabilities are halved."""
        result = scaled_probabilities(intensity=0.5)
        for key, value in BASE_NOISE_PROBABILITIES.items():
            assert result[key] == pytest.approx(value * 0.5)

    def test_known_keys_present(self):
        """Sanity: the canonical scan-noise pipeline names are exposed."""
        # These keys correspond to the transform groups in the pipeline; the test
        # ensures we don't silently rename a key that callers depend on.
        expected = {
            "image_compression",
            "noise_oneof",
            "blur_oneof",
            "brightness_contrast",
            "rotate",
            "grid_distortion",
            "elastic_transform",
        }
        assert set(BASE_NOISE_PROBABILITIES.keys()) == expected
