"""Unit tests for scan_degradation pipeline.

CPU-only, run locally — no GPU dependency.
"""
import numpy as np
from PIL import Image
import pytest

from src.data.scan_degradation import apply_scan_degradation


def _make_test_image(size=(800, 200)) -> Image.Image:
    """Synthesize a grayscale image with simulated staff lines."""
    arr = np.full((size[1], size[0]), 255, dtype=np.uint8)
    for y in [40, 70, 100, 130, 160]:
        arr[y, :] = 0
    return Image.fromarray(arr, mode="L")


def test_apply_is_deterministic_for_fixed_seed():
    img = _make_test_image()
    out_a = apply_scan_degradation(img, seed=12345)
    out_b = apply_scan_degradation(img, seed=12345)
    assert np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_apply_differs_for_different_seeds():
    img = _make_test_image()
    out_a = apply_scan_degradation(img, seed=12345)
    out_b = apply_scan_degradation(img, seed=67890)
    assert not np.array_equal(np.asarray(out_a), np.asarray(out_b))


def test_output_dimensions_preserved():
    img = _make_test_image(size=(800, 200))
    out = apply_scan_degradation(img, seed=12345)
    assert out.size == (800, 200)


def test_output_is_grayscale_when_input_is_grayscale():
    img = _make_test_image()
    out = apply_scan_degradation(img, seed=12345)
    assert out.mode == "L"


def test_output_differs_from_input():
    img = _make_test_image()
    out = apply_scan_degradation(img, seed=12345)
    assert not np.array_equal(np.asarray(img), np.asarray(out)), \
        "Degradation must actually change the image"
