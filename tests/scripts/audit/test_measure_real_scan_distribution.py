"""Test real-scan distribution measurement script."""
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts.audit.measure_real_scan_distribution import (
    estimate_rotation_degrees,
    estimate_noise_sigma,
    estimate_blur_laplacian_var,
)


def _synthetic_staff_image(rot_deg: float = 0.0, noise_sigma: float = 0.0) -> Image.Image:
    """Synthesize a 200x600 image with 5 horizontal staff lines, optionally rotated/noised."""
    arr = np.full((200, 600), 255, dtype=np.uint8)
    for y in [40, 70, 100, 130, 160]:
        arr[y, :] = 0
    if noise_sigma > 0:
        noise = np.random.default_rng(42).normal(0, noise_sigma * 255, arr.shape)
        arr = np.clip(arr.astype(float) + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="L")
    if rot_deg != 0.0:
        img = img.rotate(rot_deg, resample=Image.BICUBIC, fillcolor=255)
    return img


def test_estimate_rotation_recovers_known_angle():
    img = _synthetic_staff_image(rot_deg=1.0)
    estimated = estimate_rotation_degrees(img)
    assert abs(estimated - 1.0) < 0.5  # within 0.5°


def test_estimate_noise_is_monotonic():
    clean = _synthetic_staff_image(noise_sigma=0.0)
    noisy = _synthetic_staff_image(noise_sigma=0.15)
    assert estimate_noise_sigma(noisy) > estimate_noise_sigma(clean)


def test_estimate_blur_is_monotonic():
    clean = _synthetic_staff_image()
    blurred = clean.filter(__import__("PIL.ImageFilter", fromlist=["GaussianBlur"]).GaussianBlur(2.0))
    assert estimate_blur_laplacian_var(clean) > estimate_blur_laplacian_var(blurred)
