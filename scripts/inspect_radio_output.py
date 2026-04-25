"""Verify RADIO loads on the 5090 and report its forward-output shapes.

Used as Task 1's smoke test and re-used in Task 4 to inform the positional-bridge dim.
"""
import torch
from torch.nn import functional as F


def main():
    print("Loading C-RADIOv4-H from NVlabs/RADIO via torchhub...")
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version="c-radio_v4-h",
        progress=True,
        skip_validation=True,
        force_reload=True,  # first run: re-pull source
        trust_repo=True,
    )
    model.cuda().eval()
    print(f"Model loaded. Hidden dim: {getattr(model, 'embed_dim', 'unknown')}")

    # Stage B input: grayscale staff crop, height 192, width up to 2048
    h, w = 192, 1024
    x = torch.randn(1, 3, h, w).cuda()  # RADIO expects 3-channel; grayscale->RGB upstream

    # Snap to RADIO's nearest supported resolution
    nearest = model.get_nearest_supported_resolution(h, w)
    print(f"Requested: {(h, w)}, RADIO nearest: {nearest}")
    if nearest != (h, w):
        x = F.interpolate(x, nearest, mode="bilinear", align_corners=False)

    # RADIO expects [0, 1]
    x = x.sigmoid()

    with torch.autocast("cuda", dtype=torch.bfloat16), torch.no_grad():
        summary, spatial_features = model(x, feature_fmt="NCHW")

    print(f"Summary shape: {tuple(summary.shape)}")
    print(f"Spatial features shape (NCHW): {tuple(spatial_features.shape)}")
    print(f"Spatial features dtype: {spatial_features.dtype}")
    expected_grid = (nearest[0] // 16, nearest[1] // 16)
    actual_grid = tuple(spatial_features.shape[-2:])
    print(f"Expected grid (H/16, W/16): {expected_grid}")
    print(f"Actual grid: {actual_grid}")
    assert actual_grid == expected_grid, "Grid mismatch — check patch size assumption"
    print("OK — RADIO loaded, ran forward, grid shape matches patch-16 assumption.")


if __name__ == "__main__":
    main()
