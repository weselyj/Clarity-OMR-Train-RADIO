"""Train a YOLO model on the mixed Stage A dataset.

Small ultralytics wrapper. Hyperparameters mirror the recovered config from
the original Stage A YOLO checkpoint (rect=True, cos_lr=True, augmentation
disabled), with the mixed dataset and chosen base model.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Base model (e.g. yolov8m.pt or yolo26m.pt)")
    parser.add_argument("--data", type=Path, required=True, help="data.yaml path")
    parser.add_argument("--name", required=True, help="Run name (becomes runs/<name>)")
    parser.add_argument("--project", default="runs", help="Top-level run dir")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1920)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.compile for ~30%% speedup. Requires triton; OOMs at "
             "batch=16 imgsz=1920. Default off.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed-precision training. Default on. The YOLOv8m baseline "
             "had a NaN at epoch 83 with AMP+low LR; pair with --nan-guard "
             "to zero NaN/Inf gradients before they corrupt weights.",
    )
    parser.add_argument(
        "--nan-guard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Wrap torch.nn.utils.clip_grad_norm_ to detect and zero out NaN/Inf "
             "gradients before optimizer step. Cheap insurance against AMP fp16 "
             "overflows that get past the GradScaler. Recommended with --amp.",
    )
    parser.add_argument(
        "--noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable scan-noise augmentation pipeline. Simulates JPEG compression, "
            "sensor noise, slight blur, brightness/contrast variation, scan skew, "
            "and gentle page curvature (non-flat scans). Intended for YOLO26m and "
            "later runs; leave off for the YOLOv8m clean-data baseline."
        ),
    )
    parser.add_argument(
        "--noise-warmup-steps",
        type=int,
        default=0,
        help=(
            "Linear warmup ramp on scan-noise intensity over the first N batches "
            "(only with --noise). Avoids the early-training cls_loss-blow-up "
            "failure mode where a fresh detection head + heavy augmentation = NaN. "
            "Default 0 = no warmup, full strength from step 0."
        ),
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Explicit gradient-clip max_norm (overrides Ultralytics' opaque "
             "internal value). Mirrors the proven Stage-B clip=1.0. Only "
             "applied when --nan-guard is on (it owns the clip_grad_norm_ hook).",
    )
    parser.add_argument(
        "--save-period",
        type=int,
        default=5,
        help="Save a checkpoint every N epochs (enables resume-from-last on a "
             "mid-run death; bounds wasted work). Default 5.",
    )
    parser.add_argument(
        "--debug-anomaly",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable torch.autograd.set_detect_anomaly (slow; off by default). "
             "Use only to capture a recurring NaN's origin.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume Ultralytics training from the checkpoint passed as "
             "--model (restores epoch/optimizer/LR state). The seder worker "
             "sets this with --model <run>/weights/last.pt after a mid-run "
             "death; a plain warm-start would re-run the full epoch schedule.",
    )
    return parser.parse_args()


def _patch_nan_guard(max_grad_norm: float | None = None) -> None:
    """Wrap torch.nn.utils.clip_grad_norm_ to zero out NaN/Inf gradients before they corrupt weights.

    AMP's GradScaler detects and skips steps when its scaled grads have inf, but
    occasional NaN can sneak through (e.g., when forward-pass numerics produce NaN
    that propagates into grads). ultralytics calls clip_grad_norm_ on every step
    right before optimizer.step(). We hook that call: enumerate parameters, scan
    each .grad tensor for NaN/Inf, zero them in-place if found, then defer to the
    original clip implementation.

    A zeroed gradient turns the optimizer step into an effective no-op for that
    parameter (preserves the previous weight). The next step continues from a
    clean state. Loss for that batch is logged as NaN but training continues.

    Counter is logged so the user can monitor frequency. Light NaN-guard hits
    (a few per epoch) are normal under AMP; heavy hits (many per epoch) suggest
    the LR or model state is unstable and other interventions may be needed.
    """
    import torch

    original_clip = torch.nn.utils.clip_grad_norm_
    nan_event_count = [0]

    def safe_clip(parameters, max_norm, *args, **kwargs):
        # Materialize parameters so we can iterate twice (NaN scan + original clip)
        params = list(parameters) if not isinstance(parameters, list) else parameters
        any_nan = False
        for p in params:
            if p.grad is not None:
                if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                    p.grad.zero_()
                    any_nan = True
        if any_nan:
            nan_event_count[0] += 1
            print(f"[nan-guard] zeroed NaN/Inf grads (occurrence #{nan_event_count[0]})")
        effective_max_norm = max_grad_norm if max_grad_norm is not None else max_norm
        return original_clip(params, effective_max_norm, *args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = safe_clip


def _register_stagea_hardening(model, *, debug_anomaly: bool) -> None:
    """Register the active halt-on-NaN callbacks (the thin seam). Loss is
    checked every batch (a cheap scalar); the full EMA finite-scan runs only
    at epoch boundary — a per-batch full-state_dict scan would add hundreds of
    GPU->CPU syncs per step over millions of steps. On a non-finite state it
    attempts an explicit checkpoint save and sets trainer.stop so the run
    stops promptly instead of wasting ~20 epochs as in the epoch-34 incident
    (EMA NaN is an accumulating condition; catching it at the next epoch
    boundary still prevents the multi-epoch waste). If the EMA is already
    non-finite, trainer.save_model() will not write a good EMA — the last
    clean periodic/best checkpoint (from --save-period) is the last-good; the
    acceptance runbook gates it via validate_checkpoint_finite."""
    import torch
    from src.train.stagea_hardening import is_nonfinite_state, should_halt

    if debug_anomaly:
        torch.autograd.set_detect_anomaly(True)

    def _loss_scalar(trainer):
        loss = getattr(trainer, "loss", None)
        if loss is None:
            return None
        if hasattr(loss, "item"):
            try:
                return float(loss.item())
            except Exception:
                return None
        return float(loss)

    def _ema_finite(trainer) -> bool:
        ema = getattr(trainer, "ema", None)
        m = getattr(ema, "ema", None) if ema is not None else None
        if m is None:
            return True
        for t in m.state_dict().values():
            if (hasattr(t, "isfinite") and t.is_floating_point()
                    and not bool(torch.isfinite(t).all().item())):
                return False
        return True

    def _halt(trainer, msg: str) -> None:
        print(f"[stagea-hardening] {msg} at epoch "
              f"{getattr(trainer, 'epoch', '?')}; saving + halting")
        try:
            trainer.save_model()
        except Exception as exc:  # never let the guard itself crash the run
            print(f"[stagea-hardening] save_model raised: {exc!r}")
        trainer.stop = True

    def _check_loss(trainer) -> None:
        # Hot path: per-batch, loss scalar only (no EMA scan).
        nf, reason = is_nonfinite_state(_loss_scalar(trainer), True)
        msg, halt = should_halt(nonfinite=nf, reason=reason)
        if halt:
            _halt(trainer, msg)

    def _check_full(trainer) -> None:
        # Epoch boundary: loss + full EMA finite-scan (acceptable cost).
        nf, reason = is_nonfinite_state(_loss_scalar(trainer),
                                        _ema_finite(trainer))
        msg, halt = should_halt(nonfinite=nf, reason=reason)
        if halt:
            _halt(trainer, msg)

    model.add_callback("on_train_batch_end", _check_loss)
    model.add_callback("on_fit_epoch_end", _check_full)


def main() -> None:
    args = parse_args()
    if args.nan_guard:
        _patch_nan_guard(max_grad_norm=args.max_grad_norm)
    if args.noise:
        from src.train.scan_noise import patch_albumentations_for_scan_noise
        patch_albumentations_for_scan_noise(warmup_steps=args.noise_warmup_steps)
    from src.train.stagea_hardening import build_hardened_overrides
    hardened = build_hardened_overrides(
        amp=args.amp, save_period=args.save_period,
        max_grad_norm=args.max_grad_norm,
    )
    model = YOLO(args.model)
    _register_stagea_hardening(model, debug_anomaly=args.debug_anomaly)
    train_kwargs = dict(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        rect=True,
        cos_lr=True,
        name=args.name,
        project=args.project,
        hsv_h=0, hsv_s=0,
        flipud=0, fliplr=0,
        mosaic=0, mixup=0,
        save=True,
        save_period=hardened.save_period,
        lr0=hardened.lr0,
        lrf=hardened.lrf,
        patience=args.patience,
        workers=args.workers,
        amp=hardened.amp,
    )
    if args.compile:
        train_kwargs["compile"] = True
    if args.resume:
        train_kwargs["resume"] = True
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
