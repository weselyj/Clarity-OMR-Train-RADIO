"""Validate that DoRA wraps the expected RADIO modules and gradients flow through them.

Run from the repo root:
    venv\\Scripts\\python scripts\\validate_dora_attachment.py

Expected output:
    DoRA-wrapped modules: <N>
    Encoder DoRA-wrapped: <N>
    encoder DoRA params with non-None grad: <N>
    sum of grad norms: <float>
    non-DoRA encoder params with requires_grad=True: 0
    trainable params: ~30.12M of ~757.37M total
    OK -- DoRA attached, encoder frozen, gradients flow through encoder.
"""
import sys
from pathlib import Path

# Allow running from scripts/ or from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F


def main():
    # Build RADIO model + DoRA config via the factory.
    from src.train.model_factory import build_stage_b_components, ModelFactoryConfig

    print("Building RADIO Stage B model (this downloads RADIO weights if not cached)...")
    mfc = ModelFactoryConfig(stage_b_encoder="radio_h")
    components = build_stage_b_components(mfc)
    model = components["model"].cuda()
    dora_config = components["dora_config"]

    # Apply DoRA. _prepare_model_for_dora takes (model, dora_config: dict).
    print("Applying DoRA adapters...")
    from src.train.train import _prepare_model_for_dora
    model, dora_applied = _prepare_model_for_dora(model, dora_config)
    assert dora_applied, "_prepare_model_for_dora returned dora_applied=False"

    # Count DoRA-wrapped modules. In PEFT 0.19.1, use_dora=True wraps via LoraLayer
    # (there is no separate DoraLayer class in this version).
    from peft.tuners.lora import LoraLayer
    dora_count = sum(1 for m in model.modules() if isinstance(m, LoraLayer))
    print(f"DoRA-wrapped modules: {dora_count}")
    assert dora_count > 0, "DoRA wrapped zero modules -- DoRA targets are wrong"

    # Encoder-specific DoRA count.
    encoder_dora = sum(
        1 for n, m in model.named_modules()
        if isinstance(m, LoraLayer) and "encoder" in n
    )
    print(f"Encoder DoRA-wrapped: {encoder_dora}")
    assert encoder_dora >= 100, (
        f"Expected ~128 encoder DoRA modules, got {encoder_dora}. "
        "Check list_radio_dora_target_modules() in src/train/model_factory.py."
    )

    # Forward + backward pass.
    # Use keyword-argument form: PEFT's PeftModel wrapper intercepts positional args and
    # the RadioStageB.forward supports pixel_values/input_ids aliases for compatibility.
    print("Running forward + backward pass...")
    model.train()
    img = torch.rand(1, 1, 192, 1024).cuda()
    tgt = torch.zeros(1, 32, dtype=torch.long).cuda()
    out = model(pixel_values=img, input_ids=tgt)
    loss = F.cross_entropy(
        out["logits"].view(-1, out["logits"].shape[-1]),
        tgt.view(-1),
    )
    loss.backward()

    # Confirm gradients flow through encoder DoRA params.
    encoder_grad_norm = 0.0
    encoder_param_count = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        if "encoder" in name and ("lora_" in name or "dora_" in name):
            encoder_grad_norm += p.grad.norm().item()
            encoder_param_count += 1

    print(f"encoder DoRA params with non-None grad: {encoder_param_count}")
    print(f"sum of grad norms: {encoder_grad_norm:.4f}")
    assert encoder_param_count > 0, (
        "No encoder DoRA params received gradients. "
        "Check if RADIO encoder is in eval() mode or has requires_grad=False internally."
    )
    assert encoder_grad_norm > 0, (
        "Encoder DoRA grad norm is zero -- gradients aren't flowing through the encoder."
    )

    # Verify encoder base weights are frozen (DoRA freeze logic should cover them all).
    base_trainable = sum(
        1 for n, p in model.named_parameters()
        if "encoder" in n and "lora_" not in n and "dora_" not in n and p.requires_grad
    )
    print(f"non-DoRA encoder params with requires_grad=True: {base_trainable}")
    # Allow a small number for PEFT's modules_to_save entries, but not the full encoder.
    assert base_trainable < 100, (
        f"Too many non-DoRA encoder params are trainable ({base_trainable}). "
        "Encoder base weights should be frozen when DoRA is applied."
    )

    # Trainable parameter budget check.
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"trainable params: {total_trainable:.2f}M of {total_params:.2f}M total")
    assert 5 < total_trainable < 80, (
        f"Trainable params outside expected range: {total_trainable:.1f}M. "
        "Expected ~30M (DoRA adapters + decoder + bridge). "
        "If RADIO architecture changed, update the bounds."
    )

    # Assert no DoRA magnitude weights are zero (zero-row NaN fix actually worked).
    # If this fails, the fix in src/train/train.py is broken — see Task 7 review for context.
    from peft.tuners.lora.dora import DoraLinearLayer

    zero_magnitude_rows = []
    for name, mod in model.named_modules():
        if not isinstance(mod, LoraLayer):
            continue
        if not hasattr(mod, "lora_magnitude_vector"):
            continue
        for adapter, mag in mod.lora_magnitude_vector.items():
            if not isinstance(mag, DoraLinearLayer):
                continue
            zero_count = int((mag.weight.data == 0).sum())
            if zero_count > 0:
                zero_magnitude_rows.append((name, adapter, zero_count))

    assert not zero_magnitude_rows, (
        f"DoRA magnitude weights have zero entries (NaN fix in train.py is broken):\n"
        + "\n".join(f"  {n}[{a}]: {c} zero rows" for n, a, c in zero_magnitude_rows)
    )
    print(
        f"DoRA magnitude zero-row check passed "
        f"(0 layers have zero magnitude entries after NaN fix)."
    )

    print(
        "\nOK -- DoRA attached, encoder frozen, gradients flow through encoder."
    )


if __name__ == "__main__":
    main()
