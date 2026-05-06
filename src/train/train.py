#!/usr/bin/env python3
"""Curriculum training driver for Stage-B staff recognition."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.utils.data
import yaml

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.tokenizer.vocab import build_default_vocabulary
from src.train.model_factory import (
    ModelFactoryConfig,
    build_stage_b_components,
    model_factory_config_from_checkpoint_payload,
)


PITCH_CLASS_TO_SEMITONE = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}

CONTOUR_DOWN = 0
CONTOUR_SAME = 1
CONTOUR_UP = 2


@dataclass(frozen=True)
class DatasetMix:
    dataset: str
    ratio: float
    split: str = "train"
    required: bool = False


@dataclass(frozen=True)
class StageTrainingConfig:
    stage_name: str
    epochs: int
    effective_samples_per_epoch: int
    batch_size: int
    max_sequence_length: int
    lr_dora: float
    lr_new_modules: float
    warmup_steps: int
    schedule: str
    label_smoothing: float
    contour_loss_weight: float
    weight_decay: float
    checkpoint_every_steps: int
    validate_every_steps: int
    grad_accumulation_steps: int
    loraplus_lr_ratio: float
    dataset_mix: Tuple[DatasetMix, ...]
    stage_b_encoder: str = "davit"


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must decode to an object: {path}")
    return payload


def load_stage_config(path: Path) -> StageTrainingConfig:
    raw = load_yaml(path)
    mix_raw = raw.get("dataset_mix", [])
    if not isinstance(mix_raw, list) or not mix_raw:
        raise ValueError(f"dataset_mix must be a non-empty list in {path}")

    mix: List[DatasetMix] = []
    for item in mix_raw:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid dataset mix entry in {path}: {item}")
        mix.append(
            DatasetMix(
                dataset=str(item["dataset"]).lower(),
                ratio=float(item["ratio"]),
                split=str(item.get("split", "train")).lower(),
                required=bool(item.get("required", False)),
            )
        )

    ratio_sum = sum(item.ratio for item in mix)
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"dataset_mix ratios must sum to 1.0 in {path}, got {ratio_sum}")

    return StageTrainingConfig(
        stage_name=str(raw["stage_name"]),
        epochs=int(raw["epochs"]),
        effective_samples_per_epoch=int(raw["effective_samples_per_epoch"]),
        batch_size=int(raw["batch_size"]),
        max_sequence_length=int(raw["max_sequence_length"]),
        lr_dora=float(raw["lr_dora"]),
        lr_new_modules=float(raw["lr_new_modules"]),
        warmup_steps=int(raw["warmup_steps"]),
        schedule=str(raw.get("schedule", "cosine")).lower(),
        label_smoothing=float(raw.get("label_smoothing", 0.0)),
        contour_loss_weight=float(raw.get("contour_loss_weight", 0.1)),
        weight_decay=max(0.0, float(raw.get("weight_decay", 0.01))),
        checkpoint_every_steps=max(1, int(raw.get("checkpoint_every_steps", 1000))),
        validate_every_steps=max(1, int(raw.get("validate_every_steps", 500))),
        grad_accumulation_steps=max(1, int(raw.get("grad_accumulation_steps", 1))),
        loraplus_lr_ratio=float(raw.get("loraplus_lr_ratio", 1.0)),
        dataset_mix=tuple(mix),
        stage_b_encoder=str(raw.get("stage_b_encoder", "davit")).lower().strip(),
    )


def load_token_manifest(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Token manifest not found: {path}")
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON line in {path}:{line_no}") from exc
    return entries


def _resolve_manifest_paths(project_root: Path, manifest_arg: str) -> List[Path]:
    resolved: List[Path] = []
    for manifest_str in manifest_arg.split(","):
        raw = manifest_str.strip()
        if not raw:
            continue
        manifest_path = Path(raw)
        if not manifest_path.is_absolute():
            manifest_path = project_root / manifest_path
        resolved.append(manifest_path.resolve())
    return resolved


def _assert_not_stale_merged_manifest(project_root: Path, manifest_paths: Sequence[Path]) -> None:
    if len(manifest_paths) != 1:
        return
    merged_path = manifest_paths[0]
    if merged_path.name.lower() != "token_manifest_train.jsonl":
        return
    if not merged_path.exists():
        return

    base_manifest = (project_root / "src" / "data" / "manifests" / "token_manifest.jsonl").resolve()
    synthetic_manifest = (
        project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"
    ).resolve()
    dependencies = [path for path in (base_manifest, synthetic_manifest) if path.exists()]
    if len(dependencies) < 2:
        return

    merged_mtime = merged_path.stat().st_mtime
    newer_dependencies = [path for path in dependencies if path.stat().st_mtime > (merged_mtime + 1.0)]
    if not newer_dependencies:
        return

    newer_paths = ", ".join(str(path) for path in newer_dependencies)
    raise RuntimeError(
        "Detected stale merged token manifest "
        f"'{merged_path}'. Newer dependency manifest(s): {newer_paths}. "
        "Regenerate token_manifest_train.jsonl or pass comma-separated manifests directly "
        "(src/data/manifests/token_manifest.jsonl,data/processed/synthetic/manifests/synthetic_token_manifest.jsonl)."
    )


def group_entries_by_dataset_and_split(
    entries: Sequence[Dict[str, object]]
) -> Dict[Tuple[str, str], List[Dict[str, object]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, object]]] = {}
    for entry in entries:
        dataset = str(entry.get("dataset", "")).lower()
        split = str(entry.get("split", "train")).lower()
        grouped.setdefault((dataset, split), []).append(entry)
    return grouped


def sanitize_token_entries(
    entries: Sequence[Dict[str, object]],
    *,
    enforce_strict_sequences: bool = True,
    allow_relaxed_fallback: bool = True,
) -> Tuple[List[Dict[str, object]], int]:
    if not enforce_strict_sequences:
        return list(entries), 0

    from src.data.convert_tokens import validate_token_sequence

    vocab = build_default_vocabulary()
    cleaned: List[Dict[str, object]] = []
    dropped = 0
    for entry in entries:
        sequence = entry.get("token_sequence", [])
        if not isinstance(sequence, list) or not sequence:
            dropped += 1
            continue
        normalized_sequence = [str(token) for token in sequence]
        try:
            vocab.encode(normalized_sequence, strict=True)
        except Exception:
            dropped += 1
            continue
        try:
            validate_token_sequence(normalized_sequence, strict=True)
        except Exception:
            if not allow_relaxed_fallback:
                dropped += 1
                continue
            try:
                validate_token_sequence(normalized_sequence, strict=False)
            except Exception:
                dropped += 1
                continue
        cleaned.append(entry)
    return cleaned, dropped


def _compute_sample_targets(total: int, mix: Sequence[DatasetMix]) -> Dict[str, int]:
    raw_targets = [total * item.ratio for item in mix]
    floored = [int(math.floor(value)) for value in raw_targets]
    remainder = total - sum(floored)
    order = sorted(range(len(mix)), key=lambda idx: raw_targets[idx] - floored[idx], reverse=True)
    for idx in order[:remainder]:
        floored[idx] += 1
    return {mix[idx].dataset: floored[idx] for idx in range(len(mix))}


def build_stage_plan(
    stage: StageTrainingConfig,
    grouped_entries: Dict[Tuple[str, str], List[Dict[str, object]]],
) -> Dict[str, object]:
    targets = _compute_sample_targets(stage.effective_samples_per_epoch, stage.dataset_mix)
    source_summary: Dict[str, Dict[str, object]] = {}
    warnings: List[str] = []

    for source in stage.dataset_mix:
        available = len(grouped_entries.get((source.dataset, source.split), []))
        target = targets[source.dataset]
        source_summary[source.dataset] = {
            "split": source.split,
            "required": source.required,
            "available_samples": available,
            "target_samples_per_epoch": target,
        }
        if source.required and available == 0:
            raise ValueError(
                f"Required dataset '{source.dataset}' split '{source.split}' is empty for stage '{stage.stage_name}'."
            )
        if available == 0:
            warnings.append(
                f"Dataset '{source.dataset}' split '{source.split}' has no samples; target={target} will be deferred."
            )

    return {
        "stage_name": stage.stage_name,
        "epochs": stage.epochs,
        "effective_samples_per_epoch": stage.effective_samples_per_epoch,
        "batch_size": stage.batch_size,
        "max_sequence_length": stage.max_sequence_length,
        "lr_dora": stage.lr_dora,
        "lr_new_modules": stage.lr_new_modules,
        "warmup_steps": stage.warmup_steps,
        "schedule": stage.schedule,
        "label_smoothing": stage.label_smoothing,
        "contour_loss_weight": stage.contour_loss_weight,
        "weight_decay": stage.weight_decay,
        "checkpoint_every_steps": stage.checkpoint_every_steps,
        "validate_every_steps": stage.validate_every_steps,
        "sources": source_summary,
        "warnings": warnings,
    }



def _parse_note_token_to_midi(token: str) -> Optional[int]:
    if not token.startswith("note-"):
        return None
    symbol = token[5:]
    if len(symbol) < 2:
        return None
    octave_text = symbol[-1]
    pitch_class = symbol[:-1]
    if not octave_text.isdigit():
        return None
    if pitch_class not in PITCH_CLASS_TO_SEMITONE:
        return None
    octave = int(octave_text)
    return 12 * (octave + 1) + PITCH_CLASS_TO_SEMITONE[pitch_class]


def _derive_pitch_contour(sequence: Sequence[str]) -> int:
    notes: List[int] = []
    for token in sequence:
        pitch = _parse_note_token_to_midi(str(token))
        if pitch is None:
            continue
        notes.append(pitch)
        if len(notes) >= 2:
            break
    if len(notes) < 2:
        return CONTOUR_SAME
    if notes[1] > notes[0]:
        return CONTOUR_UP
    if notes[1] < notes[0]:
        return CONTOUR_DOWN
    return CONTOUR_SAME


def _resolve_project_path(project_root: Path, value: object) -> Optional[Path]:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    resolved = path.resolve()
    if not resolved.exists():
        return None
    return resolved


def _fit_grayscale_to_canvas(image_obj, *, height: int, max_width: int):
    from PIL import Image

    source_width, source_height = image_obj.size
    if source_height <= 0:
        raise ValueError("Invalid source image height.")
    target_width = int(round((float(source_width) / float(source_height)) * float(height)))
    target_width = max(1, min(max_width, target_width))
    resized = image_obj.convert("L").resize((target_width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (max_width, height), color=255)
    canvas.paste(resized, (0, 0))
    return canvas, target_width


def _load_raster_image_tensor(path: Path, *, height: int, max_width: int):
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image_obj:
        canvas, content_width = _fit_grayscale_to_canvas(image_obj, height=height, max_width=max_width)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0), int(content_width)


def _render_symbolic_source_tensor(path: Path, *, height: int, max_width: int):
    import numpy as np
    import torch
    import verovio
    from PIL import Image
    from cairosvg import svg2png
    from io import BytesIO

    candidates = [path]
    for extension in (".mxl", ".mscx", ".musicxml", ".xml"):
        candidate = path.with_suffix(extension)
        if candidate.exists() and candidate not in candidates:
            candidates.append(candidate)

    svg_text = None
    for candidate in candidates:
        renderer = verovio.toolkit()
        renderer.loadFile(str(candidate))
        renderer.setOptions(
            {
                "font": "Bravura",
                "pageWidth": int(max_width * 3),
                "pageHeight": int(height * 3),
                "scale": 40,
                "breaks": "auto",
                "svgBoundingBoxes": True,
                "svgViewBox": True,
            }
        )
        renderer.redoLayout()
        page_count = int(renderer.getPageCount())
        if page_count < 1:
            continue
        svg_text = renderer.renderToSVG(1)
        break

    if svg_text is None:
        raise ValueError(f"Could not render symbolic source to image: {path}")

    png_bytes = svg2png(bytestring=svg_text.encode("utf-8"), background_color="white")
    with Image.open(BytesIO(png_bytes)) as image_obj:
        if "A" in image_obj.getbands():
            rgba = image_obj.convert("RGBA")
            white_bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            grayscale_source = Image.alpha_composite(white_bg, rgba).convert("L")
        else:
            grayscale_source = image_obj.convert("L")
        canvas, content_width = _fit_grayscale_to_canvas(grayscale_source, height=height, max_width=max_width)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0), int(content_width)


def _load_entry_image_tensor(
    entry: Dict[str, object],
    *,
    project_root: Path,
    height: int,
    max_width: int,
):
    sample_id = str(entry.get("sample_id", "<unknown>"))
    image_path = _resolve_project_path(project_root, entry.get("image_path"))
    if image_path is not None:
        return _load_raster_image_tensor(image_path, height=height, max_width=max_width)

    source_path = _resolve_project_path(project_root, entry.get("source_path"))
    if source_path is None:
        raise FileNotFoundError(
            f"Missing image_path and source_path for sample '{sample_id}'. "
            "Regenerate token manifest after converter update."
        )

    # Backward compatibility for older token manifests that omitted image_path.
    for extension in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        candidate = source_path.with_suffix(extension)
        if candidate.exists():
            return _load_raster_image_tensor(candidate, height=height, max_width=max_width)

    if source_path.suffix.lower() in {".mxl", ".musicxml", ".xml", ".mscx"}:
        return _render_symbolic_source_tensor(source_path, height=height, max_width=max_width)

    raise FileNotFoundError(
        f"Could not resolve image for sample '{sample_id}'. "
        f"Tried image_path/source-derived raster for '{source_path}'."
    )



def _apply_online_augmentations(images, rng: random.Random):
    from io import BytesIO

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import functional as TF

    augmented = images.clone()
    for idx in range(augmented.shape[0]):
        image = augmented[idx]
        if rng.random() < 0.8:
            image = TF.affine(
                image,
                angle=rng.uniform(-2.0, 2.0),
                translate=[0, 0],
                scale=rng.uniform(0.92, 1.08),
                shear=[0.0, 0.0],
            )
        if rng.random() < 0.65:
            image = TF.adjust_brightness(image, rng.uniform(0.90, 1.10))
        if rng.random() < 0.65:
            image = TF.adjust_contrast(image, rng.uniform(0.88, 1.12))
        if rng.random() < 0.45:
            image = TF.gaussian_blur(image, kernel_size=[3, 3], sigma=rng.uniform(0.0, 1.0))
        if rng.random() < 0.25:
            quality = int(rng.randint(70, 95))
            as_uint8 = (
                image.squeeze(0).detach().cpu().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
            )
            source = Image.fromarray(as_uint8, mode="L")
            buffer = BytesIO()
            source.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            with Image.open(buffer) as jpeg_image:
                restored = np.asarray(jpeg_image.convert("L"), dtype=np.float32) / 255.0
            image = torch.from_numpy(restored).unsqueeze(0).to(device=image.device, dtype=image.dtype)
        if rng.random() < 0.25:
            h, w = image.shape[-2:]
            factor = rng.uniform(0.85, 1.0)
            down = F.interpolate(
                image.unsqueeze(0),
                size=(max(1, int(h * factor)), max(1, int(w * factor))),
                mode="bilinear",
                align_corners=False,
            )
            image = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False).squeeze(0)
        if rng.random() < 0.30:
            array = image.squeeze(0).detach().cpu().numpy().copy()
            flat = array.reshape(-1)
            noise_fraction = rng.uniform(0.001, 0.0025)
            noise_count = max(1, int(round(flat.size * noise_fraction)))
            generator = np.random.default_rng(rng.randrange(0, 2**32))
            indices = generator.choice(flat.size, size=noise_count, replace=False)
            split_idx = noise_count // 2
            flat[indices[:split_idx]] = 0.0
            flat[indices[split_idx:]] = 1.0
            image = torch.from_numpy(array).unsqueeze(0).to(device=image.device, dtype=image.dtype)
        augmented[idx] = image.clamp(0.0, 1.0)
    return augmented


class StageBDataset(torch.utils.data.Dataset):
    """torch.utils.data.Dataset wrapping a grouped_entries manifest for Stage-B.

    Each __getitem__ call:
      1. Loads + resizes the image (via _load_entry_image_tensor).
      2. Encodes the token sequence (vocab encode + truncate + pad).
      3. Optionally applies online augmentations per-sample.
      4. Returns a per-sample dict compatible with the training loop's
         expected tensor shapes.

    Collation: call ``dataset.collate_fn`` as the DataLoader's collate_fn.
    """

    def __init__(
        self,
        stage: "StageTrainingConfig",
        grouped_entries: "Dict[Tuple[str, str], List[Dict[str, object]]]",
        *,
        split: str = "train",
        project_root: "Path",
        image_height: int = 250,
        image_width: int = 2500,
        max_sequence_length: int = 512,
        vocab=None,
        augment: bool = True,
        rng_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.stage = stage
        self.split = split
        self.project_root = project_root
        self.image_height = image_height
        self.image_width = image_width
        self.max_sequence_length = max_sequence_length
        self.augment = augment

        # Build vocab once here; avoids repeated rebuild in __getitem__.
        if vocab is not None:
            self._vocab = vocab
        else:
            self._vocab = build_default_vocabulary()

        # Flatten grouped_entries for all (dataset, split) pairs referenced by
        # this stage's dataset_mix into a single self.entries list.
        # The ``split`` arg overrides the per-mix-item split so that the same
        # dataset_mix definition can be reused to build a val-side dataset.
        # Each entry dict is kept as-is; the flat index is what the sampler uses.
        self.entries: List[Dict[str, object]] = []
        for mix_item in stage.dataset_mix:
            key = (mix_item.dataset, split)
            rows = grouped_entries.get(key, [])
            self.entries.extend(rows)

        # Per-sample RNG for augmentations; seeded to rng_seed when given.
        # Worker-safe: stage_b_worker_init_fn re-seeds _rng per worker so each
        # worker draws from a distinct RNG stream (avoids augmentation collapse
        # when num_workers > 0).
        self._rng_base_seed = rng_seed
        self._rng = random.Random(rng_seed)

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> "Dict[str, torch.Tensor]":
        import torch

        entry = self.entries[idx]
        vocab = self._vocab
        pad_id = vocab.token_to_id["<pad>"]
        bos_id = vocab.token_to_id["<bos>"]
        eos_id = vocab.token_to_id["<eos>"]
        measure_end_id = vocab.token_to_id.get("<measure_end>")

        # 1. Load + resize image
        sample_id = str(entry.get("sample_id", f"<idx:{idx}>"))
        try:
            image_tensor, content_width = _load_entry_image_tensor(
                entry,
                project_root=self.project_root,
                height=self.image_height,
                max_width=self.image_width,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            # Return a blank image + minimal token sequence so the batch stays
            # consistent in shape even when a single file is missing.
            import torch as _torch
            print(f"[StageBDataset] skipping {sample_id}: {exc}", file=sys.stderr)
            image_tensor = _torch.zeros(1, self.image_height, self.image_width, dtype=_torch.float32)
            content_width = self.image_width

        # 2. Token encode
        sequence = entry.get("token_sequence", [])
        if not isinstance(sequence, list) or not sequence:
            sequence = ["<bos>", "<eos>"]
        try:
            token_ids = vocab.encode(sequence, strict=True)
        except KeyError:
            token_ids = [bos_id, eos_id]
        if len(token_ids) < 2:
            token_ids = [bos_id, eos_id]
        if len(token_ids) > self.max_sequence_length:
            truncated = token_ids[: self.max_sequence_length - 1]
            if measure_end_id is not None:
                last_me = -1
                for i in range(len(truncated) - 1, -1, -1):
                    if truncated[i] == measure_end_id:
                        last_me = i
                        break
                if last_me > 0:
                    token_ids = truncated[: last_me + 1] + [eos_id]
                else:
                    token_ids = truncated + [eos_id]
            else:
                token_ids = truncated + [eos_id]

        # 3. Apply online augmentations (per-sample)
        if self.augment:
            image_tensor = _apply_online_augmentations(image_tensor.unsqueeze(0), self._rng).squeeze(0)

        # 4. Derive contour target
        contour_target = _derive_pitch_contour(sequence)

        # 5. Build decoder inputs / labels via teacher-forcing shift
        input_ids = token_ids[:-1]
        label_ids = token_ids[1:]
        if not input_ids:
            input_ids = [bos_id]
            label_ids = [eos_id]
        seq_len = self.max_sequence_length - 1
        input_pad = [pad_id] * max(0, seq_len - len(input_ids))
        label_pad = [-100] * max(0, seq_len - len(label_ids))
        decoder_inputs = (input_ids + input_pad)[:seq_len]
        labels = (label_ids + label_pad)[:seq_len]

        return {
            "images": image_tensor,
            "decoder_inputs": torch.tensor(decoder_inputs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "contour_targets": torch.tensor(contour_target, dtype=torch.long),
            "content_widths": torch.tensor(int(content_width), dtype=torch.long),
        }

    @staticmethod
    def collate_fn(samples: "List[Dict[str, torch.Tensor]]") -> "Dict[str, torch.Tensor]":
        """Stack a list of per-sample dicts into a batched dict of tensors."""
        import torch

        images = torch.stack([s["images"] for s in samples], dim=0)
        decoder_inputs = torch.stack([s["decoder_inputs"] for s in samples], dim=0)
        labels = torch.stack([s["labels"] for s in samples], dim=0)
        contour_targets = torch.stack([s["contour_targets"] for s in samples], dim=0)
        content_widths = torch.stack([s["content_widths"] for s in samples], dim=0)
        return {
            "images": images,
            "decoder_inputs": decoder_inputs,
            "labels": labels,
            "contour_targets": contour_targets,
            "content_widths": content_widths,
        }


def build_stage_b_sampler(
    stage: "StageTrainingConfig",
    dataset: "StageBDataset",
    *,
    total_samples: int,
    seed: int = 0,
    split_override: Optional[str] = None,
) -> "torch.utils.data.WeightedRandomSampler":
    """Build a WeightedRandomSampler that reproduces the stage dataset_mix ratios.

    Each sample in the dataset is weighted by:
        weight_i = mix_ratio(dataset_i) / len(group_i)

    This yields the correct long-term proportions for each dataset without
    enforcing exact per-batch counts (the legacy floor+remainder allocation
    per batch has been removed in favour of this sampler).

    Args:
        stage: StageTrainingConfig carrying the dataset_mix.
        dataset: The StageBDataset instance whose .entries list we weight.
        total_samples: Total number of indices the sampler will yield (with
            replacement). Equivalent to one epoch's sample budget.
        seed: Generator seed for reproducibility.
        split_override: When set (e.g. ``"val"``), use this split string in
            place of ``mix_item.split`` when building ratio and group-size
            maps.  Pass the same value used to construct the StageBDataset so
            that the ratio lookup correctly matches val-split entries.

    Returns:
        A WeightedRandomSampler that yields ``total_samples`` indices.
    """
    import torch

    # Build a weight-per-entry list aligned with dataset.entries.
    # Weights are computed as: ratio / len(group) for each entry.
    # If a dataset has ratio=0 its entries get weight 0 and are never drawn.
    # When split_override is given, substitute it for mix_item.split so that
    # val-split entries (which differ from the mix definition's "train" split)
    # are still weighted by the correct dataset ratio.
    def _effective_split(mix_item: "DatasetMix") -> str:
        return split_override if split_override is not None else mix_item.split

    ratio_map: Dict[Tuple[str, str], float] = {
        (mix_item.dataset, _effective_split(mix_item)): mix_item.ratio
        for mix_item in stage.dataset_mix
    }
    group_size_map: Dict[Tuple[str, str], int] = {}
    for mix_item in stage.dataset_mix:
        key = (mix_item.dataset, _effective_split(mix_item))
        group_size_map[key] = sum(
            1 for e in dataset.entries
            if e.get("dataset") == mix_item.dataset and e.get("split", "train") == _effective_split(mix_item)
        )

    weights: List[float] = []
    for entry in dataset.entries:
        ds_name = str(entry.get("dataset", ""))
        split = str(entry.get("split", "train"))
        key = (ds_name, split)
        ratio = ratio_map.get(key, 0.0)
        group_sz = group_size_map.get(key, 1)
        if ratio <= 0.0 or group_sz <= 0:
            weights.append(0.0)
        else:
            weights.append(ratio / group_sz)

    # Robustness guard: an all-zero weight vector (or empty dataset) crashes
    # WeightedRandomSampler with an opaque torch error. Raise early with a
    # diagnostic message naming the split + dataset_mix so the call site can
    # see what's wrong (e.g. dataset_mix references a split that has no
    # samples in the manifest).
    if not weights or sum(weights) <= 0.0:
        split_for_msg = split_override if split_override is not None else "<per-mix-item>"
        mix_summary = ", ".join(
            f"{m.dataset}@{_effective_split(m)}={ratio_map.get((m.dataset, _effective_split(m)), 0.0):.3f}"
            for m in stage.dataset_mix
        )
        raise ValueError(
            "build_stage_b_sampler: no samples have non-zero weight. "
            f"split_override={split_for_msg!r}, dataset_size={len(dataset.entries)}, "
            f"ratios=[{mix_summary}]. "
            "Likely cause: dataset_mix references datasets/splits with no entries "
            "in the manifest (common for split='val' if the manifest hasn't been "
            "regenerated with val splits). Fix: add entries for the requested "
            "(dataset, split) combinations or adjust dataset_mix."
        )

    weights_tensor = torch.tensor(weights, dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.WeightedRandomSampler(
        weights=weights_tensor,
        num_samples=total_samples,
        replacement=True,
        generator=generator,
    )


def stage_b_worker_init_fn(worker_id: int) -> None:
    """Re-seed each DataLoader worker's dataset RNG to avoid augmentation collisions.

    With num_workers > 0, all workers share the same initial dataset state
    (fork: same RNG object; spawn: re-imported with same seed). This function
    is passed as ``worker_init_fn`` to the DataLoader so each worker gets a
    distinct seed derived from the base seed plus its worker id.

    Args:
        worker_id: The integer worker id supplied by the DataLoader (0-indexed).
    """
    import torch.utils.data

    info = torch.utils.data.get_worker_info()
    if info is None:
        return  # called in main process — no-op
    dataset = info.dataset
    base_seed = getattr(dataset, "_rng_base_seed", None)
    if base_seed is None:
        # Dataset was constructed without a seed; derive one from worker id only.
        dataset._rng = random.Random(worker_id)
    else:
        dataset._rng = random.Random(base_seed + worker_id)


def _build_optimizer(model, stage: StageTrainingConfig):
    import torch

    weight_decay = max(0.0, float(stage.weight_decay))
    no_decay_keywords = ("bias", "norm", "embedding")
    lr_ratio = stage.loraplus_lr_ratio
    # LoRA+ splits: lora_A gets base LR, lora_B + lora_magnitude get ratio * LR
    dora_a_decay_params = []
    dora_a_no_decay_params = []
    dora_b_decay_params = []
    dora_b_no_decay_params = []
    new_module_decay_params = []
    new_module_no_decay_params = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = str(name).lower()
        use_no_decay = any(keyword in lowered for keyword in no_decay_keywords)
        if "lora_" in name:
            # lora_B and lora_magnitude get the higher learning rate
            is_b_or_mag = "lora_B" in name or "lora_magnitude" in name
            if is_b_or_mag:
                if use_no_decay:
                    dora_b_no_decay_params.append(parameter)
                else:
                    dora_b_decay_params.append(parameter)
            else:
                if use_no_decay:
                    dora_a_no_decay_params.append(parameter)
                else:
                    dora_a_decay_params.append(parameter)
        else:
            if use_no_decay:
                new_module_no_decay_params.append(parameter)
            else:
                new_module_decay_params.append(parameter)

    param_groups = []
    if dora_a_decay_params:
        param_groups.append(
            {
                "params": dora_a_decay_params,
                "lr": stage.lr_dora,
                "weight_decay": weight_decay,
                "group_name": "dora_A",
            }
        )
    if dora_a_no_decay_params:
        param_groups.append(
            {
                "params": dora_a_no_decay_params,
                "lr": stage.lr_dora,
                "weight_decay": 0.0,
                "group_name": "dora_A",
            }
        )
    if dora_b_decay_params:
        param_groups.append(
            {
                "params": dora_b_decay_params,
                "lr": stage.lr_dora * lr_ratio,
                "weight_decay": weight_decay,
                "group_name": "dora_B",
            }
        )
    if dora_b_no_decay_params:
        param_groups.append(
            {
                "params": dora_b_no_decay_params,
                "lr": stage.lr_dora * lr_ratio,
                "weight_decay": 0.0,
                "group_name": "dora_B",
            }
        )
    if new_module_decay_params:
        param_groups.append(
            {
                "params": new_module_decay_params,
                "lr": stage.lr_new_modules,
                "weight_decay": weight_decay,
                "group_name": "new_modules",
            }
        )
    if new_module_no_decay_params:
        param_groups.append(
            {
                "params": new_module_no_decay_params,
                "lr": stage.lr_new_modules,
                "weight_decay": 0.0,
                "group_name": "new_modules",
            }
        )
    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimizer setup.")
    fused = bool(torch.cuda.is_available())
    return torch.optim.AdamW(param_groups, fused=fused)


def _build_scheduler(optimizer, stage: StageTrainingConfig, total_steps: int):
    import torch

    warmup_steps = max(1, stage.warmup_steps)
    total_steps = max(1, total_steps)
    cosine_span = max(1, total_steps - warmup_steps)
    restart_cycle = max(1, cosine_span // 3)
    # Cosine decays to min_lr_ratio * max_lr, not to zero.
    # Research: "min LR is typically 10% of max, not zero."
    min_lr_ratio = 0.1

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        warm_step = max(0, step - warmup_steps)
        progress = min(1.0, max(0.0, warm_step / cosine_span))
        if stage.schedule == "cosine":
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        if stage.schedule == "cosine-warm-restart":
            cycle_progress = (warm_step % restart_cycle) / float(restart_cycle)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _categorize_parameter_group(name: str) -> str:
    lowered = name.lower()
    if "lora_" in lowered:
        return "dora"
    if "encoder." in lowered:
        return "encoder"
    if "deformable_attention" in lowered:
        return "encoder_adapter"
    if "decoder_blocks" in lowered or "decoder_norm" in lowered:
        return "decoder"
    if "lm_head" in lowered:
        return "lm_head"
    if "contour_head" in lowered:
        return "contour_head"
    return "other"


def _compute_per_group_grad_norms(model) -> Dict[str, float]:
    accum: Dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        group = _categorize_parameter_group(name)
        value = float(parameter.grad.detach().float().norm(2).item())
        accum[group] = accum.get(group, 0.0) + (value * value)
    return {group: math.sqrt(value) for group, value in accum.items()}


def _detect_grad_anomalies(
    grad_norms: Dict[str, float],
    running_average: Dict[str, float],
    *,
    threshold_multiplier: float = 10.0,
) -> List[str]:
    alerts: List[str] = []
    for group, norm in grad_norms.items():
        baseline = running_average.get(group)
        if baseline is not None and baseline > 0 and norm > (threshold_multiplier * baseline):
            alerts.append(
                f"grad_norm_spike[{group}]: current={norm:.4f}, baseline={baseline:.4f}, ratio={norm / baseline:.2f}"
            )
        if baseline is None:
            running_average[group] = norm
        else:
            running_average[group] = 0.95 * baseline + 0.05 * norm
    return alerts


def _should_run_diagnostics(optimizer_step: int, cadence: int) -> bool:
    """Return True when heavy diagnostic syncs should fire.

    Convention: fires on optimizer_step == 1 (the very first step) and then
    every ``cadence`` steps thereafter (1-indexed).  With ``cadence=1`` every
    step is a diagnostic step, matching the legacy behavior.

    Args:
        optimizer_step: 1-indexed count of completed optimizer steps within a
            stage (i.e. ``global_step`` incremented *before* this call).
        cadence: how many optimizer steps between full-diagnostic runs.
    """
    if cadence <= 1:
        return True
    # Fire on step 1 and every cadence-th step after.
    return optimizer_step == 1 or (optimizer_step % cadence == 0)


def _maybe_compile_decoder_and_bridge(model, *, enabled: bool):
    """Apply torch.compile to the Stage-B decoder + positional_bridge submodules.

    Deliberately skips the RADIO encoder: trust_remote_code with pinned
    transformers_version=4.51.3 is a known compile hazard (graph breaks +
    cache misses on every step). Decoder and bridge are stable custom code
    that benefit cleanly from compilation.

    First 50-100 opt-steps after compile are slower (cache warm-up); profile
    with --profile-step-timing on a sufficiently long run to measure the
    steady-state win.

    Args:
        model: Stage-B model with .decoder, .positional_bridge attrs (and
            optionally .encoder which is left untouched).
        enabled: When False, returns model unchanged. When True, replaces
            the .decoder and .positional_bridge attrs with torch.compile()'d
            versions.

    Returns:
        The same model object (mutated in place when enabled).
    """
    if not enabled:
        return model
    import torch
    if hasattr(model, "decoder"):
        model.decoder = torch.compile(model.decoder, dynamic=False, fullgraph=False)
    if hasattr(model, "positional_bridge"):
        model.positional_bridge = torch.compile(model.positional_bridge, dynamic=False, fullgraph=False)
    return model


def _prepare_model_for_dora(model, dora_config: Dict[str, object]):
    new_module_keywords = (
        "token_embedding",
        "lm_head",
        "decoder_norm",
        "contour_head",
        "deformable_attention",
        "positional_bridge",
    )

    try:
        from peft import LoraConfig, TaskType, get_peft_model
        import torch.nn as torch_nn
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "peft is required for DoRA training but is not installed. Install with: pip install peft"
        ) from exc

    configured_targets = list(dora_config.get("target_modules", []))
    linear_targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch_nn.Linear):
            continue
        if any(marker in name for marker in new_module_keywords):
            continue
        linear_targets.append(name)
    if configured_targets:
        linear_targets = [
            name
            for name in linear_targets
            if any(name.endswith(target) or f".{target}" in name for target in configured_targets)
            or name.startswith("encoder.backbone")
        ]
    if not linear_targets:
        raise RuntimeError(
            "DoRA target-module matching failed; no linear layers matched configured targets."
        )

    peft_config = LoraConfig(
        r=int(dora_config["rank"]),
        lora_alpha=int(dora_config["alpha"]),
        lora_dropout=float(dora_config["dropout"]),
        target_modules=linear_targets,
        task_type=TaskType.FEATURE_EXTRACTION,
        use_dora=True,
        bias="none",
        modules_to_save=list(new_module_keywords),
    )
    try:
        model = get_peft_model(model, peft_config)
    except Exception as exc:
        raise RuntimeError(f"Failed to apply DoRA adapters: {exc}") from exc
    dora_applied = True

    # DoRA zero-row fix: if a base weight has an all-zero row, DoRA initialises
    # the magnitude to 0.  The forward-pass recomputes weight_norm each call;
    # if that row stays zero (base + lora_B@lora_A is zero), weight_norm=0 and
    # magnitude/weight_norm = 0/0 = NaN.  Fix by perturbing zero base rows with
    # a tiny constant so weight_norm > 0, and resetting the stored magnitude.
    #
    # In PEFT 0.19.1, lora_magnitude_vector[adapter] is a DoraLinearLayer module;
    # the actual nn.Parameter is at DoraLinearLayer.weight (set in update_layer as
    # `self.weight = nn.Parameter(weight_norm, ...)`).  Writing to _mag_param.data
    # would hit an AttributeError because DoraLinearLayer has no .data attribute.
    #
    # NOTE: this writes through to the pretrained base weight tensor.  Affected rows
    # are zero-init by construction (e.g., blocks.0.attn.proj has one such row in
    # C-RADIOv4-H), so the model's forward output at scale is unchanged, but saved
    # checkpoints will differ from the upstream RADIO weights by 1e-6 on those rows.
    import torch as _torch
    from peft.tuners.lora import LoraLayer as _LoraLayer
    from peft.tuners.lora.dora import DoraLinearLayer as _DoraLinearLayer
    _zero_row_modules_patched = 0
    for _name, _module in model.named_modules():
        if not (isinstance(_module, _LoraLayer) and hasattr(_module, "lora_magnitude_vector")):
            continue
        _base_w = _module.base_layer.weight.data  # shape (out, in)
        _row_norms = _torch.linalg.norm(_base_w, dim=1)  # (out,)
        _zero_rows = _row_norms == 0
        if not _zero_rows.any():
            continue
        # Perturb zero rows with a tiny constant so weight_norm > 0
        _base_w[_zero_rows] = 1e-6
        # Re-derive magnitude once from the corrected base weight; it doesn't depend on the adapter.
        _new_norms = _torch.linalg.norm(_base_w, dim=1)
        for _adapter_name, _mag_param in _module.lora_magnitude_vector.items():
            if not isinstance(_mag_param, _DoraLinearLayer):
                raise RuntimeError(
                    f"DoRA NaN fix: expected DoraLinearLayer at "
                    f"{_name}.lora_magnitude_vector[{_adapter_name!r}], "
                    f"got {type(_mag_param).__name__}. "
                    "Update the fix if the PEFT version changed."
                )
            _mag_param.weight.data[_zero_rows] = _new_norms[_zero_rows]
        _zero_row_modules_patched += 1
    if _zero_row_modules_patched > 0:
        print(f"DoRA NaN fix: patched {_zero_row_modules_patched} module(s) with zero base-weight rows")
    model._dora_zero_row_fixes_applied = _zero_row_modules_patched

    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name or any(marker in name for marker in new_module_keywords):
            parameter.requires_grad = True
    if not any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad = True

    return model, dora_applied


def _run_validation(
    model,
    stage: StageTrainingConfig,
    val_loader,
    device,
    bf16_enabled: bool,
    validation_batches: int,
    vocab_size: int,
    channels_last: bool = False,
) -> Optional[Dict[str, float]]:
    """Run validation over ``validation_batches`` batches from ``val_loader``.

    Args:
        model: The model to evaluate.
        stage: StageTrainingConfig (used for label_smoothing and contour_loss_weight).
        val_loader: A DataLoader yielding batched dicts from StageBDataset with
            split="val".  Iterated fresh each call (iter() called internally).
        device: torch.device to move tensors onto.
        bf16_enabled: Whether to enable bfloat16 autocast.
        validation_batches: Maximum number of batches to evaluate.
        vocab_size: Vocabulary size for cross-entropy logits reshape.
        channels_last: When True, move images with channels_last memory format.

    Returns:
        Dict with ``val_loss`` and ``val_contour_loss`` (mean over evaluated
        batches), or None if the DataLoader yields no batches.
    """
    import torch
    import torch.nn.functional as F

    losses: List[float] = []
    contour_losses: List[float] = []
    model.eval()
    with torch.no_grad():
        val_iter = iter(val_loader)
        for _ in range(validation_batches):
            try:
                batch_dict = next(val_iter)
            except StopIteration:
                break
            images = batch_dict["images"]
            decoder_inputs = batch_dict["decoder_inputs"]
            labels = batch_dict["labels"]
            contour_targets = batch_dict["contour_targets"]
            if channels_last:
                images = images.to(device, non_blocking=True, memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            decoder_inputs = decoder_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            contour_targets = contour_targets.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=bf16_enabled,
            ):
                outputs = model(pixel_values=images, input_ids=decoder_inputs, return_aux=True)
                token_loss = F.cross_entropy(
                    outputs["logits"].reshape(-1, vocab_size),
                    labels.reshape(-1),
                    ignore_index=-100,
                    label_smoothing=stage.label_smoothing,
                )
                contour_loss = F.cross_entropy(outputs["contour_logits"], contour_targets)
                total_loss = token_loss + (stage.contour_loss_weight * contour_loss)
            losses.append(float(total_loss.item()))
            contour_losses.append(float(contour_loss.item()))
    model.train()
    if not losses:
        return None
    return {
        "val_loss": float(sum(losses) / len(losses)),
        "val_contour_loss": float(sum(contour_losses) / len(contour_losses)),
    }


def _save_checkpoint(
    checkpoint_dir: Path,
    model,
    optimizer,
    scheduler=None,
    *,
    stage_name: str,
    global_step: int,
    stage_step: Optional[int] = None,
    stage_steps_total: Optional[int] = None,
    stage_b_config: Optional[Dict[str, object]] = None,
    name_suffix: Optional[str] = None,
) -> Path:
    import torch

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if name_suffix is not None:
        # Stable filename for best/final/etc. (overwrites prior).
        checkpoint_path = checkpoint_dir / f"{stage_name}_{name_suffix}.pt"
    else:
        checkpoint_path = checkpoint_dir / f"{stage_name}_step_{global_step:07d}.pt"
    payload = {
        "stage_name": stage_name,
        "global_step": global_step,
        "stage_step": stage_step,
        "stage_steps_total": stage_steps_total,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "stage_b_config": stage_b_config,
    }
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, checkpoint_path)
    return checkpoint_path


class _CpuPhase:
    """Inner context manager for CPU phase timing. See _StepTimer."""
    __slots__ = ("_parent", "_name", "_t0")

    def __init__(self, parent: "_StepTimer", name: str) -> None:
        self._parent = parent
        self._name = name
        self._t0 = 0.0

    def __enter__(self):
        import time as _time
        self._t0 = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        import time as _time
        dt_ms = (_time.perf_counter() - self._t0) * 1000.0
        self._parent._cpu_ms[self._name] = self._parent._cpu_ms.get(self._name, 0.0) + dt_ms
        return False


class _GpuPhase:
    """Inner context manager for GPU phase timing via cuda.Event. See _StepTimer."""
    __slots__ = ("_parent", "_name", "_start", "_end")

    def __init__(self, parent: "_StepTimer", name: str) -> None:
        self._parent = parent
        self._name = name
        self._start = None
        self._end = None

    def __enter__(self):
        import torch as _torch
        self._start = _torch.cuda.Event(enable_timing=True)
        self._end = _torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._end.record()
        self._parent._gpu_events.append((self._name, self._start, self._end))
        return False


class _StepTimer:
    """Per-optimizer-step phase timing for --profile-step-timing.

    Buckets:
      cpu_*  measured with time.perf_counter() (sample/encode/augment/h2d/log_io)
      gpu_*  measured with torch.cuda.Event (forward/backward/grad_diag/optimizer)

    Per-optimizer-step rollup: micro-batch buckets accumulate; flush() resolves
    cuda.Events with a single cuda.synchronize(), writes one JSONL row, and the
    caller resets via reset_step() at the start of each accumulation window.

    All methods are no-ops when ``enabled`` is False, so wrapping the loop has
    near-zero overhead in normal runs.
    """

    def __init__(self, enabled: bool, output_path: Optional[Path] = None) -> None:
        self.enabled = bool(enabled)
        self._fh = None
        if self.enabled and output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = output_path.open("w", buffering=1, encoding="utf-8")
        self._cpu_ms: Dict[str, float] = {}
        self._gpu_events: List[Tuple[str, object, object]] = []
        self._micro_batch_count = 0
        self._step_wall_start: Optional[float] = None

    def reset_step(self) -> None:
        if not self.enabled:
            return
        import time as _time
        self._cpu_ms.clear()
        self._gpu_events.clear()
        self._micro_batch_count = 0
        self._step_wall_start = _time.perf_counter()

    def cpu(self, name: str):
        return _CpuPhase(self, name) if self.enabled else nullcontext()

    def gpu(self, name: str):
        return _GpuPhase(self, name) if self.enabled else nullcontext()

    def micro_batch_done(self) -> None:
        if self.enabled:
            self._micro_batch_count += 1

    def flush(
        self,
        *,
        global_step: int,
        stage_name: str,
        validation_ms: Optional[float] = None,
        checkpoint_ms: Optional[float] = None,
    ) -> None:
        if not self.enabled or self._fh is None:
            return
        import time as _time
        import torch as _torch
        wall_total_ms = (
            (_time.perf_counter() - self._step_wall_start) * 1000.0
            if self._step_wall_start is not None
            else 0.0
        )
        gpu_ms: Dict[str, float] = {}
        if self._gpu_events:
            _torch.cuda.synchronize()
            for name, start_evt, end_evt in self._gpu_events:
                gpu_ms[name] = gpu_ms.get(name, 0.0) + start_evt.elapsed_time(end_evt)
        record: Dict[str, object] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "global_step": global_step,
            "stage_name": stage_name,
            "micro_batches": self._micro_batch_count,
            "wall_total_ms": round(wall_total_ms, 2),
            "validation_ms": (round(validation_ms, 2) if validation_ms is not None else None),
            "checkpoint_ms": (round(checkpoint_ms, 2) if checkpoint_ms is not None else None),
        }
        for k, v in self._cpu_ms.items():
            record[f"cpu_{k}_ms"] = round(v, 2)
        for k, v in gpu_ms.items():
            record[f"gpu_{k}_ms"] = round(v, 2)
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def _expand_lora_rank_tensors_for_resume(
    state_dict: Dict[str, object],
    model_state: Dict[str, object],
) -> Tuple[Dict[str, object], List[str]]:
    import torch

    expanded_state = dict(state_dict)
    notes: List[str] = []
    for key, source_value in list(expanded_state.items()):
        target_value = model_state.get(key)
        if target_value is None:
            continue
        if not isinstance(source_value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            continue
        source_shape = tuple(int(dim) for dim in source_value.shape)
        target_shape = tuple(int(dim) for dim in target_value.shape)
        if source_shape == target_shape:
            continue

        if "lora_A" in key and len(source_shape) == 2 and len(target_shape) == 2:
            same_input = source_shape[1] == target_shape[1]
            rank_expand = source_shape[0] < target_shape[0]
            if same_input and rank_expand:
                expanded = target_value.detach().clone()
                expanded[: source_shape[0], :] = source_value.to(device=expanded.device, dtype=expanded.dtype)
                expanded_state[key] = expanded
                notes.append(f"{key}: expanded lora_A rank {source_shape[0]} -> {target_shape[0]}")
                continue

        if "lora_B" in key and len(source_shape) == 2 and len(target_shape) == 2:
            same_output = source_shape[0] == target_shape[0]
            rank_expand = source_shape[1] < target_shape[1]
            if same_output and rank_expand:
                expanded = target_value.detach().clone()
                expanded[:, : source_shape[1]] = source_value.to(device=expanded.device, dtype=expanded.dtype)
                expanded_state[key] = expanded
                notes.append(f"{key}: expanded lora_B rank {source_shape[1]} -> {target_shape[1]}")
                continue
    return expanded_state, notes


def _extend_vocab_tensors_for_resume(
    state_dict: Dict[str, object],
    model_state: Dict[str, object],
    init_seed: int = 0,
) -> Tuple[Dict[str, object], List[str]]:
    """Pad vocab-shaped tensors with new rows when checkpoint vocab < target vocab.

    Handles both plain and PEFT/DoRA-wrapped names.  Any key in the live model that contains
    ``"token_embedding"`` or ``"lm_head"`` as a substring is a candidate.  Examples:

    * Plain:        ``token_embedding.weight``, ``lm_head.weight``, ``lm_head.bias``
    * PEFT-wrapped: ``base_model.model.token_embedding.original_module.weight``,
                    ``base_model.model.token_embedding.modules_to_save.default.weight``,
                    ``base_model.model.lm_head.original_module.weight``, … (6 keys total)

    Padding init: weight rows = mean(existing rows) + N(0, 0.01); bias entries = 0.
    Caller is responsible for weight tying — if ``lm_head.weight is token_embedding.weight``
    in the live model, ``model.load_state_dict`` will preserve the tie.
    """
    import torch

    expanded_state = dict(state_dict)
    notes: List[str] = []
    g = torch.Generator()
    g.manual_seed(int(init_seed))
    # Track already-extended tensors by source object identity to handle weight tying.
    # When lm_head.weight is the same tensor object as token_embedding.weight, reuse
    # the already-computed extended result so tied weights remain identical after padding.
    extended_by_id: Dict[int, object] = {}

    _VOCAB_SUBSTRINGS = ("token_embedding", "lm_head")

    # Iterate ALL keys in the live model that look like vocab-shaped layers.
    # The checkpoint is expected to have the exact same key (PEFT wraps both consistently).
    for model_key in model_state:
        if not any(sub in model_key for sub in _VOCAB_SUBSTRINGS):
            continue
        # The checkpoint must contain an identically named key.
        if model_key not in state_dict:
            continue

        ckpt_key = model_key
        source_value = state_dict[ckpt_key]
        target_value = model_state[model_key]
        if not isinstance(source_value, torch.Tensor) or not isinstance(target_value, torch.Tensor):
            continue
        source_shape = tuple(int(d) for d in source_value.shape)
        target_shape = tuple(int(d) for d in target_value.shape)
        if source_shape == target_shape:
            continue

        # Validate diff: target = source with N extra rows at the end (axis 0), every other axis equal.
        if len(source_shape) != len(target_shape):
            raise ValueError(f"only +N rows at end of axis 0 supported; got rank diff for {ckpt_key}")
        if any(s != t for s, t in zip(source_shape[1:], target_shape[1:])):
            raise ValueError(
                f"only +N rows at end of axis 0 supported; non-axis-0 dims differ for {ckpt_key}: {source_shape} -> {target_shape}"
            )
        if target_shape[0] <= source_shape[0]:
            raise ValueError(
                f"only +N rows at end of axis 0 supported; target rows not greater than source for {ckpt_key}"
            )
        n_new = target_shape[0] - source_shape[0]

        # Reuse previously computed extension if this is a tied tensor (same object).
        src_id = id(source_value)
        if src_id in extended_by_id:
            expanded_state[ckpt_key] = extended_by_id[src_id]
            notes.append(f"{ckpt_key}: extended vocab dim 0 from {source_shape[0]} to {target_shape[0]} (tied)")
            continue

        if ckpt_key.endswith("bias"):
            new_rows = torch.zeros(n_new, dtype=source_value.dtype)
        else:
            mean_row = source_value.mean(dim=0, keepdim=True)
            noise = torch.randn(n_new, source_shape[1], generator=g, dtype=source_value.dtype) * 0.01
            new_rows = mean_row.expand(n_new, source_shape[1]) + noise

        extended = torch.cat([source_value, new_rows.to(device=source_value.device, dtype=source_value.dtype)], dim=0)
        expanded_state[ckpt_key] = extended
        extended_by_id[src_id] = extended
        notes.append(f"{ckpt_key}: extended vocab dim 0 from {source_shape[0]} to {target_shape[0]}")

    return expanded_state, notes


def _apply_cuda_perf_toggles() -> None:
    """Enable cuDNN auto-tuner and TF32 matmuls when CUDA is available.

    Static input shape (250x2500) makes cudnn.benchmark a free 2-5% win.
    set_float32_matmul_precision('high') routes any fp32 matmul fallback
    (e.g. CE-loss reduce) through TF32; bf16 forward path is unaffected.
    """
    import torch
    if not torch.cuda.is_available():
        return
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def run_execute_mode(
    stages: Sequence[StageTrainingConfig],
    grouped_entries: Dict[Tuple[str, str], List[Dict[str, object]]],
    project_root: Path,
    image_height: int,
    image_width: int,
    max_steps_per_stage: Optional[int],
    seed: int,
    step_log_path: Optional[Path] = None,
    checkpoint_dir: Optional[Path] = None,
    validation_batches: int = 2,
    resume_checkpoint: Optional[Path] = None,
    start_stage: Optional[str] = None,
    stage_b_backbone: str = "davit_base.msft_in1k",
    stage_b_decoder_dim: int = 768,
    stage_b_decoder_layers: int = 8,
    stage_b_decoder_heads: int = 12,
    stage_b_dora_rank: Optional[int] = None,
    stage_b_encoder: Optional[str] = None,
    keep_last_checkpoints: int = 0,
    profile_step_timing: bool = False,
    profile_output_path: Optional[Path] = None,
    stage_b_radio_pool_to_stride32: bool = False,
    diag_cadence: int = 25,
    channels_last: bool = False,
    torch_compile: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 4,
) -> Dict[str, object]:
    import torch
    import torch.nn.functional as F

    if max_steps_per_stage is not None and max_steps_per_stage <= 0:
        raise ValueError("max_steps_per_stage must be positive when provided")

    rng = random.Random(seed)
    vocab = build_default_vocabulary()
    # Cross-stage encoder consistency check: all stages must agree on encoder choice.
    # The encoder is a fixed architectural decision (RADIO vs DaViT) that determines
    # model structure -- mixing them across stages would silently use stage[0]'s encoder.
    if len(stages) > 1 and any(s.stage_b_encoder != stages[0].stage_b_encoder for s in stages[1:]):
        raise ValueError(
            f"All stages must use the same stage_b_encoder. "
            f"Got: {[s.stage_b_encoder for s in stages]}"
        )

    # encoder: YAML value from first stage wins; explicit CLI arg overrides.
    resolved_encoder = stage_b_encoder if stage_b_encoder else (
        stages[0].stage_b_encoder if stages else "davit"
    )
    factory_cfg = ModelFactoryConfig(
        stage_b_vocab_size=vocab.size,
        stage_b_backbone=stage_b_backbone,
        stage_b_decoder_dim=stage_b_decoder_dim,
        stage_b_decoder_layers=stage_b_decoder_layers,
        stage_b_decoder_heads=stage_b_decoder_heads,
        stage_b_encoder=resolved_encoder,
        stage_b_radio_pool_to_stride32=bool(stage_b_radio_pool_to_stride32),
    )
    resume_payload = None
    resume_factory_cfg = None
    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        resume_payload = torch.load(resume_checkpoint, map_location="cpu")
        resume_factory_cfg = model_factory_config_from_checkpoint_payload(
            resume_payload,
            vocab_size=vocab.size,
            fallback=factory_cfg,
        )
        factory_cfg = resume_factory_cfg
        if isinstance(resume_payload, dict) and isinstance(resume_payload.get("stage_b_config"), dict):
            print(
                "[train] loaded Stage-B architecture settings from checkpoint metadata.",
                file=sys.stderr,
            )
    if stage_b_dora_rank is not None:
        requested_rank = max(1, int(stage_b_dora_rank))
        factory_cfg = dataclasses.replace(factory_cfg, stage_b_dora_rank=requested_rank)
        print(
            f"[train] overriding DoRA rank with CLI value: {requested_rank}",
            file=sys.stderr,
        )
    components = build_stage_b_components(factory_cfg)
    stage_b_config_dict = dataclasses.asdict(components["stage_b_config"])
    # Persist encoder type explicitly (RadioStageBConfig vs StageBModelConfig don't carry
    # this as a field - it's encoded in the dataclass type, which asdict() drops).
    # Without this, model_factory_config_from_checkpoint_payload has to infer from
    # tensor shapes; explicit metadata is cleaner.
    stage_b_config_dict["encoder"] = factory_cfg.stage_b_encoder
    base_model = components["model"]
    model, dora_applied = _prepare_model_for_dora(base_model, components["dora_config"])

    use_cuda = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    if use_cuda:
        try:
            torch.cuda.get_device_properties(0)
        except Exception:
            use_cuda = False
    if use_cuda:
        _apply_cuda_perf_toggles()
    device = torch.device("cuda" if use_cuda else "cpu")
    if channels_last:
        # Move weights to channels_last memory format before any forward pass.
        # This lets cuDNN pick NHWC-optimised conv kernels (5-10% win on bf16 hardware).
        # Only the 4D conv layers benefit; the decoder linear stack is unaffected.
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    # NOTE: torch.compile is applied AFTER the resume-checkpoint load below.
    # Compiling first wraps decoder + positional_bridge in OptimizedModule, which
    # adds an `._orig_mod.` prefix to state_dict keys; load_state_dict then sees
    # an architecture mismatch and raises. See cu132 plan Phase 4.2 bug fix.
    model.train()

    resume_stage_name: Optional[str] = None
    resume_optimizer_state = None
    resume_scheduler_state = None
    global_step = 0
    if resume_checkpoint is not None:
        checkpoint_payload = resume_payload
        if checkpoint_payload is None:
            raise RuntimeError("Resume checkpoint payload failed to load.")
        model_state = checkpoint_payload.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError(f"Invalid checkpoint payload (missing model_state_dict): {resume_checkpoint}")
        checkpoint_rank = (
            int(resume_factory_cfg.stage_b_dora_rank)
            if resume_factory_cfg is not None
            else int(factory_cfg.stage_b_dora_rank)
        )
        target_rank = int(factory_cfg.stage_b_dora_rank)
        if target_rank < checkpoint_rank:
            raise ValueError(
                f"Cannot resume with smaller DoRA rank (checkpoint={checkpoint_rank}, requested={target_rank})."
            )
        if target_rank > checkpoint_rank:
            model_state, expansion_notes = _expand_lora_rank_tensors_for_resume(
                model_state,
                model.state_dict(),
            )
            if expansion_notes:
                print(
                    f"[train] expanded {len(expansion_notes)} LoRA tensors for rank change "
                    f"{checkpoint_rank} -> {target_rank}.",
                    file=sys.stderr,
                )
        # Vocab extension: pad embedding/lm_head when target vocab > checkpoint vocab.
        # Required when adding new tokens (e.g., <staff_idx_N>) at the end of the vocabulary.
        model_state, vocab_extension_notes = _extend_vocab_tensors_for_resume(
            model_state,
            model.state_dict(),
            init_seed=int(seed),
        )
        if vocab_extension_notes:
            print(
                f"[train] extended {len(vocab_extension_notes)} vocab tensors for resume.",
                file=sys.stderr,
            )
        load_result = model.load_state_dict(model_state, strict=False)
        missing_keys = set(load_result.missing_keys)
        unexpected_keys = set(load_result.unexpected_keys)
        allowed_ctc_keys = {
            "ctc_head.weight",
            "ctc_head.bias",
            "base_model.model.ctc_head.weight",
            "base_model.model.ctc_head.bias",
            "ctc_head.modules_to_save.default.weight",
            "ctc_head.modules_to_save.default.bias",
            "base_model.model.ctc_head.modules_to_save.default.weight",
            "base_model.model.ctc_head.modules_to_save.default.bias",
        }
        disallowed_missing = sorted(key for key in missing_keys if key not in allowed_ctc_keys)
        disallowed_unexpected = sorted(key for key in unexpected_keys if key not in allowed_ctc_keys)
        if disallowed_missing or disallowed_unexpected:
            raise RuntimeError(
                "Checkpoint architecture mismatch. "
                f"missing={disallowed_missing[:10]}, unexpected={disallowed_unexpected[:10]}"
            )
        if missing_keys or unexpected_keys:
            print(
                "[train] resume checkpoint is legacy format; ignored compatibility keys.",
                file=sys.stderr,
            )
        global_step = max(0, int(checkpoint_payload.get("global_step", 0)))
        resume_stage_name = str(checkpoint_payload.get("stage_name") or "")
        resume_optimizer_state = checkpoint_payload.get("optimizer_state_dict")
        resume_scheduler_state = checkpoint_payload.get("scheduler_state_dict")
        print(
            f"[train] resuming from checkpoint: {resume_checkpoint} "
            f"(global_step={global_step}, stage='{resume_stage_name}')",
            file=sys.stderr,
        )

    # Apply torch.compile AFTER the resume-checkpoint load so the wrap doesn't
    # change state-dict key prefixes during the load. (See cu132 plan Phase 4.2.)
    model = _maybe_compile_decoder_and_bridge(model, enabled=torch_compile)

    bf16_enabled = False
    if device.type == "cuda":
        try:
            bf16_enabled = bool(torch.cuda.is_bf16_supported())
        except Exception:
            bf16_enabled = False

    stage_runtime: List[Dict[str, int]] = []
    for stage in stages:
        steps_per_epoch = max(1, math.ceil(stage.effective_samples_per_epoch / stage.batch_size))
        planned_total_steps = max(1, steps_per_epoch * stage.epochs)
        stage_total_steps = (
            min(planned_total_steps, max_steps_per_stage)
            if max_steps_per_stage is not None
            else planned_total_steps
        )
        stage_runtime.append(
            {
                "steps_per_epoch": steps_per_epoch,
                "planned_total_steps": planned_total_steps,
                "stage_total_steps": stage_total_steps,
            }
        )

    requested_stage_index: Optional[int] = None
    requested_stage_label: Optional[str] = None
    if start_stage is not None and str(start_stage).strip():
        raw = str(start_stage).strip()
        if raw.isdigit():
            parsed_index = int(raw) - 1
            if parsed_index < 0 or parsed_index >= len(stages):
                raise ValueError(
                    f"start-stage index out of range: {raw}. Valid range is 1..{len(stages)}."
                )
            requested_stage_index = parsed_index
            requested_stage_label = stages[parsed_index].stage_name
        else:
            lowered = raw.lower()
            exact_matches = [idx for idx, stage in enumerate(stages) if stage.stage_name.lower() == lowered]
            if len(exact_matches) == 1:
                requested_stage_index = exact_matches[0]
            else:
                prefix_matches = [idx for idx, stage in enumerate(stages) if stage.stage_name.lower().startswith(lowered)]
                if len(prefix_matches) == 1:
                    requested_stage_index = prefix_matches[0]
                elif len(prefix_matches) > 1:
                    choices = ", ".join(stage.stage_name for stage in stages)
                    raise ValueError(
                        f"start-stage '{raw}' is ambiguous. Available stages: {choices}"
                    )
                else:
                    choices = ", ".join(stage.stage_name for stage in stages)
                    raise ValueError(
                        f"start-stage '{raw}' not found. Available stages: {choices}"
                    )
            if requested_stage_index is not None:
                requested_stage_label = stages[requested_stage_index].stage_name

    resume_stage_index = 0
    resume_stage_completed_steps = 0
    if resume_checkpoint is not None:
        remaining_steps = global_step
        resume_stage_index = len(stages)
        resume_stage_completed_steps = 0
        for stage_index, runtime in enumerate(stage_runtime):
            stage_total_steps = int(runtime["stage_total_steps"])
            if remaining_steps >= stage_total_steps:
                remaining_steps -= stage_total_steps
                continue
            resume_stage_index = stage_index
            resume_stage_completed_steps = remaining_steps
            break
        if resume_stage_index < len(stages):
            resolved_stage_name = stages[resume_stage_index].stage_name
            if resume_stage_name and resume_stage_name != resolved_stage_name:
                print(
                    "[train] warning: checkpoint stage name does not match computed stage boundary. "
                    f"checkpoint='{resume_stage_name}', computed='{resolved_stage_name}'",
                    file=sys.stderr,
                )

    if requested_stage_index is not None:
        baseline_step = sum(int(runtime["stage_total_steps"]) for runtime in stage_runtime[:requested_stage_index])
        if resume_checkpoint is not None:
            print(
                "[train] start-stage override enabled: forcing start from "
                f"'{stages[requested_stage_index].stage_name}' (index={requested_stage_index + 1}), "
                "ignoring checkpoint stage progress.",
                file=sys.stderr,
            )
        else:
            print(
                f"[train] starting from stage '{stages[requested_stage_index].stage_name}' "
                f"(index={requested_stage_index + 1}).",
                file=sys.stderr,
            )
        resume_stage_index = requested_stage_index
        resume_stage_completed_steps = 0
        global_step = baseline_step

    stage_metrics: List[Dict[str, object]] = []
    initial_global_step = global_step
    grad_running_average: Dict[str, float] = {}
    grad_alerts: List[Dict[str, object]] = []
    checkpoints_written: List[str] = []
    validation_events: List[Dict[str, object]] = []
    step_writer = None
    # In-memory buffer for JSONL records. Flushed to disk on cadence steps,
    # validation events, checkpoint events, and stage end (see usage below).
    _step_log_buffer: List[str] = []

    if step_log_path is not None:
        step_log_path.parent.mkdir(parents=True, exist_ok=True)
        step_log_mode = "a" if (resume_checkpoint is not None) else "w"
        step_writer = step_log_path.open(step_log_mode, encoding="utf-8")

    timer = _StepTimer(enabled=profile_step_timing, output_path=profile_output_path)

    try:
        for stage_index, stage in enumerate(stages):
            runtime = stage_runtime[stage_index]
            stage_steps_per_epoch = int(runtime["steps_per_epoch"])
            stage_planned_steps = int(runtime["planned_total_steps"])
            stage_total_steps = int(runtime["stage_total_steps"])
            stage_start_step = 1
            resumed_stage = False

            if (resume_checkpoint is not None) or (requested_stage_index is not None):
                if stage_index < resume_stage_index:
                    stage_metrics.append(
                        {
                            "stage_name": stage.stage_name,
                            "epochs": stage.epochs,
                            "steps_per_epoch": stage_steps_per_epoch,
                            "planned_total_steps": stage_planned_steps,
                            "steps_executed": 0,
                            "truncated_by_max_steps": max_steps_per_stage is not None and stage_planned_steps > stage_total_steps,
                            "mean_loss": None,
                            "min_loss": None,
                            "max_loss": None,
                            "final_loss": None,
                            "lr_dora": stage.lr_dora,
                            "lr_new_modules": stage.lr_new_modules,
                            "contour_loss_weight": stage.contour_loss_weight,
                            "weight_decay": stage.weight_decay,
                            "non_finite_events": 0,
                            "grad_alert_count": 0,
                            "skipped_by_resume": True,
                            "resume_start_step": stage_total_steps + 1,
                        }
                    )
                    continue
                if stage_index == resume_stage_index:
                    stage_start_step = max(1, int(resume_stage_completed_steps) + 1)
                    resumed_stage = stage_start_step > 1
                    if stage_start_step > stage_total_steps:
                        stage_metrics.append(
                            {
                                "stage_name": stage.stage_name,
                                "epochs": stage.epochs,
                                "steps_per_epoch": stage_steps_per_epoch,
                                "planned_total_steps": stage_planned_steps,
                                "steps_executed": 0,
                                "truncated_by_max_steps": max_steps_per_stage is not None and stage_planned_steps > stage_total_steps,
                                "mean_loss": None,
                                "min_loss": None,
                                "max_loss": None,
                                "final_loss": None,
                                "lr_dora": stage.lr_dora,
                                "lr_new_modules": stage.lr_new_modules,
                                "contour_loss_weight": stage.contour_loss_weight,
                                "weight_decay": stage.weight_decay,
                                "non_finite_events": 0,
                                "grad_alert_count": 0,
                                "skipped_by_resume": True,
                                "resume_start_step": stage_start_step,
                            }
                        )
                        continue

            optimizer = _build_optimizer(model, stage)
            optimizer_steps = max(1, stage_total_steps // stage.grad_accumulation_steps)
            scheduler = _build_scheduler(optimizer, stage, total_steps=optimizer_steps)
            if resumed_stage:
                if resume_optimizer_state is not None:
                    try:
                        optimizer.load_state_dict(resume_optimizer_state)
                    except Exception as exc:
                        print(
                            f"[train] warning: failed to restore optimizer state; continuing with fresh optimizer ({exc})",
                            file=sys.stderr,
                        )
                if resume_scheduler_state is not None:
                    try:
                        scheduler.load_state_dict(resume_scheduler_state)
                    except Exception as exc:
                        print(
                            f"[train] warning: failed to restore scheduler state; using step-aligned scheduler ({exc})",
                            file=sys.stderr,
                        )
                        scheduler.step(max(0, stage_start_step // stage.grad_accumulation_steps - 1))
                else:
                    scheduler.step(max(0, stage_start_step // stage.grad_accumulation_steps - 1))
            losses: List[float] = []
            non_finite_events = 0
            stage_grad_alerts = 0
            best_val_loss: Optional[float] = None
            best_val_step: Optional[int] = None

            # Build a DataLoader for this stage's training set.
            # Each epoch is defined by effective_samples_per_epoch samples drawn
            # via the WeightedRandomSampler (which preserves dataset_mix ratios).
            # persistent_workers=True avoids per-epoch worker respawn overhead.
            # pin_memory=True enables async H2D transfers via non_blocking=True.
            _stage_ds = StageBDataset(
                stage,
                grouped_entries,
                project_root=project_root,
                image_height=image_height,
                image_width=image_width,
                max_sequence_length=stage.max_sequence_length,
                augment=True,
                rng_seed=seed,
            )
            # Compute total micro-batch draws needed for the full stage.
            # stage_total_steps is OPT-steps; each OPT-step consumes
            # grad_accumulation_steps micro-batches.  Sizing the sampler to
            # cover the full count guarantees the iterator never exhausts
            # mid-accumulation-window (the StopIteration handler below is
            # purely defensive against bugs / checkpoint restarts).
            _stage_total_train_samples = (
                stage_total_steps * stage.batch_size * stage.grad_accumulation_steps
            )
            _train_sampler = build_stage_b_sampler(
                stage, _stage_ds,
                total_samples=_stage_total_train_samples,
                seed=seed,
            )
            _pin_memory = device.type == "cuda"
            _effective_prefetch = prefetch_factor if num_workers > 0 else None
            _train_loader = torch.utils.data.DataLoader(
                _stage_ds,
                batch_size=stage.batch_size,
                sampler=_train_sampler,
                num_workers=num_workers,
                pin_memory=_pin_memory,
                persistent_workers=(num_workers > 0),
                prefetch_factor=_effective_prefetch,
                collate_fn=StageBDataset.collate_fn,
                worker_init_fn=stage_b_worker_init_fn,
            )
            _train_iter = iter(_train_loader)
            # vocab size is constant; grab from the dataset's vocab object.
            vocab_size = _stage_ds._vocab.size

            # Build a val-side DataLoader for _run_validation.
            # Uses the same dataset_mix ratios as training but fetches from
            # split="val" entries.  augment=False: no augmentation during eval.
            # The sampler is sized to cover validation_batches * batch_size draws
            # (with replacement) so the iterator never exhausts during a single
            # validation cycle.  A fresh iter() is called inside _run_validation.
            _val_dataset = StageBDataset(
                stage,
                grouped_entries,
                split="val",
                project_root=project_root,
                image_height=image_height,
                image_width=image_width,
                max_sequence_length=stage.max_sequence_length,
                augment=False,
                rng_seed=seed,
            )
            _val_total_samples = validation_batches * stage.batch_size
            _val_sampler = build_stage_b_sampler(
                stage,
                _val_dataset,
                total_samples=_val_total_samples,
                seed=seed,
                split_override="val",
            )
            _val_loader = torch.utils.data.DataLoader(
                _val_dataset,
                batch_size=stage.batch_size,
                sampler=_val_sampler,
                num_workers=num_workers,
                pin_memory=_pin_memory,
                persistent_workers=(num_workers > 0),
                prefetch_factor=_effective_prefetch,
                collate_fn=StageBDataset.collate_fn,
                worker_init_fn=stage_b_worker_init_fn,
            )

            timer.reset_step()

            for stage_step in range(stage_start_step, stage_total_steps + 1):
                if (stage_step - 1) % stage.grad_accumulation_steps == 0:
                    timer.reset_step()
                with timer.cpu("sample"):
                    try:
                        _batch_dict = next(_train_iter)
                    except StopIteration:
                        # Defensive rebuild: iterator exhausted (correct
                        # total_samples sizing should prevent this, but guard
                        # against checkpoint restarts / sampler edge cases).
                        _mid_window = (stage_step - 1) % stage.grad_accumulation_steps != 0
                        if _mid_window:
                            # Partial accumulation window: accumulated gradients
                            # represent fewer micro-batches than expected.
                            # Discard them to avoid a mis-scaled optimizer step.
                            optimizer.zero_grad(set_to_none=True)
                            print(
                                "[train] DataLoader iterator exhausted mid-accumulation"
                                " window — gradient discarded",
                                flush=True,
                            )
                        _train_iter = iter(_train_loader)
                        _batch_dict = next(_train_iter)
                if _batch_dict is None:
                    break
                epoch_index = ((stage_step - 1) // stage_steps_per_epoch) + 1
                epoch_step = ((stage_step - 1) % stage_steps_per_epoch) + 1
                with timer.cpu("h2d"):
                    if channels_last:
                        images = _batch_dict["images"].to(device, non_blocking=True, memory_format=torch.channels_last)
                    else:
                        images = _batch_dict["images"].to(device, non_blocking=True)
                    decoder_inputs = _batch_dict["decoder_inputs"].to(device, non_blocking=True)
                    labels = _batch_dict["labels"].to(device, non_blocking=True)
                    contour_targets = _batch_dict["contour_targets"].to(device, non_blocking=True)

                accum_steps = stage.grad_accumulation_steps
                is_accum_step = (stage_step % accum_steps) == 0 or stage_step == stage_total_steps
                if (stage_step - 1) % accum_steps == 0:
                    optimizer.zero_grad(set_to_none=True)
                    accum_window_corrupted = False
                with timer.gpu("forward"):
                    with torch.autocast(
                        device_type=device.type,
                        dtype=torch.bfloat16,
                        enabled=bf16_enabled,
                    ):
                        outputs = model(pixel_values=images, input_ids=decoder_inputs, return_aux=True)
                        token_loss = F.cross_entropy(
                            outputs["logits"].reshape(-1, vocab_size),
                            labels.reshape(-1),
                            ignore_index=-100,
                            label_smoothing=stage.label_smoothing,
                        )
                        contour_loss = F.cross_entropy(outputs["contour_logits"], contour_targets)
                        loss = token_loss + (stage.contour_loss_weight * contour_loss)
                        if accum_steps > 1:
                            loss = loss / accum_steps

                non_finite_loss = not bool(torch.isfinite(loss).item())
                if non_finite_loss:
                    non_finite_events += 1
                    optimizer.zero_grad(set_to_none=True)
                    accum_window_corrupted = True
                    timer.micro_batch_done()
                    continue

                if accum_window_corrupted:
                    # Prior micro-batch in this window had non-finite loss;
                    # skip remaining backward passes to avoid wrong gradient scale.
                    if is_accum_step:
                        optimizer.zero_grad(set_to_none=True)
                    timer.micro_batch_done()
                    continue

                with timer.gpu("backward"):
                    loss.backward()

                if not is_accum_step:
                    # Single CPU sync per micro-batch; cache the python float so any
                    # downstream consumer reads from CPU memory rather than re-syncing.
                    _loss_scalar_cache = loss.detach().item() * accum_steps
                    losses.append(_loss_scalar_cache)
                    timer.micro_batch_done()
                    continue

                # next_optimizer_step is 1-indexed *after* this step completes.
                _next_optimizer_step = global_step - initial_global_step + 1
                _run_diag = _should_run_diagnostics(_next_optimizer_step, diag_cadence)

                with timer.gpu("grad_diagnostics"):
                    # clip_grad_norm_ must always run — it mutates the gradients.
                    grad_norm_value = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item())

                    if _run_diag:
                        # Heavy paths: per-group norms + full finite-grad scan.
                        grad_norms = _compute_per_group_grad_norms(model)
                        grad_alert_messages = _detect_grad_anomalies(grad_norms, grad_running_average)
                        if grad_alert_messages:
                            stage_grad_alerts += len(grad_alert_messages)
                            grad_alerts.append(
                                {
                                    "global_step": global_step + 1,
                                    "stage_name": stage.stage_name,
                                    "alerts": grad_alert_messages,
                                }
                            )

                        non_finite_grad = False
                        for parameter in model.parameters():
                            if parameter.grad is None:
                                continue
                            if not torch.isfinite(parameter.grad).all():
                                non_finite_grad = True
                                break
                    else:
                        # Non-diagnostic step: skip per-group norms and finite scan.
                        grad_norms = {}
                        grad_alert_messages = []
                        non_finite_grad = False

                if non_finite_grad:
                    non_finite_events += 1
                    optimizer.zero_grad(set_to_none=True)
                    timer.micro_batch_done()
                    continue

                with timer.gpu("optimizer"):
                    optimizer.step()
                    scheduler.step()
                # Single CPU sync per optimizer step; the python float is reused
                # below by the JSONL record without re-syncing.
                _loss_scalar_cache = loss.detach().item() * accum_steps
                losses.append(_loss_scalar_cache)
                global_step += 1

                lr_map = {group.get("group_name", f"group_{idx}"): group["lr"] for idx, group in enumerate(optimizer.param_groups)}
                validation_result = None
                _validation_ms: Optional[float] = None
                if global_step % stage.validate_every_steps == 0:
                    import time as _time
                    _val_t0 = _time.perf_counter()
                    validation_result = _run_validation(
                        model=model,
                        stage=stage,
                        val_loader=_val_loader,
                        device=device,
                        bf16_enabled=bf16_enabled,
                        validation_batches=validation_batches,
                        vocab_size=vocab_size,
                        channels_last=channels_last,
                    )
                    _validation_ms = (_time.perf_counter() - _val_t0) * 1000.0
                    if validation_result is not None:
                        validation_events.append(
                            {
                                "global_step": global_step,
                                "stage_name": stage.stage_name,
                                **validation_result,
                            }
                        )
                        # Save a stable _best.pt whenever val_loss improves.
                        if checkpoint_dir is not None:
                            current_val = validation_result.get("val_loss")
                            if isinstance(current_val, (int, float)):
                                current_val = float(current_val)
                                if best_val_loss is None or current_val < best_val_loss:
                                    best_val_loss = current_val
                                    best_val_step = global_step
                                    _save_checkpoint(
                                        checkpoint_dir=checkpoint_dir,
                                        model=model,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        stage_name=stage.stage_name,
                                        global_step=global_step,
                                        stage_step=stage_step,
                                        stage_steps_total=stage_total_steps,
                                        stage_b_config=stage_b_config_dict,
                                        name_suffix="best",
                                    )

                _checkpoint_ms: Optional[float] = None
                if checkpoint_dir is not None and global_step % stage.checkpoint_every_steps == 0:
                    import time as _time
                    _ckpt_t0 = _time.perf_counter()
                    checkpoint_path = _save_checkpoint(
                        checkpoint_dir=checkpoint_dir,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        stage_name=stage.stage_name,
                        global_step=global_step,
                        stage_step=stage_step,
                        stage_steps_total=stage_total_steps,
                        stage_b_config=stage_b_config_dict,
                    )
                    checkpoints_written.append(str(checkpoint_path))
                    _checkpoint_ms = (_time.perf_counter() - _ckpt_t0) * 1000.0
                    # Optional: keep only the last N step-numbered checkpoints to bound
                    # disk usage. _best.pt and _final.pt are stable filenames and are
                    # not matched by the f"{stage_name}_step_*.pt" glob, so they are
                    # never pruned by this rule.
                    if keep_last_checkpoints > 0:
                        periodic = sorted(
                            checkpoint_dir.glob(f"{stage.stage_name}_step_*.pt")
                        )
                        excess = len(periodic) - int(keep_last_checkpoints)
                        if excess > 0:
                            for old_path in periodic[:excess]:
                                try:
                                    old_path.unlink()
                                except OSError as exc:
                                    print(
                                        f"[train] warning: failed to prune old checkpoint {old_path}: {exc}",
                                        file=sys.stderr,
                                    )

                if step_writer is not None:
                    with timer.cpu("log_io"):
                        # _loss_scalar_cache is already a python float (synced once
                        # per opt-step above); reusing it costs nothing.
                        _loss_float = _loss_scalar_cache
                        record = {
                            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                            "global_step": global_step,
                            "stage_step": stage_step,
                            "stage_steps_total": stage_total_steps,
                            "epoch_index": epoch_index,
                            "epoch_step": epoch_step,
                            "epoch_steps_total": stage_steps_per_epoch,
                            "stage_name": stage.stage_name,
                            "loss": _loss_float,
                            # token_loss / contour_loss only synced on diag steps
                            # (they require .item() on still-live GPU tensors).
                            "token_loss": float(token_loss.item()) if _run_diag else None,
                            "contour_loss": float(contour_loss.item()) if _run_diag else None,
                            "lr_dora": float(lr_map.get("dora", stage.lr_dora)),
                            "lr_new_modules": float(lr_map.get("new_modules", stage.lr_new_modules)),
                            "grad_norm": grad_norm_value,
                            "grad_norm_groups": grad_norms,
                            "grad_alerts": grad_alert_messages,
                            "batch_size": int(images.shape[0]),
                            "max_sequence_length": stage.max_sequence_length,
                            "non_finite_loss": non_finite_loss,
                            "non_finite_grad": non_finite_grad,
                            "bf16_enabled": bf16_enabled,
                            "validation": validation_result,
                        }
                        _step_log_buffer.append(json.dumps(record) + "\n")
                        # Flush buffer to disk on: cadence step, validation event,
                        # checkpoint event, or when a validation result is present.
                        _should_flush_log = (
                            _run_diag
                            or validation_result is not None
                            or _checkpoint_ms is not None
                        )
                        if _should_flush_log and _step_log_buffer:
                            step_writer.writelines(_step_log_buffer)
                            step_writer.flush()
                            _step_log_buffer.clear()

                timer.micro_batch_done()
                timer.flush(
                    global_step=global_step,
                    stage_name=stage.stage_name,
                    validation_ms=_validation_ms,
                    checkpoint_ms=_checkpoint_ms,
                )

            # Save a final checkpoint at the end of the stage so the very last
            # optimizer steps (between the last periodic checkpoint and the loop
            # end) are not discarded. checkpoint_every_steps=1000 + a stage that
            # ends at e.g. step 4688 would otherwise lose the last 688 steps.
            if checkpoint_dir is not None and global_step > 0:
                final_path = _save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    stage_name=stage.stage_name,
                    global_step=global_step,
                    stage_step=stage_step,
                    stage_steps_total=stage_total_steps,
                    stage_b_config=stage_b_config_dict,
                    name_suffix="final",
                )
                checkpoints_written.append(str(final_path))

            # Flush any buffered JSONL records at stage end.
            if step_writer is not None and _step_log_buffer:
                step_writer.writelines(_step_log_buffer)
                step_writer.flush()
                _step_log_buffer.clear()

            stage_metrics.append(
                {
                    "stage_name": stage.stage_name,
                    "epochs": stage.epochs,
                    "steps_per_epoch": stage_steps_per_epoch,
                    "planned_total_steps": stage_planned_steps,
                    "steps_executed": len(losses),
                    "truncated_by_max_steps": max_steps_per_stage is not None and stage_planned_steps > stage_total_steps,
                    "mean_loss": (sum(losses) / len(losses)) if losses else None,
                    "min_loss": min(losses) if losses else None,
                    "max_loss": max(losses) if losses else None,
                    "final_loss": losses[-1] if losses else None,
                    "lr_dora": stage.lr_dora,
                    "lr_new_modules": stage.lr_new_modules,
                    "contour_loss_weight": stage.contour_loss_weight,
                    "weight_decay": stage.weight_decay,
                    "non_finite_events": non_finite_events,
                    "grad_alert_count": stage_grad_alerts,
                    "skipped_by_resume": False,
                    "resume_start_step": stage_start_step,
                }
            )
    finally:
        # Flush any remaining buffered records before closing.
        if step_writer is not None:
            if _step_log_buffer:
                step_writer.writelines(_step_log_buffer)
                _step_log_buffer.clear()
            step_writer.close()
        timer.close()

    return {
        "mode": "execute",
        "device": str(device),
        "bf16_enabled": bf16_enabled,
        "dora_applied": dora_applied,
        "stages": stage_metrics,
        "dora_target_modules": components["dora_target_modules"],
        "dora_config": components["dora_config"],
        "step_log_path": str(step_log_path) if step_log_path is not None else None,
        "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir is not None else None,
        "max_steps_per_stage": max_steps_per_stage,
        "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint is not None else None,
        "resumed_from_global_step": initial_global_step if resume_checkpoint is not None else None,
        "resume_stage_name": resume_stage_name if resume_checkpoint is not None else None,
        "start_stage": str(start_stage) if start_stage is not None else None,
        "start_stage_resolved": requested_stage_label,
        "checkpoints_written": checkpoints_written,
        "validation_events": validation_events,
        "grad_alerts": grad_alerts,
        "stage_b_config": stage_b_config_dict,
    }


def run_dry_mode(
    stages: Sequence[StageTrainingConfig],
    grouped_entries: Dict[Tuple[str, str], List[Dict[str, object]]],
) -> Dict[str, object]:
    return {
        "mode": "dry-run",
        "stages": [build_stage_plan(stage=stage, grouped_entries=grouped_entries) for stage in stages],
    }


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Run curriculum training stages for OMR Stage-B.")
    parser.add_argument(
        "--token-manifest",
        type=str,
        default=",".join([
            str(project_root / "src" / "data" / "manifests" / "token_manifest.jsonl"),
            str(project_root / "data" / "processed" / "synthetic" / "manifests" / "synthetic_token_manifest.jsonl"),
        ]),
        help="Comma-separated token manifest JSONL path(s).",
    )
    parser.add_argument(
        "--stage-configs",
        type=str,
        default="configs/train_stage1.yaml,configs/train_stage2.yaml,configs/train_stage3.yaml",
        help="Comma-separated stage config paths.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=project_root / "src" / "train" / "curriculum_summary.json",
        help="Path to write execution summary JSON.",
    )
    parser.add_argument(
        "--mode",
        choices=("dry-run", "execute"),
        default="dry-run",
        help="dry-run builds curriculum plans; execute runs stage training according to config budgets.",
    )
    parser.add_argument(
        "--max-steps-per-stage",
        type=int,
        default=None,
        help=(
            "Optional cap for execute mode debugging. "
            "When omitted, execute uses full stage budgets from epochs and effective_samples_per_epoch."
        ),
    )
    parser.add_argument(
        "--validation-batches",
        type=int,
        default=2,
        help="Validation mini-batches to run every validation cadence.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=project_root / "src" / "train" / "checkpoints",
        help="Directory used for periodic checkpoint writing in execute mode.",
    )
    parser.add_argument(
        "--step-log",
        type=Path,
        default=None,
        help="Optional JSONL telemetry output path for execute mode.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional checkpoint path (.pt) to resume execute mode from model/optimizer state.",
    )
    parser.add_argument(
        "--start-stage",
        type=str,
        default=None,
        help=(
            "Optional stage selector for execute mode. "
            "Accepts 1-based index (e.g. '2') or stage name (e.g. 'stage2-polyphonic')."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--allow-invalid-token-sequences",
        action="store_true",
        help="Skip strict grammar/vocabulary filtering of token manifest entries.",
    )
    parser.add_argument(
        "--allow-stale-merged-manifest",
        action="store_true",
        help="Allow using token_manifest_train.jsonl even if base/synthetic manifests are newer.",
    )
    parser.add_argument(
        "--stage-b-backbone",
        type=str,
        default="davit_base.msft_in1k",
        help="Stage-B timm backbone name.",
    )
    parser.add_argument(
        "--stage-b-encoder",
        type=str,
        default=None,
        help=(
            "Stage-B encoder type: 'davit' (default) or 'radio_h' (C-RADIOv4-H). "
            "Overrides the stage_b_encoder key in the stage YAML config."
        ),
    )
    parser.add_argument(
        "--stage-b-decoder-dim",
        type=int,
        default=768,
        help="Decoder hidden dimension.",
    )
    parser.add_argument(
        "--stage-b-decoder-layers",
        type=int,
        default=8,
        help="Decoder depth.",
    )
    parser.add_argument(
        "--stage-b-decoder-heads",
        type=int,
        default=12,
        help="Decoder attention heads.",
    )
    parser.add_argument(
        "--stage-b-dora-rank",
        type=int,
        default=None,
        help="Override DoRA rank (supports rank expansion when resuming from smaller-rank checkpoints).",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=250,
        help="Input image height for training batches.",
    )
    parser.add_argument(
        "--image-max-width",
        type=int,
        default=2500,
        help="Maximum preserved width (images are aspect-preserved and right-padded, max 3000).",
    )
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help=(
            "Bound disk by keeping only the most recent N step-numbered checkpoints. "
            "0 (default) = keep all. _best.pt and _final.pt are stable filenames and "
            "are never pruned."
        ),
    )
    parser.add_argument(
        "--profile-step-timing",
        action="store_true",
        help="Emit per-optimizer-step phase timings (CPU sample/encode/augment/h2d/log_io + GPU forward/backward/grad_diag/optimizer) to a JSONL.",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=None,
        help="Path for the per-step profile JSONL (default: logs/profile_step_timing.jsonl). Only used when --profile-step-timing is set.",
    )
    parser.add_argument(
        "--stage-b-radio-pool-to-stride32",
        action="store_true",
        help=(
            "When using stage_b_encoder=radio_h, apply 2x2 average pooling on the "
            "encoder output to emulate stride-32 grids (4x fewer spatial tokens). "
            "Cuts decoder cross-attention cost; trades some dense-feature precision."
        ),
    )
    parser.add_argument(
        "--diag-cadence",
        type=int,
        default=25,
        # Default of 25: Stage 1 ran 4688 steps with 0 non-finite events.
        # Cadence-25 reduces CPU↔GPU sync overhead by ~25× while still catching
        # the first non-finite within 25 optimizer steps.
        help=(
            "Run full gradient diagnostics (per-group norms + finite-grad scan) "
            "only every N optimizer steps (default: 25). "
            "Use --diag-cadence 1 to restore the original every-step behavior."
        ),
    )
    parser.add_argument(
        "--channels-last",
        action="store_true",
        default=False,
        help=(
            "Move model weights and input image batches to torch.channels_last "
            "memory format. Disabled by default. On bf16 CUDA hardware this can "
            "yield a 5-10%% encoder throughput improvement via NHWC conv kernels. "
            "A/B test with --profile-step-timing before committing to this flag."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Compile decoder + bridge modules with torch.compile. Default ON. "
            "Pass --no-torch-compile to disable (e.g., for debugging or environments "
            "that fail to compile). Encoder is excluded (RADIO uses trust_remote_code). "
            "Bench: -0.7% wall on cu132 nightly + RTX 5090 (Phase 4.2 of cu132 rollout)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help=(
            "DataLoader worker count for parallel sample loading. Default 4. "
            "Use 0 for the single-threaded fallback (required on Windows due to "
            "multiprocessing fork/spawn limitations)."
        ),
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help=(
            "DataLoader prefetch factor (number of batches loaded in advance per "
            "worker). Only effective when --num-workers > 0. Default 4."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]

    stage_paths = [project_root / Path(item.strip()) for item in args.stage_configs.split(",") if item.strip()]
    stages = [load_stage_config(path) for path in stage_paths]
    manifest_paths = _resolve_manifest_paths(project_root, args.token_manifest)
    if not args.allow_stale_merged_manifest:
        _assert_not_stale_merged_manifest(project_root, manifest_paths)

    token_entries: List[Dict[str, object]] = []
    for manifest_path in manifest_paths:
        if manifest_path.exists():
            loaded = load_token_manifest(manifest_path)
            print(f"[train] loaded {len(loaded):,} entries from {manifest_path.name}", file=sys.stderr)
            token_entries.extend(loaded)
        else:
            print(f"[train] manifest not found, skipping: {manifest_path}", file=sys.stderr)
    if not token_entries:
        raise FileNotFoundError("No token entries loaded from any manifest.")
    token_entries, dropped_entries = sanitize_token_entries(
        token_entries,
        enforce_strict_sequences=not args.allow_invalid_token_sequences,
    )
    if dropped_entries:
        print(
            f"[train] dropped {dropped_entries:,} token entries that failed vocabulary/grammar validation.",
            file=sys.stderr,
        )
    if not token_entries:
        raise RuntimeError("No valid token entries available after strict manifest filtering.")
    grouped = group_entries_by_dataset_and_split(token_entries)

    if args.mode == "execute":
        resume_checkpoint = args.resume_checkpoint
        if resume_checkpoint is not None:
            if not resume_checkpoint.is_absolute():
                resume_checkpoint = project_root / resume_checkpoint
            resume_checkpoint = resume_checkpoint.resolve()
        summary = run_execute_mode(
            stages=stages,
            grouped_entries=grouped,
            project_root=project_root,
            image_height=max(32, args.image_height),
            image_width=min(3000, max(256, args.image_max_width)),
            max_steps_per_stage=args.max_steps_per_stage,
            seed=args.seed,
            step_log_path=args.step_log,
            checkpoint_dir=args.checkpoint_dir,
            validation_batches=max(1, args.validation_batches),
            resume_checkpoint=resume_checkpoint,
            start_stage=args.start_stage,
            stage_b_backbone=str(args.stage_b_backbone),
            stage_b_decoder_dim=max(64, int(args.stage_b_decoder_dim)),
            stage_b_decoder_layers=max(1, int(args.stage_b_decoder_layers)),
            stage_b_decoder_heads=max(1, int(args.stage_b_decoder_heads)),
            stage_b_dora_rank=(
                None if args.stage_b_dora_rank is None else max(1, int(args.stage_b_dora_rank))
            ),
            stage_b_encoder=args.stage_b_encoder,
            keep_last_checkpoints=max(0, int(args.keep_last_checkpoints)),
            profile_step_timing=bool(args.profile_step_timing),
            profile_output_path=(
                args.profile_output
                if args.profile_output is not None
                else (project_root / "logs" / "profile_step_timing.jsonl")
            ),
            stage_b_radio_pool_to_stride32=bool(args.stage_b_radio_pool_to_stride32),
            diag_cadence=max(1, int(args.diag_cadence)),
            channels_last=bool(args.channels_last),
            torch_compile=bool(args.torch_compile),
            num_workers=max(0, int(args.num_workers)),
            prefetch_factor=max(1, int(args.prefetch_factor)),
        )
    else:
        summary = run_dry_mode(stages=stages, grouped_entries=grouped)

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    args.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

