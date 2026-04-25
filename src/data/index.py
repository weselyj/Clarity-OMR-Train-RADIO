#!/usr/bin/env python3
"""Build a canonical manifest for local OMR datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
# Maps physical folder name -> canonical dataset name used in train configs and manifest entries.
FOLDER_TO_DATASET: dict[str, str] = {
    "primus": "primus",
    "camera_primus": "cameraprimus",
    "grandstaff": "grandstaff",
    "openscore_lieder": "lieder-main",
}
DATASET_FOLDERS = tuple(FOLDER_TO_DATASET.keys())
IMAGE_EXTENSION_PRIORITY = {
    ".png": 0,
    ".jpg": 1,
    ".jpeg": 2,
    ".tif": 3,
    ".tiff": 4,
    ".bmp": 5,
}


def is_visible_file(name: str) -> bool:
    return not (name.startswith(".") or name.startswith("._"))


def relpath(project_root: Path, target: Path) -> str:
    return str(target.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def parse_scalar(value: str):
    value = value.strip()
    if not value:
        return ""
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_split_config(config_path: Path) -> Dict[str, Dict[str, float]]:
    config: Dict[str, object] = {
        "seed": 1337,
        "default": {"train": 0.9, "val": 0.05, "test": 0.05},
        "datasets": {},
    }

    if not config_path.exists():
        return config  # type: ignore[return-value]

    current_section: Optional[str] = None
    current_dataset: Optional[str] = None
    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        text = line.strip()

        if indent == 0:
            if ":" not in text:
                raise ValueError(f"Invalid config line: {raw_line}")
            key, value = [part.strip() for part in text.split(":", 1)]
            if key in {"seed"}:
                config[key] = parse_scalar(value)
            elif key in {"default", "datasets"}:
                if value:
                    raise ValueError(f"Section '{key}' cannot have inline value.")
                current_section = key
                current_dataset = None
            else:
                raise ValueError(f"Unknown top-level key '{key}' in {config_path}.")
            continue

        if current_section == "default" and indent == 2:
            key, value = [part.strip() for part in text.split(":", 1)]
            default_map = config["default"]  # type: ignore[index]
            default_map[key] = float(parse_scalar(value))
            continue

        if current_section == "datasets" and indent == 2 and text.endswith(":"):
            dataset = text[:-1].strip().lower()
            dataset_map = config["datasets"]  # type: ignore[index]
            dataset_map.setdefault(dataset, {})
            current_dataset = dataset
            continue

        if (
            current_section == "datasets"
            and indent == 4
            and current_dataset is not None
            and ":" in text
        ):
            key, value = [part.strip() for part in text.split(":", 1)]
            dataset_map = config["datasets"]  # type: ignore[index]
            dataset_map[current_dataset][key] = float(parse_scalar(value))
            continue

        raise ValueError(f"Could not parse config line: {raw_line}")

    return config  # type: ignore[return-value]


def validate_ratios(name: str, ratios: Dict[str, float]) -> None:
    missing = {"train", "val", "test"} - set(ratios)
    if missing:
        raise ValueError(f"{name} ratios missing keys: {sorted(missing)}")
    total = float(ratios["train"]) + float(ratios["val"]) + float(ratios["test"])
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} ratios must sum to 1.0, got {total}")


def choose_split(sample_id: str, seed: int, ratios: Dict[str, float]) -> str:
    digest = hashlib.sha1(f"{seed}:{sample_id}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    if value < ratios["train"]:
        return "train"
    if value < ratios["train"] + ratios["val"]:
        return "val"
    return "test"


def _normalize_variant_stem(stem: str) -> str:
    normalized = stem.strip()
    if normalized.endswith("_distorted"):
        return normalized[: -len("_distorted")]
    return normalized


def _split_key_for_entry(entry: Dict[str, Optional[str]]) -> str:
    dataset = str(entry.get("dataset") or "").lower()
    sample_id = str(entry.get("sample_id") or "")
    group_id = str(entry.get("group_id") or "")
    stem = sample_id.split(":")[-1] if sample_id else ""
    base_stem = _normalize_variant_stem(stem)

    # Primus and CameraPrimus often share identical notation targets.
    # Use a shared canonical key so paired clean/distorted variants
    # are forced into the same split and cannot leak across train/val/test.
    if dataset in {"primus", "cameraprimus"} and base_stem:
        return f"primus-family:{base_stem}"

    if group_id:
        return f"{dataset}:{group_id}"
    if base_stem:
        return f"{dataset}:{base_stem}"
    return f"{dataset}:{sample_id}"


def resolve_file(
    dir_path: Path, file_names: set[str], candidate_names: Iterable[str], project_root: Path
) -> Optional[str]:
    for candidate in candidate_names:
        if candidate in file_names:
            return relpath(project_root, dir_path / candidate)
    return None


def select_preferred_image_files(filenames: Iterable[str]) -> List[str]:
    preferred_by_stem: Dict[str, Tuple[int, str]] = {}
    for filename in filenames:
        suffix = Path(filename).suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            continue
        stem = Path(filename).stem
        priority = IMAGE_EXTENSION_PRIORITY.get(suffix, len(IMAGE_EXTENSION_PRIORITY))
        existing = preferred_by_stem.get(stem)
        if existing is None or priority < existing[0] or (priority == existing[0] and filename < existing[1]):
            preferred_by_stem[stem] = (priority, filename)
    return [item[1] for item in sorted(preferred_by_stem.values(), key=lambda value: value[1])]


def iter_primus_style(
    dataset_name: str, dataset_root: Path, project_root: Path
) -> Iterator[Dict[str, Optional[str]]]:
    for dirpath, _, filenames in os.walk(dataset_root):
        visible = [name for name in filenames if is_visible_file(name)]
        if not visible:
            continue
        file_set = set(visible)
        dir_path = Path(dirpath)
        for filename in select_preferred_image_files(visible):
            stem = Path(filename).stem
            is_distorted = stem.endswith("_distorted")
            if dataset_name.lower() == "cameraprimus" and not is_distorted:
                continue
            base_stem = stem[: -len("_distorted")] if is_distorted else stem

            dir_rel = str(dir_path.relative_to(dataset_root)).replace("\\", "/")
            group_id = f"{dir_rel}/{base_stem}" if dir_rel != "." else base_stem
            sample_id = f"{dataset_name.lower()}:{group_id}:{stem}"

            yield {
                "sample_id": sample_id,
                "dataset": dataset_name.lower(),
                "group_id": group_id,
                "modality": "image+notation",
                "variant": "distorted" if is_distorted else "clean",
                "image_path": relpath(project_root, dir_path / filename),
                "mei_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.mei",), project_root
                ),
                "semantic_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.semantic",), project_root
                ),
                "agnostic_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.agnostic",), project_root
                ),
                "midi_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.mid",), project_root
                ),
                "pae_path": resolve_file(
                    dir_path,
                    file_set,
                    (f"{base_stem}.pae", "regular_pae.pae"),
                    project_root,
                ),
                "musicxml_path": None,
                "krn_path": None,
                "bekrn_path": None,
                "gm_path": None,
                "mscz_path": None,
                "mscx_path": None,
                "meta_txt_path": None,
            }


def iter_grandstaff(
    dataset_name: str, dataset_root: Path, project_root: Path
) -> Iterator[Dict[str, Optional[str]]]:
    for dirpath, _, filenames in os.walk(dataset_root):
        visible = [name for name in filenames if is_visible_file(name)]
        if not visible:
            continue
        file_set = set(visible)
        dir_path = Path(dirpath)
        for filename in select_preferred_image_files(visible):

            stem = Path(filename).stem
            is_distorted = stem.endswith("_distorted")
            base_stem = stem[: -len("_distorted")] if is_distorted else stem

            dir_rel = str(dir_path.relative_to(dataset_root)).replace("\\", "/")
            group_id = f"{dir_rel}/{base_stem}" if dir_rel != "." else base_stem
            sample_id = f"{dataset_name.lower()}:{group_id}:{stem}"

            yield {
                "sample_id": sample_id,
                "dataset": dataset_name.lower(),
                "group_id": group_id,
                "modality": "image+notation",
                "variant": "distorted" if is_distorted else "clean",
                "image_path": relpath(project_root, dir_path / filename),
                "mei_path": None,
                "semantic_path": None,
                "agnostic_path": None,
                "midi_path": None,
                "pae_path": None,
                "musicxml_path": None,
                "krn_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.krn",), project_root
                ),
                "bekrn_path": resolve_file(
                    dir_path, file_set, (f"{base_stem}.bekrn",), project_root
                ),
                "gm_path": resolve_file(
                    dir_path,
                    file_set,
                    (f"{base_stem}.gm", f"{base_stem}.jpg.gm"),
                    project_root,
                ),
                "mscz_path": None,
                "mscx_path": None,
                "meta_txt_path": None,
            }


def iter_lieder(
    dataset_name: str, dataset_root: Path, project_root: Path
) -> Iterator[Dict[str, Optional[str]]]:
    for dirpath, _, filenames in os.walk(dataset_root):
        visible = [name for name in filenames if is_visible_file(name)]
        if not visible:
            continue
        file_set = set(visible)
        dir_path = Path(dirpath)
        for filename in sorted(visible):
            if Path(filename).suffix.lower() != ".mxl":
                continue
            stem = Path(filename).stem
            dir_rel = str(dir_path.relative_to(dataset_root)).replace("\\", "/")
            group_id = f"{dir_rel}/{stem}" if dir_rel != "." else stem
            sample_id = f"{dataset_name.lower()}:{group_id}:{stem}"

            yield {
                "sample_id": sample_id,
                "dataset": dataset_name.lower(),
                "group_id": group_id,
                "modality": "symbolic",
                "variant": "clean",
                "image_path": None,
                "mei_path": None,
                "semantic_path": None,
                "agnostic_path": None,
                "midi_path": None,
                "pae_path": None,
                "musicxml_path": relpath(project_root, dir_path / filename),
                "krn_path": None,
                "bekrn_path": None,
                "gm_path": None,
                "mscz_path": resolve_file(
                    dir_path, file_set, (f"{stem}.mscz",), project_root
                ),
                "mscx_path": resolve_file(
                    dir_path, file_set, (f"{stem}.mscx",), project_root
                ),
                "meta_txt_path": resolve_file(
                    dir_path, file_set, (f"{stem}.txt",), project_root
                ),
            }


def build_manifest(
    project_root: Path,
    data_root: Path,
    output_path: Path,
    summary_path: Path,
    split_config: Dict[str, Dict[str, float]],
    max_samples_per_dataset: Optional[int],
) -> Tuple[int, Dict[str, int]]:
    builders = {
        "primus": iter_primus_style,
        "cameraprimus": iter_primus_style,
        "grandstaff": iter_grandstaff,
        "lieder-main": iter_lieder,
    }

    seed = int(split_config["seed"])  # type: ignore[index]
    default_ratios = split_config["default"]  # type: ignore[index]
    validate_ratios("default", default_ratios)  # type: ignore[arg-type]

    dataset_ratios = split_config["datasets"]  # type: ignore[index]
    for dataset_name, ratios in dataset_ratios.items():  # type: ignore[assignment]
        validate_ratios(f"datasets.{dataset_name}", ratios)

    dataset_counts: Counter[str] = Counter()
    split_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    missing_counts: Dict[str, Counter[str]] = defaultdict(Counter)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as manifest_file:
        for folder_name in DATASET_FOLDERS:
            dataset_key = FOLDER_TO_DATASET.get(folder_name, folder_name.lower())
            builder = builders.get(dataset_key)
            if builder is None:
                continue
            dataset_dir = data_root / folder_name
            if not dataset_dir.exists():
                raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

            ratios = dataset_ratios.get(dataset_key, default_ratios)  # type: ignore[union-attr]
            written = 0
            for entry in builder(dataset_key, dataset_dir, project_root):
                if max_samples_per_dataset is not None and written >= max_samples_per_dataset:
                    break
                split_key = _split_key_for_entry(entry)
                entry["split"] = choose_split(split_key, seed, ratios)  # type: ignore[arg-type]
                manifest_file.write(json.dumps(entry, sort_keys=True) + "\n")
                written += 1
                dataset_counts[dataset_key] += 1
                split_counts[dataset_key][entry["split"]] += 1  # type: ignore[index]

                for required_field in ("image_path", "musicxml_path", "mei_path"):
                    if entry.get(required_field) is None:
                        missing_counts[dataset_key][required_field] += 1

    summary = {
        "seed": seed,
        "manifest_path": relpath(project_root, output_path),
        "total_samples": sum(dataset_counts.values()),
        "datasets": {},
    }
    for dataset_name in sorted(dataset_counts):
        summary["datasets"][dataset_name] = {
            "samples": dataset_counts[dataset_name],
            "splits": dict(split_counts[dataset_name]),
            "missing_fields": dict(missing_counts[dataset_name]),
        }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return sum(dataset_counts.values()), dict(dataset_counts)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build canonical manifest for OMR datasets.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root path (default: inferred from this file).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data",
        help="Root directory containing source datasets.",
    )
    parser.add_argument(
        "--split-config",
        type=Path,
        default=project_root / "configs" / "splits.yaml",
        help="Split config file path.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "master_manifest.jsonl",
        help="Output JSONL manifest path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "master_manifest_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=None,
        help="Optional cap per dataset for quick smoke tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_config = load_split_config(args.split_config)
    total_samples, dataset_counts = build_manifest(
        project_root=args.project_root,
        data_root=args.data_root,
        output_path=args.output_manifest,
        summary_path=args.output_summary,
        split_config=split_config,
        max_samples_per_dataset=args.max_samples_per_dataset,
    )
    print(f"Wrote {total_samples} samples to {args.output_manifest}")
    for dataset_name, count in sorted(dataset_counts.items()):
        print(f"  - {dataset_name}: {count}")


if __name__ == "__main__":
    main()
