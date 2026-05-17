"""Held-out real archetype-set manifest: schema, loader, validator.

Pure (no torch / no YOLO). One JSON file = a list of scenario objects.

Scenario JSON schema (all fields required):
  scenario_id  : str, unique across the manifest
  archetype    : str, the failure-mode label (free-form, e.g. "title_over_system")
  image        : str, path to the scan image (relative to the manifest or absolute)
  is_non_music : bool, True => a pure non-music page; gt_systems MUST be []
  gt_systems   : list of {box:[x1,y1,x2,y2], has_lyrics:bool,
                          lyric_bands:[[x1,y1,x2,y2], ...]}
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

Box = tuple[float, float, float, float]


def _as_box(value, where: str) -> Box:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"{where}: box must be 4 numbers [x1,y1,x2,y2], got {value!r}")
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError(f"{where}: box values must be numeric, got {value!r}")
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{where}: box must have x2>x1 and y2>y1, got {value!r}")
    return (x1, y1, x2, y2)


@dataclass(frozen=True)
class GtSystem:
    box: Box
    has_lyrics: bool
    lyric_bands: list[Box]


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    archetype: str
    image: str
    is_non_music: bool
    gt_systems: list[GtSystem]


_REQUIRED = ("scenario_id", "archetype", "image", "is_non_music", "gt_systems")


def load_manifest(path: str | Path) -> list[Scenario]:
    """Parse + validate the manifest. Raises ValueError on any malformation."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("manifest root must be a JSON list of scenarios")

    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for i, entry in enumerate(raw):
        where = f"scenario[{i}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{where}: must be an object")
        for field in _REQUIRED:
            if field not in entry:
                raise ValueError(f"{where}: missing required field '{field}'")

        sid = str(entry["scenario_id"])
        if sid in seen:
            raise ValueError(f"{where}: duplicate scenario_id {sid!r}")
        seen.add(sid)

        is_non_music = entry["is_non_music"]
        if not isinstance(is_non_music, bool):
            raise ValueError(f"{where}: is_non_music must be a bool")

        gt_raw = entry["gt_systems"]
        if not isinstance(gt_raw, list):
            raise ValueError(f"{where}: gt_systems must be a list")
        if is_non_music and gt_raw:
            raise ValueError(
                f"{where}: is_non_music=True requires empty gt_systems")

        gt_systems: list[GtSystem] = []
        for j, g in enumerate(gt_raw):
            gwhere = f"{where}.gt_systems[{j}]"
            if not isinstance(g, dict) or "box" not in g:
                raise ValueError(f"{gwhere}: must be an object with a 'box'")
            box = _as_box(g["box"], gwhere)
            has_lyrics = bool(g.get("has_lyrics", False))
            bands_raw = g.get("lyric_bands", []) or []
            if not isinstance(bands_raw, list):
                raise ValueError(f"{gwhere}: lyric_bands must be a list")
            bands = [_as_box(b, f"{gwhere}.lyric_bands[{k}]")
                     for k, b in enumerate(bands_raw)]
            gt_systems.append(GtSystem(box=box, has_lyrics=has_lyrics,
                                       lyric_bands=bands))

        scenarios.append(Scenario(
            scenario_id=sid,
            archetype=str(entry["archetype"]),
            image=str(entry["image"]),
            is_non_music=is_non_music,
            gt_systems=gt_systems,
        ))
    return scenarios
