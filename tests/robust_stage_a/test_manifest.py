"""CPU unit tests for the held-out archetype manifest loader/validator."""
import json
from pathlib import Path

import pytest

from eval.robust_stage_a.manifest import (
    GtSystem,
    Scenario,
    load_manifest,
)


def _write(tmp_path: Path, obj) -> Path:
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def test_loads_music_scenario(tmp_path):
    p = _write(tmp_path, [
        {
            "scenario_id": "title_over_system_01",
            "archetype": "title_over_system",
            "image": "scans/a.png",
            "is_non_music": False,
            "gt_systems": [
                {"box": [10, 20, 300, 80], "has_lyrics": True,
                 "lyric_bands": [[12, 70, 298, 80]]},
                {"box": [10, 100, 300, 160], "has_lyrics": False,
                 "lyric_bands": []},
            ],
        }
    ])
    scenarios = load_manifest(p)
    assert len(scenarios) == 1
    s = scenarios[0]
    assert isinstance(s, Scenario)
    assert s.scenario_id == "title_over_system_01"
    assert s.archetype == "title_over_system"
    assert s.is_non_music is False
    assert len(s.gt_systems) == 2
    assert isinstance(s.gt_systems[0], GtSystem)
    assert s.gt_systems[0].box == (10.0, 20.0, 300.0, 80.0)
    assert s.gt_systems[0].has_lyrics is True
    assert s.gt_systems[0].lyric_bands == [(12.0, 70.0, 298.0, 80.0)]
    assert s.gt_systems[1].has_lyrics is False
    assert s.gt_systems[1].lyric_bands == []


def test_loads_non_music_scenario_with_no_systems(tmp_path):
    p = _write(tmp_path, [
        {
            "scenario_id": "warranty_deed_01",
            "archetype": "pure_non_music",
            "image": "scans/deed.png",
            "is_non_music": True,
            "gt_systems": [],
        }
    ])
    scenarios = load_manifest(p)
    assert scenarios[0].is_non_music is True
    assert scenarios[0].gt_systems == []


def test_rejects_missing_required_field(tmp_path):
    p = _write(tmp_path, [{"scenario_id": "x", "archetype": "a",
                           "image": "i.png", "gt_systems": []}])  # no is_non_music
    with pytest.raises(ValueError, match="is_non_music"):
        load_manifest(p)


def test_rejects_non_music_with_systems(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "x", "archetype": "a", "image": "i.png",
         "is_non_music": True,
         "gt_systems": [{"box": [0, 0, 1, 1], "has_lyrics": False,
                         "lyric_bands": []}]}
    ])
    with pytest.raises(ValueError, match="is_non_music.*gt_systems"):
        load_manifest(p)


def test_rejects_bad_box_arity(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "x", "archetype": "a", "image": "i.png",
         "is_non_music": False,
         "gt_systems": [{"box": [0, 0, 1], "has_lyrics": False,
                         "lyric_bands": []}]}
    ])
    with pytest.raises(ValueError, match="box"):
        load_manifest(p)


def test_rejects_duplicate_scenario_ids(tmp_path):
    p = _write(tmp_path, [
        {"scenario_id": "dup", "archetype": "a", "image": "i.png",
         "is_non_music": True, "gt_systems": []},
        {"scenario_id": "dup", "archetype": "a", "image": "j.png",
         "is_non_music": True, "gt_systems": []},
    ])
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(p)
