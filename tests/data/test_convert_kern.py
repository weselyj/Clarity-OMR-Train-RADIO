"""Tests for src/data/convert_tokens.py::convert_kern_file (multi-spine support)."""
from __future__ import annotations

from pathlib import Path

from src.data.convert_tokens import convert_kern_file


def _write_kern(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "sample.krn"
    p.write_text(content, encoding="utf-8")
    return p


def test_single_spine_emits_no_marker(tmp_path: Path) -> None:
    """A 1-spine kern file produces a sequence without <staff_idx_*>."""
    krn = _write_kern(
        tmp_path,
        "**kern\n"
        "*clefG2\n"
        "*k[]\n"
        "*M4/4\n"
        "=1\n"
        "4c\n"
        "*-\n",
    )
    tokens = convert_kern_file(krn)
    assert "<bos>" in tokens
    assert "<eos>" in tokens
    assert "<staff_start>" in tokens
    assert "<staff_end>" in tokens
    assert not any(t.startswith("<staff_idx_") for t in tokens)
