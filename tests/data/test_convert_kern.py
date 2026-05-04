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


def test_two_spine_grand_staff_reverses_columns_for_top_down(tmp_path: Path) -> None:
    """A 2-spine kern (bass column 0, treble column 1) emits treble first as <staff_idx_0>."""
    krn = _write_kern(
        tmp_path,
        "**kern\t**kern\n"
        "*clefF4\t*clefG2\n"
        "*k[]\t*k[]\n"
        "*M4/4\t*M4/4\n"
        "=1\t=1\n"
        "4C\t4c\n"
        "*-\t*-\n",
    )
    tokens = convert_kern_file(krn)

    # Two staff_start blocks, top-down (treble first).
    assert tokens.count("<staff_start>") == 2
    assert tokens.count("<staff_end>") == 2
    idx0_pos = tokens.index("<staff_idx_0>")
    idx1_pos = tokens.index("<staff_idx_1>")
    assert idx0_pos < idx1_pos

    # Idx 0 (treble) should appear immediately after first <staff_start>.
    first_start = tokens.index("<staff_start>")
    assert tokens[first_start + 1] == "<staff_idx_0>"

    # Treble's clef is G2; bass clef F4. Verify ordering by clef.
    g2_pos = tokens.index("clef-G2")
    f4_pos = tokens.index("clef-F4")
    assert g2_pos < f4_pos, "treble (G2) block must precede bass (F4) block"
