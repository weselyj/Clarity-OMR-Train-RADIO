"""Post-decode cleanup heuristics applied to per-system token streams.

The Stage B decoder occasionally produces two specific kinds of
nonsense on real-world scans that the clean-rendering training data
didn't teach it to avoid:

1. **Phantom staff chunks.** A grand-staff system has 2 staves, but the
   decoder emits 3 ``<staff_start>...<staff_end>`` blocks. The extra
   block is typically either (a) an identical duplicate of the bass
   staff under a different clef, or (b) an all-rest "draft" followed
   by the real staff. Symptom in the MXL: alternating clefs on the
   bass part across systems, because the assembly stripes a varying
   staff count across two parts.

2. **Bass clef misread as treble.** The decoder emits ``clef-G2`` on a
   staff whose notes are predominantly in the bass register (below
   middle C). The MXL then ends up with two treble staves on a
   grand-staff system.

Both heuristics are pure-logic functions on a per-system token list.
They are gated behind ``enable=True`` kwargs on the cleaning entry
point so the caller (eg. predict_pdf) can disable them per-run.

These are NOT a substitute for fixing the underlying Stage B
generalization gap. They are a cheap, observable-failure-targeted
defensive layer that should be revisited after a proper retrain.
"""
from __future__ import annotations

import statistics
from typing import List, Optional, Sequence


_STAFF_START = "<staff_start>"
_STAFF_END = "<staff_end>"


def split_system_tokens_into_staves(tokens: Sequence[str]) -> List[List[str]]:
    """Split a per-system token list at ``<staff_start>...<staff_end>`` boundaries.

    Returns one inner list per staff, INCLUDING the ``<staff_start>`` and
    ``<staff_end>`` markers. Anything outside a staff block (eg. a leading
    ``<bos>``, a trailing ``<eos>``, or a stray token between blocks) is
    dropped because the rejoin logic expects each chunk to be a complete
    staff. If the decoder omitted ``<staff_end>`` for the final staff, the
    chunk extends to the end of the token list and is still returned.

    The function tolerates malformed input rather than raising — it is a
    diagnostic preprocessor, not a strict validator.
    """
    chunks: List[List[str]] = []
    i = 0
    n = len(tokens)
    while i < n:
        if tokens[i] != _STAFF_START:
            i += 1
            continue
        end = i + 1
        while end < n and tokens[end] != _STAFF_END:
            end += 1
        if end < n:
            chunks.append(list(tokens[i : end + 1]))
            i = end + 1
        else:
            # No <staff_end> found — assume the rest of the stream is this staff
            chunks.append(list(tokens[i:]))
            break
    return chunks


def _note_event_signature(chunk: Sequence[str]) -> tuple:
    """Compact signature of a chunk's musical content, used to detect duplicates.

    Returns a tuple of just the note / chord / rest / duration tokens — i.e.
    everything that affects the rhythmic and pitch content but NOT clef or
    staff_idx tokens (which are exactly the tokens that distinguish a
    phantom chunk from the real one).
    """
    keep_prefixes = ("note-",)
    keep_exact = {"rest", "<chord_start>", "<chord_end>"}
    keep_duration_prefixes = ("_",)  # _quarter, _half, _eighth, _dot, etc.
    out = []
    for t in chunk:
        if t in keep_exact:
            out.append(t)
        elif any(t.startswith(p) for p in keep_prefixes):
            out.append(t)
        elif any(t.startswith(p) for p in keep_duration_prefixes):
            out.append(t)
    return tuple(out)


def _has_any_notes(chunk: Sequence[str]) -> bool:
    """True if the chunk contains at least one note-* or <chord_start> token."""
    return any(t.startswith("note-") or t == "<chord_start>" for t in chunk)


def _median_note_octave(chunk: Sequence[str]) -> Optional[float]:
    """Median octave of pitches in the chunk. None if no pitches present.

    Pitch tokens have the shape ``note-<letter><accidental?><octave>``,
    e.g. ``note-C4``, ``note-G#3``, ``note-Bb2``. The octave is the
    trailing digit(s).
    """
    octaves: List[int] = []
    for t in chunk:
        if not t.startswith("note-"):
            continue
        # Trailing digits are the octave
        digits = ""
        for c in reversed(t):
            if c.isdigit():
                digits = c + digits
            else:
                break
        if digits:
            octaves.append(int(digits))
    if not octaves:
        return None
    return statistics.median(octaves)


def _clef_index(chunk: Sequence[str]) -> Optional[int]:
    """Index of the first clef-* token in the chunk, or None."""
    for i, t in enumerate(chunk):
        if t.startswith("clef-"):
            return i
    return None


def _drop_phantom_chunks(chunks: List[List[str]]) -> List[List[str]]:
    """Remove phantom staff chunks within one system.

    A chunk C is phantom if either:
      (a) Its note-event signature is identical to another chunk's
          (the decoder emitted the same staff content twice under
          different clefs), OR
      (b) C contains zero notes/chords AND at least one other chunk
          in the same system DOES contain notes (the decoder emitted
          a "draft" all-rest block followed by the real staff).

    When two chunks tie, the chunk whose clef matches its content's
    register is preferred (low-register content + bass clef wins over
    low-register content + treble clef).
    """
    if len(chunks) <= 2:
        return chunks  # 0, 1, or 2 staves is normal — never collapse

    # First pass: drop all-rest chunks if any sibling has notes
    any_has_notes = any(_has_any_notes(c) for c in chunks)
    after_rest_drop: List[List[str]] = []
    for c in chunks:
        if any_has_notes and not _has_any_notes(c):
            continue  # phantom: all-rest while sibling has notes
        after_rest_drop.append(c)
    chunks = after_rest_drop

    if len(chunks) <= 2:
        return chunks

    # Second pass: deduplicate chunks with identical note-event signatures.
    # When two have the same content but differ in clef, prefer the chunk
    # whose clef best fits the median note octave.
    out: List[List[str]] = []
    seen: dict[tuple, int] = {}  # signature -> index in `out`
    for c in chunks:
        sig = _note_event_signature(c)
        if sig in seen and sig != ():
            existing_idx = seen[sig]
            existing = out[existing_idx]
            # Decide which clef is more sensible
            existing_clef = _get_clef(existing)
            new_clef = _get_clef(c)
            existing_oct = _median_note_octave(existing)
            new_oct = _median_note_octave(c)
            # Same content → same octave; use existing octave or new — they tie.
            ref_oct = existing_oct if existing_oct is not None else new_oct
            # Bass clef preferred when content is below middle C; else treble.
            prefer_bass = ref_oct is not None and ref_oct < 4
            keep_new = (
                (prefer_bass and new_clef == "clef-F4" and existing_clef != "clef-F4")
                or (not prefer_bass and new_clef == "clef-G2" and existing_clef != "clef-G2")
            )
            if keep_new:
                out[existing_idx] = c
        else:
            seen[sig] = len(out)
            out.append(c)
    return out


def _get_clef(chunk: Sequence[str]) -> Optional[str]:
    """Return the first clef-* token in the chunk, or None."""
    for t in chunk:
        if t.startswith("clef-"):
            return t
    return None


def _maybe_swap_clef_for_bass_register(chunk: List[str]) -> List[str]:
    """If the chunk has clef-G2 but its notes are predominantly below middle
    C (median octave < 4), rewrite the clef to clef-F4.

    Threshold: median note octave < 4 (i.e. mostly B3 and lower). This
    cleanly fires on the bass staff of a grand-staff system that's been
    misread as treble, and is unlikely to fire on a real treble staff
    (which would have median octave 4-6).

    Returns a new list (does not mutate input).
    """
    if not chunk:
        return chunk
    idx = _clef_index(chunk)
    if idx is None or chunk[idx] != "clef-G2":
        return chunk
    median_oct = _median_note_octave(chunk)
    if median_oct is None or median_oct >= 4:
        return chunk
    new_chunk = list(chunk)
    new_chunk[idx] = "clef-F4"
    return new_chunk


def clean_system_tokens(
    tokens: Sequence[str],
    *,
    drop_phantom_staves: bool = True,
    repair_bass_clef: bool = False,
) -> List[str]:
    """Apply post-decode heuristics to a per-system token list.

    Splits the token list into per-staff chunks, optionally drops phantom
    chunks, optionally rewrites a misread bass clef, and rejoins. Anything
    outside a staff block (eg. a leading ``<bos>``) is preserved at the
    start of the output; trailing tokens are preserved at the end.

    ``drop_phantom_staves`` defaults ON: cleanly beneficial — drops
    duplicate / all-rest chunks the decoder emitted between real staves.

    ``repair_bass_clef`` defaults OFF: experimental. When the decoder
    misreads a bass-clef glyph as treble, it ALSO interprets the staff
    positions under the wrong clef and emits wrong pitch tokens. Swapping
    only the clef makes the clef tag correct but leaves pitches off by
    roughly 20 semitones (the treble↔bass staff-line mapping difference).
    The correct fix would transpose pitches alongside the clef swap; until
    that's implemented, enable only when visual clef-correctness matters
    more than pitch correctness (eg. when manually re-typesetting).

    Returns a new list; the input is not mutated.
    """
    tokens = list(tokens)
    if not tokens:
        return tokens

    # Preserve any tokens before the first <staff_start> (typically <bos>).
    try:
        first_staff_start = tokens.index(_STAFF_START)
    except ValueError:
        return tokens  # no staves to process
    prefix = tokens[:first_staff_start]

    # Preserve any tokens after the last <staff_end> (typically <eos>).
    suffix: List[str] = []
    last_staff_end = -1
    for i in range(len(tokens) - 1, -1, -1):
        if tokens[i] == _STAFF_END:
            last_staff_end = i
            break
    if last_staff_end >= 0:
        suffix = tokens[last_staff_end + 1 :]

    chunks = split_system_tokens_into_staves(tokens[first_staff_start:])
    if not chunks:
        return tokens

    if drop_phantom_staves:
        chunks = _drop_phantom_chunks(chunks)

    if repair_bass_clef and len(chunks) >= 2:
        # Repair the clef on the LAST staff (the bottom of the system).
        # Only the bottom staff in a grand-staff layout is at risk of the
        # bass-as-treble misread; rewriting an inner staff's clef is too
        # aggressive without more context.
        chunks[-1] = _maybe_swap_clef_for_bass_register(chunks[-1])

    out: List[str] = list(prefix)
    for c in chunks:
        out.extend(c)
    out.extend(suffix)
    return out
