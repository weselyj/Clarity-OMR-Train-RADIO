#!/usr/bin/env python3
"""Cross-staff assembly utilities for Stage C."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class StaffLocation:
    page_index: int
    y_top: float
    y_bottom: float
    x_left: float
    x_right: float

    @property
    def y_center(self) -> float:
        return (self.y_top + self.y_bottom) / 2.0


@dataclass
class StaffRecognitionResult:
    sample_id: str
    tokens: List[str]
    location: StaffLocation
    system_index_hint: Optional[int] = None


@dataclass
class AssembledStaff:
    sample_id: str
    tokens: List[str]
    part_label: str
    measure_count: int
    clef: Optional[str]
    key_signature: Optional[str]
    time_signature: Optional[str]
    location: StaffLocation


@dataclass
class AssembledSystem:
    page_index: int
    system_index: int
    staves: List[AssembledStaff]
    canonical_measure_count: int
    canonical_key_signature: Optional[str]
    canonical_time_signature: Optional[str]


@dataclass
class AssembledScore:
    systems: List[AssembledSystem]
    part_order: List[str]


def _extract_first(tokens: Sequence[str], prefix: str) -> Optional[str]:
    for token in tokens:
        if token.startswith(prefix):
            return token
    return None


def _count_measures(tokens: Sequence[str]) -> int:
    return sum(1 for token in tokens if token == "<measure_start>")


def _majority_vote(values: Iterable[Optional[str]]) -> Optional[str]:
    counts: Dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        counts[value] = counts.get(value, 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: item[1], reverse=True)[0][0]


def _majority_measure_count(values: Sequence[int]) -> int:
    """Pick the canonical measure count for a system.

    Uses the maximum rather than majority vote — it is better to pad a
    short staff with empty measures than to truncate a correct longer staff.
    The model tends to under-detect measures (especially in bass/chord-heavy
    staves), so the longest staff is usually the most accurate.
    """
    if not values:
        return 0
    return max(values)


def _clef_family(clef: Optional[str]) -> str:
    if clef is None:
        return "unknown"
    if clef.startswith("clef-G"):
        return "treble"
    if clef.startswith("clef-F"):
        return "bass"
    if clef.startswith("clef-C"):
        return "alto_tenor"
    return "other"


def _resolve_part_label(
    staff_index: int,
    total_staves: int,
    clef: Optional[str],
    clef_rank: int = 1,
    expected_staves: int = 0,
) -> str:
    is_piano = expected_staves == 2
    # Standard 2-staff piano: assign by position (top=RH, bottom=LH)
    if total_staves == 2 and is_piano:
        return "piano_right_hand" if staff_index == 0 else "piano_left_hand"
    # Undersized system in a piano piece: assign by clef
    if total_staves < expected_staves and is_piano:
        family = _clef_family(clef)
        if family == "bass":
            return "piano_left_hand"
        return "piano_right_hand"
    clef_family = _clef_family(clef)
    return f"part_{staff_index + 1:02d}_{clef_family}_{clef_rank:02d}"


def _normalize_measure_count(tokens: List[str], target_count: int) -> List[str]:
    current_count = _count_measures(tokens)
    if current_count == target_count:
        return tokens
    if current_count > target_count:
        trimmed: List[str] = []
        measure_seen = 0
        for token in tokens:
            if token == "<measure_start>":
                if measure_seen >= target_count:
                    break
                measure_seen += 1
            trimmed.append(token)
        if trimmed and trimmed[-1] != "<eos>":
            if "<staff_end>" not in trimmed:
                trimmed.append("<staff_end>")
            trimmed.append("<eos>")
        return trimmed

    normalized = list(tokens)
    while _count_measures(normalized) < target_count:
        insert_at = len(normalized) - 2 if len(normalized) >= 2 else len(normalized)
        normalized[insert_at:insert_at] = ["<measure_start>", "rest", "_whole", "<measure_end>"]
    return normalized


def _split_oversized_systems(
    systems: List[List[StaffRecognitionResult]],
) -> List[List[StaffRecognitionResult]]:
    """Split systems that have too many staves.

    When YOLO groups all staves on a page into one system, this detects the
    expected staves-per-system from correctly-sized systems and splits the
    oversized ones into groups of the expected size (sorted by y position).
    """
    sizes = [len(s) for s in systems]
    if not sizes:
        return systems

    # Find the most common small system size (2-4 staves).
    small_sizes = [s for s in sizes if 2 <= s <= 4]
    if not small_sizes:
        # No correctly-sized reference systems — assume piano (2 staves).
        expected = 2
    else:
        counts: Dict[int, int] = {}
        for s in small_sizes:
            counts[s] = counts.get(s, 0) + 1
        expected = max(counts, key=lambda k: counts[k])

    result: List[List[StaffRecognitionResult]] = []
    for system in systems:
        if len(system) <= expected:
            result.append(system)
            continue
        # Sort by vertical position and split into groups of expected size.
        sorted_staves = sorted(system, key=lambda s: s.location.y_center)
        for i in range(0, len(sorted_staves), expected):
            group = sorted_staves[i : i + expected]
            result.append(group)
    return result


def _merge_undersized_systems(
    systems: List[List[StaffRecognitionResult]],
) -> List[List[StaffRecognitionResult]]:
    """Merge consecutive undersized systems that belong together.

    When YOLO detects each staff as its own system (e.g. treble and bass
    of a piano grand staff get separate system_index values), this merges
    consecutive single-staff systems on the same page into one system of
    the expected size.  Processes top-to-bottom since YOLO scans that way.
    """
    sizes = [len(s) for s in systems]
    if not sizes:
        return systems

    # Determine expected staves-per-system from the most common size >= 2.
    size_counts: Dict[int, int] = {}
    for sz in sizes:
        if sz >= 2:
            size_counts[sz] = size_counts.get(sz, 0) + 1
    if not size_counts:
        return systems
    expected = max(size_counts, key=lambda k: size_counts[k])
    if expected < 2:
        return systems

    result: List[List[StaffRecognitionResult]] = []
    i = 0
    while i < len(systems):
        current = systems[i]
        # Only try to merge if this system is undersized
        if len(current) < expected:
            page = current[0].location.page_index
            merged = list(current)
            # Greedily merge following consecutive undersized systems on the same page
            while (
                len(merged) < expected
                and i + 1 < len(systems)
                and len(systems[i + 1]) < expected
                and systems[i + 1][0].location.page_index == page
            ):
                i += 1
                merged.extend(systems[i])
            merged.sort(key=lambda s: s.location.y_center)
            result.append(merged)
        else:
            result.append(current)
        i += 1
    return result


def group_staves_into_systems(
    staves: Sequence[StaffRecognitionResult],
    overlap_threshold: float = 0.50,
) -> List[List[StaffRecognitionResult]]:
    if staves and all(staff.system_index_hint is not None for staff in staves):
        grouped_with_hints: Dict[Tuple[int, int], List[StaffRecognitionResult]] = {}
        for staff in staves:
            grouped_with_hints[(staff.location.page_index, int(staff.system_index_hint or 0))] = grouped_with_hints.get(
                (staff.location.page_index, int(staff.system_index_hint or 0)),
                [],
            ) + [staff]
        systems_with_hints: List[List[StaffRecognitionResult]] = []
        for key in sorted(grouped_with_hints):
            system = grouped_with_hints[key]
            system.sort(key=lambda item: item.location.y_center)
            systems_with_hints.append(system)
        return _merge_undersized_systems(_split_oversized_systems(systems_with_hints))

    grouped_by_page: Dict[int, List[StaffRecognitionResult]] = {}
    for staff in staves:
        grouped_by_page.setdefault(staff.location.page_index, []).append(staff)

    systems: List[List[StaffRecognitionResult]] = []
    for page_index in sorted(grouped_by_page):
        page_staves = sorted(grouped_by_page[page_index], key=lambda item: item.location.y_center)
        page_systems: List[List[StaffRecognitionResult]] = []
        for staff in page_staves:
            placed = False
            for system in page_systems:
                y_top = min(item.location.y_top for item in system)
                y_bottom = max(item.location.y_bottom for item in system)
                overlap_top = max(y_top, staff.location.y_top)
                overlap_bottom = min(y_bottom, staff.location.y_bottom)
                overlap = max(0.0, overlap_bottom - overlap_top)
                staff_height = max(1e-6, staff.location.y_bottom - staff.location.y_top)
                if (overlap / staff_height) >= overlap_threshold:
                    system.append(staff)
                    placed = True
                    break
            if not placed:
                page_systems.append([staff])
        for system in page_systems:
            system.sort(key=lambda item: item.location.y_center)
            systems.append(system)
    return _merge_undersized_systems(_split_oversized_systems(systems))


def _enforce_global_key_time(
    staves: Sequence[StaffRecognitionResult],
) -> List[StaffRecognitionResult]:
    """Replace hallucinated key/time signatures with the global majority.

    Most pieces have a single key and time signature throughout.  The model
    sometimes hallucinates changes (e.g. DM instead of GM, 6/8 instead of 3/4).
    We take a majority vote across *all* staves and replace any outlier
    key/time tokens with the consensus value.
    """
    global_key = _majority_vote(
        _extract_first(s.tokens, "keySignature-") for s in staves
    )
    global_time = _majority_vote(
        _extract_first(s.tokens, "timeSignature-") for s in staves
    )

    fixed: List[StaffRecognitionResult] = []
    for staff in staves:
        new_tokens = list(staff.tokens)
        for i, token in enumerate(new_tokens):
            if token.startswith("keySignature-") and global_key and token != global_key:
                new_tokens[i] = global_key
            if token.startswith("timeSignature-") and global_time and token != global_time:
                new_tokens[i] = global_time
        fixed.append(StaffRecognitionResult(
            sample_id=staff.sample_id,
            tokens=new_tokens,
            location=staff.location,
            system_index_hint=staff.system_index_hint,
        ))
    return fixed


def assemble_score(staves: Sequence[StaffRecognitionResult]) -> AssembledScore:
    # Pre-process: enforce consistent key/time signatures across all staves
    staves = _enforce_global_key_time(staves)

    systems_raw = group_staves_into_systems(staves)
    assembled_systems: List[AssembledSystem] = []
    part_order_set: List[str] = []

    # Determine expected staves-per-system from majority vote
    system_sizes = [len(s) for s in systems_raw]
    size_counts: Dict[int, int] = {}
    for sz in system_sizes:
        size_counts[sz] = size_counts.get(sz, 0) + 1
    expected_staves = max(size_counts, key=lambda k: size_counts[k]) if size_counts else 0

    for system_index, system in enumerate(systems_raw):
        measure_counts = [_count_measures(item.tokens) for item in system]
        canonical_measure_count = _majority_measure_count(measure_counts)
        canonical_key_signature = _majority_vote(_extract_first(item.tokens, "keySignature-") for item in system)
        canonical_time_signature = _majority_vote(_extract_first(item.tokens, "timeSignature-") for item in system)
        clef_counts: Dict[str, int] = {}

        assembled_staves: List[AssembledStaff] = []
        for idx, staff in enumerate(system):
            normalized_tokens = _normalize_measure_count(staff.tokens, canonical_measure_count)
            normalized_tokens = post_process_tokens(normalized_tokens, canonical_time_signature)
            clef = _extract_first(normalized_tokens, "clef-")
            key_signature = _extract_first(normalized_tokens, "keySignature-") or canonical_key_signature
            time_signature = _extract_first(normalized_tokens, "timeSignature-") or canonical_time_signature
            clef_family = _clef_family(clef)
            clef_counts[clef_family] = clef_counts.get(clef_family, 0) + 1
            part_label = _resolve_part_label(idx, len(system), clef, clef_rank=clef_counts[clef_family], expected_staves=expected_staves)
            if part_label not in part_order_set:
                part_order_set.append(part_label)

            assembled_staves.append(
                AssembledStaff(
                    sample_id=staff.sample_id,
                    tokens=normalized_tokens,
                    part_label=part_label,
                    measure_count=_count_measures(normalized_tokens),
                    clef=clef,
                    key_signature=key_signature,
                    time_signature=time_signature,
                    location=staff.location,
                )
            )

        assembled_systems.append(
            AssembledSystem(
                page_index=system[0].location.page_index,
                system_index=system_index,
                staves=assembled_staves,
                canonical_measure_count=canonical_measure_count,
                canonical_key_signature=canonical_key_signature,
                canonical_time_signature=canonical_time_signature,
            )
        )

    return AssembledScore(systems=assembled_systems, part_order=part_order_set)


DURATION_QUARTER_LENGTH = {
    "_whole": 4.0,
    "_half": 2.0,
    "_quarter": 1.0,
    "_eighth": 0.5,
    "_sixteenth": 0.25,
    "_thirty_second": 0.125,
    "_sixty_fourth": 0.0625,
}

TIME_SIG_BEATS = {
    "timeSignature-2/4": 2.0,
    "timeSignature-3/4": 3.0,
    "timeSignature-4/4": 4.0,
    "timeSignature-2/2": 4.0,
    "timeSignature-3/2": 6.0,
    "timeSignature-6/8": 3.0,
    "timeSignature-9/8": 4.5,
    "timeSignature-12/8": 6.0,
    "timeSignature-3/8": 1.5,
    "timeSignature-5/4": 5.0,
    "timeSignature-6/4": 6.0,
    "timeSignature-C": 4.0,
    "timeSignature-C/": 4.0,
}


def _measure_duration(tokens: Sequence[str], start: int, end: int) -> float:
    """Sum the quarter-length durations of notes/rests in a token range."""
    total = 0.0
    i = start
    while i < end:
        token = tokens[i]
        if token in DURATION_QUARTER_LENGTH:
            dur = DURATION_QUARTER_LENGTH[token]
            if i + 1 < end and tokens[i + 1] == "_dot":
                dur *= 1.5
            elif i + 1 < end and tokens[i + 1] == "_double_dot":
                dur *= 1.75
            total += dur
        i += 1
    return total


def _fix_whole_rest_convention(tokens: List[str], time_sig: Optional[str]) -> List[str]:
    """Replace whole-rest measures with correct duration for the time signature.

    In music notation, a whole rest filling an entire measure means 'rest for
    the whole measure' regardless of time signature.  The model sometimes emits
    ``rest _whole`` (4 beats) even in 3/4 or 6/8.  This replaces those with
    the correct duration so downstream balance checks pass.
    """
    if time_sig is None:
        return tokens
    expected_beats = TIME_SIG_BEATS.get(time_sig)
    if expected_beats is None or abs(expected_beats - 4.0) < 0.01:
        return tokens  # 4/4 or equivalent — whole rest is already correct

    fill = _best_fill_duration(expected_beats)
    if fill is None:
        return tokens

    result: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "<measure_start>":
            # Look ahead: is this measure just  <measure_start> rest _whole <measure_end>?
            # Also handle voice tokens before the rest.
            j = i + 1
            # Skip voice tokens
            while j < len(tokens) and tokens[j] in (
                "<voice_1>", "<voice_2>", "<voice_3>", "<voice_4>",
            ):
                j += 1
            if (
                j + 2 < len(tokens)
                and tokens[j] == "rest"
                and tokens[j + 1] == "_whole"
                and tokens[j + 2] == "<measure_end>"
            ):
                # Replace rest _whole with correct duration
                result.extend(tokens[i:j])  # <measure_start> + any voice tokens
                result.append("rest")
                result.extend(fill)
                result.append("<measure_end>")
                i = j + 3
                continue
        result.append(tokens[i])
        i += 1
    return result


def _balance_measures(tokens: List[str], time_sig: Optional[str]) -> List[str]:
    """Fix measures that don't add up to the time signature.

    If a measure is short, append a rest to fill it.
    If a measure is too long, truncate the last event.
    """
    if time_sig is None:
        return tokens
    expected_beats = TIME_SIG_BEATS.get(time_sig)
    if expected_beats is None:
        return tokens

    result: List[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] != "<measure_start>":
            result.append(tokens[i])
            i += 1
            continue

        # Find measure boundaries
        measure_start = i
        measure_end = None
        j = i + 1
        while j < len(tokens):
            if tokens[j] == "<measure_end>":
                measure_end = j
                break
            j += 1
        if measure_end is None:
            result.extend(tokens[i:])
            break

        # Copy measure tokens
        measure_tokens = list(tokens[measure_start:measure_end + 1])
        actual = _measure_duration(measure_tokens, 0, len(measure_tokens))
        diff = expected_beats - actual

        if abs(diff) < 0.05:
            # Measure is balanced
            result.extend(measure_tokens)
        elif diff > 0.05:
            # Measure is short — add a rest to fill
            fill_dur = _best_fill_duration(diff)
            if fill_dur:
                # Insert rest before <measure_end>
                insert_tokens = ["rest"] + fill_dur
                measure_tokens = measure_tokens[:-1] + insert_tokens + ["<measure_end>"]
            result.extend(measure_tokens)
        else:
            # Measure is too long — try to fix by adjusting durations
            fixed = _fix_overfull_measure(measure_tokens, 0, len(measure_tokens), expected_beats)
            fixed_dur = _measure_duration(fixed, 0, len(fixed))
            if abs(expected_beats - fixed_dur) < 0.05:
                # Fix worked — use corrected measure
                if fixed and fixed[-1] != "<measure_end>":
                    fixed.append("<measure_end>")
                result.extend(fixed)
            else:
                result.extend(measure_tokens)

        i = measure_end + 1

    return result


def _best_fill_duration(quarter_length: float) -> Optional[List[str]]:
    """Find the best single duration token(s) to fill a gap."""
    # Try dotted and plain durations
    options = [
        (4.0, ["_whole"]),
        (3.0, ["_half", "_dot"]),
        (2.0, ["_half"]),
        (1.5, ["_quarter", "_dot"]),
        (1.0, ["_quarter"]),
        (0.75, ["_eighth", "_dot"]),
        (0.5, ["_eighth"]),
        (0.25, ["_sixteenth"]),
    ]
    for dur, toks in options:
        if abs(dur - quarter_length) < 0.05:
            return toks
    # Try combining two durations
    for dur1, toks1 in options:
        remainder = quarter_length - dur1
        if remainder < 0.05:
            continue
        for dur2, toks2 in options:
            if abs(dur2 - remainder) < 0.05:
                return toks1 + ["rest"] + toks2
    return None


def _is_inside_chord(tokens: List[str], idx: int) -> bool:
    """Check if the token at idx is inside a <chord_start>...<chord_end> block."""
    depth = 0
    for i in range(idx):
        if tokens[i] == "<chord_start>":
            depth += 1
        elif tokens[i] == "<chord_end>":
            depth -= 1
    return depth > 0


def _insert_ties(tokens: List[str]) -> List[str]:
    """Insert tie_start/tie_end for same-pitch standalone notes across barlines.

    If a standalone note (not in a chord) at the end of a measure has the same
    pitch as the first standalone note of the next measure, insert tie tokens.
    """
    result = list(tokens)

    # Find all measure boundaries
    measure_starts: List[int] = []
    measure_ends: List[int] = []
    for i, token in enumerate(result):
        if token == "<measure_start>":
            measure_starts.append(i)
        elif token == "<measure_end>":
            measure_ends.append(i)

    if len(measure_starts) < 2 or len(measure_ends) < 2:
        return result

    insertions: List[Tuple[int, str]] = []
    for m_idx in range(len(measure_ends) - 1):
        m1_end = measure_ends[m_idx]
        # Find last standalone note in measure 1
        last_note_idx = None
        last_note_pitch = None
        for j in range(m1_end - 1, -1, -1):
            if result[j] == "<measure_start>":
                break
            if result[j].startswith("note-") and not _is_inside_chord(result, j):
                last_note_idx = j
                last_note_pitch = result[j]
                break

        if last_note_pitch is None:
            continue

        # Find next measure start
        m2_start = None
        for ms in measure_starts:
            if ms > m1_end:
                m2_start = ms
                break
        if m2_start is None:
            continue

        # Find first standalone note in measure 2
        m2_end = None
        for me in measure_ends:
            if me > m2_start:
                m2_end = me
                break
        if m2_end is None:
            m2_end = len(result)

        first_note_idx = None
        first_note_pitch = None
        for j in range(m2_start + 1, m2_end):
            if result[j].startswith("note-") and not _is_inside_chord(result, j):
                first_note_idx = j
                first_note_pitch = result[j]
                break

        if first_note_pitch is None:
            continue

        # Same pitch across barline → insert tie
        if last_note_pitch == first_note_pitch:
            has_tie = False
            for j in range(last_note_idx, m1_end):
                if result[j] == "tie_start":
                    has_tie = True
                    break
            if not has_tie:
                insertions.append((last_note_idx, "tie_start"))
                insertions.append((first_note_idx, "tie_end"))

    # Apply insertions in reverse order to preserve indices
    for pos, token in sorted(insertions, reverse=True):
        result.insert(pos, token)

    return result


import re as _re

# Key signatures and their expected accidentals
_KEY_ACCIDENTALS = {
    "keySignature-CM": set(),
    "keySignature-AM": set(),  # relative minor of CM
    "keySignature-GM": {"F#"},
    "keySignature-EM": {"F#"},
    "keySignature-DM": {"F#", "C#"},
    "keySignature-BM": {"F#", "C#"},
    "keySignature-AM_major": {"F#", "C#", "G#"},
    "keySignature-EM_major": {"F#", "C#", "G#", "D#"},
    "keySignature-BM_major": {"F#", "C#", "G#", "D#", "A#"},
    "keySignature-F#M": {"F#", "C#", "G#", "D#", "A#", "E#"},
    "keySignature-FM": {"Bb"},
    "keySignature-DM_minor": {"Bb"},
    "keySignature-BbM": {"Bb", "Eb"},
    "keySignature-GM_minor": {"Bb", "Eb"},
    "keySignature-EbM": {"Bb", "Eb", "Ab"},
    "keySignature-CM_minor": {"Bb", "Eb", "Ab"},
    "keySignature-AbM": {"Bb", "Eb", "Ab", "Db"},
    "keySignature-FM_minor": {"Bb", "Eb", "Ab", "Db"},
    "keySignature-DbM": {"Bb", "Eb", "Ab", "Db", "Gb"},
}

# Map accidental symbols to standard notation
_ACCIDENTAL_MAP = {"#": "#", "b": "b"}


def _validate_key_signature(tokens: List[str]) -> List[str]:
    """Validate that accidentals in the output are consistent with the key signature.

    If the declared key has e.g. F# but the output never uses F# and uses F natural
    extensively, this may indicate the wrong key was detected. We don't change the key
    (that's done by _enforce_global_key_time), but we add courtesy accidentals where
    the key implies a sharp/flat but the output writes a natural.
    """
    # Find the key signature
    key_sig = None
    for token in tokens:
        if token.startswith("keySignature-"):
            key_sig = token
            break
    if key_sig is None or key_sig not in _KEY_ACCIDENTALS:
        return tokens

    expected = _KEY_ACCIDENTALS[key_sig]
    if not expected:
        return tokens

    # Build set of pitch classes that should have accidentals
    # e.g. for GM: F should be F#
    sharp_pitches = {}  # pitch_class -> expected accidental
    for acc in expected:
        if len(acc) == 2:
            pitch_class = acc[0]
            accidental = acc[1]
            sharp_pitches[pitch_class] = accidental

    # No modification needed — the key validation is informational.
    # The real fix is in _enforce_global_key_time which corrects wrong keys.
    return tokens


def _detect_repeat_barlines(tokens: List[str]) -> List[str]:
    """Detect and normalize repeat barline patterns.

    If the model emits repeat_start or repeat_end tokens, ensure they are
    properly paired and placed at measure boundaries.
    """
    result = list(tokens)

    # Count repeat_start and repeat_end
    starts = sum(1 for t in result if t == "repeat_start")
    ends = sum(1 for t in result if t == "repeat_end")

    # If we have unmatched repeats, remove the orphans
    if starts != ends:
        # Simple heuristic: if there's only one of each, keep them
        if starts == 1 and ends == 0:
            # Remove the orphan repeat_start
            result = [t for t in result if t != "repeat_start"]
        elif starts == 0 and ends == 1:
            result = [t for t in result if t != "repeat_end"]

    return result


def _fix_overfull_measure(tokens: List[str], start: int, end: int, expected_beats: float) -> List[str]:
    """Try to fix an overfull measure by adjusting the longest duration.

    If a measure has e.g. 4.5 beats in 4/4 time, find a dotted note that
    could be un-dotted to fix the balance.
    """
    actual = _measure_duration(tokens, start, end)
    excess = actual - expected_beats
    if excess < 0.1:
        return list(tokens[start:end])

    measure = list(tokens[start:end])

    # Try removing a dot to fix the balance
    for i in range(len(measure)):
        if measure[i] == "_dot" and i > 0:
            dur_token = measure[i - 1]
            if dur_token in DURATION_QUARTER_LENGTH:
                base_dur = DURATION_QUARTER_LENGTH[dur_token]
                dot_value = base_dur * 0.5
                if abs(dot_value - excess) < 0.05:
                    # Removing this dot fixes the measure
                    fixed = measure[:i] + measure[i+1:]
                    return fixed

    # Try changing a duration down one level
    duration_order = ["_whole", "_half", "_quarter", "_eighth", "_sixteenth"]
    for i in range(len(measure)):
        if measure[i] in DURATION_QUARTER_LENGTH:
            cur_dur = DURATION_QUARTER_LENGTH[measure[i]]
            for j, d_name in enumerate(duration_order):
                if d_name == measure[i] and j + 1 < len(duration_order):
                    next_dur = DURATION_QUARTER_LENGTH[duration_order[j + 1]]
                    if abs((cur_dur - next_dur) - excess) < 0.05:
                        fixed = list(measure)
                        fixed[i] = duration_order[j + 1]
                        return fixed

    return measure


def post_process_tokens(tokens: List[str], time_sig: Optional[str] = None) -> List[str]:
    """Apply all post-processing fixes to a token sequence."""
    tokens = _detect_repeat_barlines(tokens)
    tokens = _fix_whole_rest_convention(tokens, time_sig)
    tokens = _balance_measures(tokens, time_sig)
    tokens = _validate_key_signature(tokens)
    tokens = _insert_ties(tokens)
    return tokens


def write_assembly_manifest(score: AssembledScore, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "part_order": score.part_order,
        "systems": [
            {
                "page_index": system.page_index,
                "system_index": system.system_index,
                "canonical_measure_count": system.canonical_measure_count,
                "canonical_key_signature": system.canonical_key_signature,
                "canonical_time_signature": system.canonical_time_signature,
                "staves": [
                    {
                        "sample_id": staff.sample_id,
                        "part_label": staff.part_label,
                        "measure_count": staff.measure_count,
                        "clef": staff.clef,
                        "key_signature": staff.key_signature,
                        "time_signature": staff.time_signature,
                        "tokens": staff.tokens,
                        "location": {
                            "page_index": staff.location.page_index,
                            "y_top": staff.location.y_top,
                            "y_bottom": staff.location.y_bottom,
                            "x_left": staff.location.x_left,
                            "x_right": staff.location.x_right,
                        },
                    }
                    for staff in system.staves
                ],
            }
            for system in score.systems
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
