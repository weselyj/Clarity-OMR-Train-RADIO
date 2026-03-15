#!/usr/bin/env python3
"""Core evaluation metrics for OMR outputs."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


STRUCTURAL_TOKENS = {
    "<staff_start>",
    "<staff_end>",
    "<measure_start>",
    "<measure_end>",
    "<chord_start>",
    "<chord_end>",
    "<voice_1>",
    "<voice_2>",
    "<voice_3>",
    "<voice_4>",
    "barline",
    "double_barline",
    "final_barline",
    "repeat_start",
    "repeat_end",
    "repeat_both",
}

BASE_DURATION_BEATS = {
    "_whole": 4.0,
    "_half": 2.0,
    "_quarter": 1.0,
    "_eighth": 0.5,
    "_sixteenth": 0.25,
    "_thirty_second": 0.125,
    "_sixty_fourth": 0.0625,
}
TUPLET_BEAT_SCALE = {
    "<tuplet_3>": 2.0 / 3.0,
    "<tuplet_5>": 4.0 / 5.0,
    "<tuplet_6>": 4.0 / 6.0,
    "<tuplet_7>": 4.0 / 7.0,
}
DURATION_MODIFIERS = {"_dot", "_double_dot"}
VOICE_TOKENS = {"<voice_1>", "<voice_2>", "<voice_3>", "<voice_4>"}


def levenshtein_distance(a: Sequence[str], b: Sequence[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        curr = [i]
        for j, token_b in enumerate(b, start=1):
            cost = 0 if token_a == token_b else 1
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + cost,
                )
            )
        prev = curr
    return prev[-1]


def symbol_error_rate(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    if not gt_tokens:
        return 0.0 if not pred_tokens else 1.0
    return levenshtein_distance(pred_tokens, gt_tokens) / float(len(gt_tokens))


def _extract_first(tokens: Sequence[str], prefix: str) -> Optional[str]:
    for token in tokens:
        if token.startswith(prefix):
            return token
    return None


def _parse_event_stream(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    events: List[Tuple[str, str]] = []
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if token.startswith("note-"):
            duration = None
            for j in range(idx + 1, min(idx + 5, len(tokens))):
                if tokens[j].startswith("_"):
                    duration = tokens[j]
                    break
            events.append((token, duration or "_missing"))
        elif token == "rest":
            duration = None
            for j in range(idx + 1, min(idx + 5, len(tokens))):
                if tokens[j].startswith("_"):
                    duration = tokens[j]
                    break
            events.append(("rest", duration or "_missing"))
        elif token == "<chord_start>":
            chord_notes: List[str] = []
            j = idx + 1
            while j < len(tokens) and tokens[j] != "<chord_end>":
                if tokens[j].startswith("note-"):
                    chord_notes.append(tokens[j])
                j += 1
            duration = "_missing"
            if j < len(tokens):
                for k in range(j + 1, min(j + 6, len(tokens))):
                    if tokens[k].startswith("_"):
                        duration = tokens[k]
                        break
            events.append(("chord:" + "|".join(chord_notes), duration))
        idx += 1
    return events


def _serialize_events(events: Sequence[Tuple[str, str]]) -> List[str]:
    return [f"{pitch_or_kind}|{duration}" for pitch_or_kind, duration in events]


def _parse_time_signature_beats(token: str) -> Optional[float]:
    if token == "timeSignature-C":
        return 4.0
    if token == "timeSignature-C/":
        return 2.0
    match = re.fullmatch(r"timeSignature-(\d+)/(\d+)", token)
    if not match:
        return None
    numerator = int(match.group(1))
    denominator = int(match.group(2))
    if denominator <= 0:
        return None
    return float(numerator) * (4.0 / float(denominator))


def _duration_to_beats(
    duration_token: str,
    modifier_token: Optional[str],
    *,
    tuplet_scale: float,
    grace: bool,
) -> Optional[float]:
    if grace:
        return 0.0
    base = BASE_DURATION_BEATS.get(duration_token)
    if base is None:
        return None
    beats = base
    if modifier_token == "_dot":
        beats *= 1.5
    elif modifier_token == "_double_dot":
        beats *= 1.75
    return beats * max(1e-9, float(tuplet_scale))


def _extract_duration(tokens: Sequence[str], start_idx: int) -> Tuple[str, Optional[str], int]:
    limit = min(len(tokens), start_idx + 6)
    for idx in range(start_idx, limit):
        token = tokens[idx]
        if token in BASE_DURATION_BEATS:
            modifier = None
            end_idx = idx + 1
            if end_idx < len(tokens) and tokens[end_idx] in DURATION_MODIFIERS:
                modifier = tokens[end_idx]
                end_idx += 1
            return token, modifier, end_idx
    return "_missing", None, start_idx


def _normalize_onset(value: float) -> float:
    return round(float(value) + 1e-9, 3)


def _pitch_accidental(token: str) -> Optional[str]:
    if not token.startswith("note-"):
        return None
    symbol = token[len("note-") :]
    if "#" in symbol:
        return "sharp"
    if "b" in symbol:
        return "flat"
    return "natural"


def _parse_score_tokens(tokens: Sequence[str], *, fallback_beats_per_measure: float = 4.0) -> Dict[str, object]:
    events: List[Dict[str, object]] = []
    metadata = {"clef": None, "key": None, "time": None}
    beats_per_measure = float(fallback_beats_per_measure)
    in_measure = False
    measure_index = -1
    current_voice = "<voice_1>"
    voice_positions: Dict[str, float] = {current_voice: 0.0}
    beat_position = 0.0
    max_voice_beat = 0.0
    measure_count = 0
    balanced_count = 0
    saw_voice_switch = False
    pending_tuplet_scale = 1.0

    measure_is_whole_rest_only = False

    def close_measure() -> None:
        nonlocal in_measure, measure_count, balanced_count, measure_is_whole_rest_only
        if not in_measure:
            return
        measure_count += 1
        tolerance = max(0.2, 0.03 * max(1.0, beats_per_measure))
        # A whole rest filling an entire measure is the standard notation for
        # "rest for the whole measure" in any time signature — treat it as balanced.
        if measure_is_whole_rest_only:
            balanced_count += 1
        elif math.isfinite(max_voice_beat) and abs(max_voice_beat - beats_per_measure) <= tolerance:
            balanced_count += 1
        in_measure = False

    idx = 0
    while idx < len(tokens):
        token = tokens[idx]

        if token.startswith("clef-") and metadata["clef"] is None:
            metadata["clef"] = token
        elif token.startswith("keySignature-") and metadata["key"] is None:
            metadata["key"] = token
        elif token.startswith("timeSignature-"):
            if metadata["time"] is None:
                metadata["time"] = token
            parsed_beats = _parse_time_signature_beats(token)
            if parsed_beats is not None:
                beats_per_measure = parsed_beats

        if token == "<measure_start>":
            close_measure()
            in_measure = True
            measure_is_whole_rest_only = True
            measure_index += 1
            current_voice = "<voice_1>"
            voice_positions = {current_voice: 0.0}
            beat_position = 0.0
            max_voice_beat = 0.0
            pending_tuplet_scale = 1.0
            idx += 1
            continue

        if token == "<measure_end>":
            close_measure()
            idx += 1
            continue

        if token in VOICE_TOKENS:
            if in_measure:
                voice_positions[current_voice] = beat_position
                current_voice = token
                beat_position = float(voice_positions.get(current_voice, 0.0))
                saw_voice_switch = True
            idx += 1
            continue

        if token in TUPLET_BEAT_SCALE:
            pending_tuplet_scale = TUPLET_BEAT_SCALE[token]
            idx += 1
            continue

        if token == "<chord_start>":
            measure_is_whole_rest_only = False
            notes: List[str] = []
            j = idx + 1
            while j < len(tokens) and tokens[j] != "<chord_end>":
                if tokens[j].startswith("note-"):
                    notes.append(tokens[j])
                j += 1
            if j >= len(tokens):
                idx += 1
                continue
            duration_token, modifier_token, next_idx = _extract_duration(tokens, j + 1)
            beats = _duration_to_beats(
                duration_token,
                modifier_token,
                tuplet_scale=pending_tuplet_scale,
                grace=False,
            )
            onset = _normalize_onset(beat_position)
            sorted_notes = tuple(sorted(notes))
            events.append(
                {
                    "kind": "chord",
                    "measure": measure_index,
                    "voice": current_voice,
                    "onset": onset,
                    "duration": duration_token,
                    "notes": sorted_notes,
                }
            )
            if in_measure and beats is not None:
                beat_position += beats
                voice_positions[current_voice] = beat_position
                max_voice_beat = max(max_voice_beat, beat_position)
            pending_tuplet_scale = 1.0
            idx = max(next_idx, j + 1)
            continue

        if token.startswith("note-") or token.startswith("gracenote-") or token == "rest":
            is_grace = token.startswith("gracenote-")
            duration_token, modifier_token, next_idx = _extract_duration(tokens, idx + 1)
            # A measure is whole-rest-only if it contains exactly one rest
            # with _whole duration and nothing else (standard full-measure rest).
            if token == "rest" and duration_token == "_whole" and beat_position == 0.0:
                pass  # keep measure_is_whole_rest_only as True
            else:
                measure_is_whole_rest_only = False
            beats = _duration_to_beats(
                duration_token,
                modifier_token,
                tuplet_scale=pending_tuplet_scale,
                grace=is_grace,
            )
            onset = _normalize_onset(beat_position)
            notes_tuple = (token,) if token.startswith("note-") else tuple()
            events.append(
                {
                    "kind": ("grace" if is_grace else ("rest" if token == "rest" else "note")),
                    "token": token,
                    "measure": measure_index,
                    "voice": current_voice,
                    "onset": onset,
                    "duration": duration_token,
                    "notes": notes_tuple,
                }
            )
            if in_measure and beats is not None and not is_grace:
                beat_position += beats
                voice_positions[current_voice] = beat_position
                max_voice_beat = max(max_voice_beat, beat_position)
            pending_tuplet_scale = 1.0
            idx = max(next_idx, idx + 1)
            continue

        idx += 1

    close_measure()
    measure_balance = (balanced_count / float(measure_count)) if measure_count > 0 else 0.0
    return {
        "events": events,
        "metadata": metadata,
        "measure_count": measure_count,
        "balanced_measures": balanced_count,
        "measure_balance_rate": measure_balance,
        "saw_voice_switch": saw_voice_switch,
    }


def _note_event_labels(parsed: Dict[str, object]) -> List[str]:
    labels: List[str] = []
    for event in parsed["events"]:
        kind = str(event.get("kind", ""))
        duration = str(event.get("duration", "_missing"))
        if kind == "note":
            labels.append(f"{event.get('token', 'note')}|{duration}")
        elif kind == "rest":
            labels.append(f"rest|{duration}")
        elif kind == "chord":
            notes = "/".join(str(token) for token in event.get("notes", ()))
            labels.append(f"chord:{notes}|{duration}")
    return labels


def _onset_labels(parsed: Dict[str, object]) -> List[str]:
    return [
        f"m{event['measure']}|v{event['voice']}|o{event['onset']}"
        for event in parsed["events"]
        if event["kind"] in {"note", "chord", "rest"}
    ]


def _chord_labels(parsed: Dict[str, object]) -> List[str]:
    return [
        f"m{event['measure']}|v{event['voice']}|o{event['onset']}|{'/'.join(event['notes'])}"
        for event in parsed["events"]
        if event["kind"] == "chord"
    ]


def _accidental_labels(parsed: Dict[str, object]) -> List[str]:
    labels: List[str] = []
    for event in parsed["events"]:
        if event["kind"] == "note":
            accidental = _pitch_accidental(str(event.get("token", "")))
            if accidental is not None:
                labels.append(accidental)
        elif event["kind"] == "chord":
            for note_token in event.get("notes", ()):
                accidental = _pitch_accidental(str(note_token))
                if accidental is not None:
                    labels.append(accidental)
    return labels


def _voice_labels(parsed: Dict[str, object]) -> List[str]:
    return [
        str(event["voice"])
        for event in parsed["events"]
        if event["kind"] in {"note", "chord", "rest"}
    ]


def _fallback_beats_from_gt(gt_parsed: Dict[str, object]) -> float:
    fallback_beats = 4.0
    gt_time = gt_parsed.get("metadata", {}).get("time")
    if isinstance(gt_time, str):
        parsed_beats = _parse_time_signature_beats(gt_time)
        if parsed_beats is not None:
            fallback_beats = parsed_beats
    return fallback_beats


def _note_event_accuracy_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    return _sequence_alignment_accuracy(_note_event_labels(pred), _note_event_labels(gt))


def _note_event_f1_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    _, _, f1 = _multiset_prf(_note_event_labels(pred), _note_event_labels(gt))
    return f1


def _onset_accuracy_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    return _sequence_alignment_accuracy(_onset_labels(pred), _onset_labels(gt))


def _onset_f1_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    _, _, f1 = _multiset_prf(_onset_labels(pred), _onset_labels(gt))
    return f1


def _chord_note_f1_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    pred_items = _chord_labels(pred)
    gt_items = _chord_labels(gt)
    if not gt_items and not pred_items:
        return 1.0
    _, _, f1 = _multiset_prf(pred_items, gt_items)
    return f1


def _accidental_accuracy_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    return _sequence_alignment_accuracy(_accidental_labels(pred), _accidental_labels(gt))


def _measure_balance_rate_from_parsed(pred: Dict[str, object]) -> float:
    return float(pred.get("measure_balance_rate", 0.0))


def _metadata_presence_rate_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    gt_meta = gt.get("metadata", {})
    pred_meta = pred.get("metadata", {})
    required = [key for key in ("clef", "key", "time") if gt_meta.get(key)]
    if not required:
        return 1.0
    hits = sum(1 for key in required if pred_meta.get(key))
    return float(hits) / float(len(required))


def _voice_assignment_accuracy_from_parsed(pred: Dict[str, object], gt: Dict[str, object]) -> float:
    gt_voices = _voice_labels(gt)
    if len(set(gt_voices)) <= 1:
        return 1.0
    pred_voices = _voice_labels(pred)
    return _sequence_alignment_accuracy(pred_voices, gt_voices)


def _sequence_alignment_accuracy(pred_items: Sequence[str], gt_items: Sequence[str]) -> float:
    if not gt_items:
        return 1.0 if not pred_items else 0.0
    distance = levenshtein_distance(pred_items, gt_items)
    return max(0.0, 1.0 - (distance / float(len(gt_items))))


def pitch_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    pred_events = _parse_event_stream(pred_tokens)
    gt_events = _parse_event_stream(gt_tokens)
    pred_pitches = [event[0] for event in pred_events]
    gt_pitches = [event[0] for event in gt_events]
    return _sequence_alignment_accuracy(pred_pitches, gt_pitches)


def rhythm_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    pred_events = _parse_event_stream(pred_tokens)
    gt_events = _parse_event_stream(gt_tokens)
    pred_durations = [event[1] for event in pred_events]
    gt_durations = [event[1] for event in gt_events]
    return _sequence_alignment_accuracy(pred_durations, gt_durations)


def note_event_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _note_event_accuracy_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def note_event_f1(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _note_event_f1_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def onset_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _onset_accuracy_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def onset_f1(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _onset_f1_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def chord_note_f1(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _chord_note_f1_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def accidental_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _accidental_accuracy_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def measure_balance_rate(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    gt_parsed = _parse_score_tokens(gt_tokens)
    pred_parsed = _parse_score_tokens(
        pred_tokens,
        fallback_beats_per_measure=_fallback_beats_from_gt(gt_parsed),
    )
    return _measure_balance_rate_from_parsed(pred_parsed)


def metadata_presence_rate(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _metadata_presence_rate_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def voice_assignment_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    return _voice_assignment_accuracy_from_parsed(
        _parse_score_tokens(pred_tokens),
        _parse_score_tokens(gt_tokens),
    )


def key_time_signature_accuracy(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    pred_key = _extract_first(pred_tokens, "keySignature-")
    gt_key = _extract_first(gt_tokens, "keySignature-")
    pred_time = _extract_first(pred_tokens, "timeSignature-")
    gt_time = _extract_first(gt_tokens, "timeSignature-")
    key_ok = pred_key == gt_key
    time_ok = pred_time == gt_time
    return 1.0 if (key_ok and time_ok) else 0.0


def _multiset_prf(pred_items: Sequence[str], gt_items: Sequence[str]) -> Tuple[float, float, float]:
    pred_counter = Counter(pred_items)
    gt_counter = Counter(gt_items)
    tp = sum(min(pred_counter[token], gt_counter[token]) for token in pred_counter)
    pred_total = sum(pred_counter.values())
    gt_total = sum(gt_counter.values())
    precision = tp / pred_total if pred_total else 0.0
    recall = tp / gt_total if gt_total else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    return precision, recall, 2 * precision * recall / (precision + recall)


def structural_f1(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> float:
    pred_structural = [token for token in pred_tokens if token in STRUCTURAL_TOKENS]
    gt_structural = [token for token in gt_tokens if token in STRUCTURAL_TOKENS]
    _, _, f1 = _multiset_prf(pred_structural, gt_structural)
    return f1


def quality_score(metrics: Dict[str, object]) -> Dict[str, object]:
    ser = max(0.0, float(metrics.get("ser", 0.0)))
    ser_accuracy = max(0.0, 1.0 - ser)
    note_acc = max(0.0, min(1.0, float(metrics.get("note_event_accuracy", 0.0))))
    note_f1_value = max(0.0, min(1.0, float(metrics.get("note_event_f1", 0.0))))
    pitch = max(0.0, min(1.0, float(metrics.get("pitch_accuracy", 0.0))))
    rhythm = max(0.0, min(1.0, float(metrics.get("rhythm_accuracy", 0.0))))
    onset_acc = max(0.0, min(1.0, float(metrics.get("onset_accuracy", 0.0))))
    onset_f1_value = max(0.0, min(1.0, float(metrics.get("onset_f1", 0.0))))
    chord_f1_value = max(0.0, min(1.0, float(metrics.get("chord_note_f1", 0.0))))
    accidental = max(0.0, min(1.0, float(metrics.get("accidental_accuracy", 0.0))))
    measure_balance = max(0.0, min(1.0, float(metrics.get("measure_balance_rate", 0.0))))
    metadata_presence = max(0.0, min(1.0, float(metrics.get("metadata_presence_rate", 0.0))))
    voice_assignment = max(0.0, min(1.0, float(metrics.get("voice_assignment_accuracy", 0.0))))
    key_time = max(0.0, min(1.0, float(metrics.get("key_time_accuracy", 0.0))))
    structural = max(0.0, min(1.0, float(metrics.get("structural_f1", 0.0))))

    # Weighted composite targeting practical OMR usefulness.
    score_0_1 = (
        0.23 * ser_accuracy
        + 0.20 * note_f1_value
        + 0.15 * note_acc
        + 0.10 * onset_f1_value
        + 0.08 * onset_acc
        + 0.08 * chord_f1_value
        + 0.05 * accidental
        + 0.04 * measure_balance
        + 0.03 * metadata_presence
        + 0.01 * voice_assignment
        + 0.02 * pitch
        + 0.01 * rhythm
    )

    penalty = 1.0
    if metadata_presence < 0.5:
        penalty *= 0.80
    elif metadata_presence < 0.8:
        penalty *= 0.90
    if measure_balance < 0.6:
        penalty *= 0.80
    elif measure_balance < 0.8:
        penalty *= 0.90
    if key_time < 0.5:
        penalty *= 0.90
    if structural < 0.6:
        penalty *= 0.90

    score = max(0.0, min(100.0, 100.0 * score_0_1 * penalty))

    if score >= 90.0:
        rating = "excellent"
    elif score >= 80.0:
        rating = "good"
    elif score >= 70.0:
        rating = "usable"
    elif score >= 60.0:
        rating = "weak"
    else:
        rating = "poor"

    return {
        "score": score,
        "rating": rating,
        "is_good": bool(score >= 80.0),
        "penalty_factor": penalty,
        "base_score": 100.0 * score_0_1,
        "score_components": {
            "ser_accuracy": ser_accuracy,
            "note_event_accuracy": note_acc,
            "note_event_f1": note_f1_value,
            "onset_accuracy": onset_acc,
            "onset_f1": onset_f1_value,
            "chord_note_f1": chord_f1_value,
            "accidental_accuracy": accidental,
            "measure_balance_rate": measure_balance,
            "metadata_presence_rate": metadata_presence,
            "voice_assignment_accuracy": voice_assignment,
            "pitch_accuracy": pitch,
            "rhythm_accuracy": rhythm,
            "key_time_accuracy": key_time,
            "structural_f1": structural,
        },
    }


def musicxml_validity(paths: Sequence[str]) -> float:
    if not paths:
        return 0.0
    try:
        from music21 import converter
        from music21.exceptions21 import Music21Exception
    except ImportError as exc:
        raise RuntimeError("music21 is required for MusicXML validity checks.") from exc

    valid = 0
    for path in paths:
        try:
            converter.parse(path)
        except (FileNotFoundError, OSError, ValueError, TypeError, Music21Exception):
            continue
        valid += 1
    return valid / float(len(paths))


def musicxml_musical_similarity(
    pairs: Sequence[Tuple[str, str]],
) -> Dict[str, Optional[float] | int]:
    """Compare predicted vs reference MusicXML files using note-level musical metrics.

    Each pair is ``(predicted_musicxml_path, reference_musicxml_path)``.
    Uses *mir_eval* transcription metrics (precision, recall, F1) to measure
    how accurately the predicted score reproduces the reference notes.
    """
    empty: Dict[str, Optional[float] | int] = {
        "musical_samples": 0,
        "musical_precision": None,
        "musical_recall": None,
        "musical_f1": None,
        "musical_overlap": None,
        "musical_onset_precision": None,
        "musical_onset_recall": None,
        "musical_onset_f1": None,
    }
    if not pairs:
        return empty

    try:
        import mir_eval
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("mir_eval and numpy are required for musical similarity metrics.") from exc

    from src.pipeline.export_musicxml import _extract_note_events, _get_bpm, _require_music21

    _, _, converter, _, _, _, _ = _require_music21()

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    overlaps: List[float] = []
    onset_precisions: List[float] = []
    onset_recalls: List[float] = []
    onset_f1s: List[float] = []

    for pred_musicxml_path, ref_musicxml_path in pairs:
        pred_path = Path(pred_musicxml_path)
        ref_path = Path(ref_musicxml_path)
        if not pred_path.is_absolute():
            pred_path = (Path.cwd() / pred_path).resolve()
        if not ref_path.is_absolute():
            ref_path = (Path.cwd() / ref_path).resolve()
        if not pred_path.exists() or not ref_path.exists():
            continue

        try:
            pred_score = converter.parse(str(pred_path))
            ref_score = converter.parse(str(ref_path))
        except Exception:
            continue

        ref_bpm = _get_bpm(ref_score)
        est_intervals, est_pitches = _extract_note_events(pred_score, bpm=ref_bpm)
        ref_intervals, ref_pitches = _extract_note_events(ref_score, bpm=ref_bpm)

        if ref_intervals.shape[0] == 0 or est_intervals.shape[0] == 0:
            precisions.append(0.0)
            recalls.append(0.0)
            f1s.append(0.0)
            overlaps.append(0.0)
            onset_precisions.append(0.0)
            onset_recalls.append(0.0)
            onset_f1s.append(0.0)
            continue

        prec, rec, f1, overlap = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0,
        )
        oprec, orec, of1, _ = mir_eval.transcription.precision_recall_f1_overlap(
            ref_intervals, ref_pitches, est_intervals, est_pitches,
            onset_tolerance=0.05, pitch_tolerance=50.0, offset_ratio=None,
        )

        precisions.append(float(prec))
        recalls.append(float(rec))
        f1s.append(float(f1))
        overlaps.append(float(overlap))
        onset_precisions.append(float(oprec))
        onset_recalls.append(float(orec))
        onset_f1s.append(float(of1))

    if not precisions:
        return empty

    n = len(precisions)
    return {
        "musical_samples": n,
        "musical_precision": sum(precisions) / n,
        "musical_recall": sum(recalls) / n,
        "musical_f1": sum(f1s) / n,
        "musical_overlap": sum(overlaps) / n,
        "musical_onset_precision": sum(onset_precisions) / n,
        "musical_onset_recall": sum(onset_recalls) / n,
        "musical_onset_f1": sum(onset_f1s) / n,
    }


@dataclass(frozen=True)
class SequenceMetrics:
    ser: float
    pitch_accuracy: float
    rhythm_accuracy: float
    note_event_accuracy: float
    note_event_f1: float
    onset_accuracy: float
    onset_f1: float
    chord_note_f1: float
    accidental_accuracy: float
    measure_balance_rate: float
    metadata_presence_rate: float
    voice_assignment_accuracy: float
    key_time_accuracy: float
    structural_f1: float


def evaluate_pair(pred_tokens: Sequence[str], gt_tokens: Sequence[str]) -> SequenceMetrics:
    gt_parsed = _parse_score_tokens(gt_tokens)
    pred_parsed = _parse_score_tokens(
        pred_tokens,
        fallback_beats_per_measure=_fallback_beats_from_gt(gt_parsed),
    )
    return SequenceMetrics(
        ser=symbol_error_rate(pred_tokens, gt_tokens),
        pitch_accuracy=pitch_accuracy(pred_tokens, gt_tokens),
        rhythm_accuracy=rhythm_accuracy(pred_tokens, gt_tokens),
        note_event_accuracy=_note_event_accuracy_from_parsed(pred_parsed, gt_parsed),
        note_event_f1=_note_event_f1_from_parsed(pred_parsed, gt_parsed),
        onset_accuracy=_onset_accuracy_from_parsed(pred_parsed, gt_parsed),
        onset_f1=_onset_f1_from_parsed(pred_parsed, gt_parsed),
        chord_note_f1=_chord_note_f1_from_parsed(pred_parsed, gt_parsed),
        accidental_accuracy=_accidental_accuracy_from_parsed(pred_parsed, gt_parsed),
        measure_balance_rate=_measure_balance_rate_from_parsed(pred_parsed),
        metadata_presence_rate=_metadata_presence_rate_from_parsed(pred_parsed, gt_parsed),
        voice_assignment_accuracy=_voice_assignment_accuracy_from_parsed(pred_parsed, gt_parsed),
        key_time_accuracy=key_time_signature_accuracy(pred_tokens, gt_tokens),
        structural_f1=structural_f1(pred_tokens, gt_tokens),
    )


def aggregate_metrics(pairs: Iterable[Tuple[Sequence[str], Sequence[str]]]) -> Dict[str, object]:
    values = [
        evaluate_pair(pred_tokens=pred_tokens, gt_tokens=gt_tokens)
        for pred_tokens, gt_tokens in pairs
    ]
    if not values:
        return {
            "ser": 0.0,
            "pitch_accuracy": 0.0,
            "rhythm_accuracy": 0.0,
            "note_event_accuracy": 0.0,
            "note_event_f1": 0.0,
            "onset_accuracy": 0.0,
            "onset_f1": 0.0,
            "chord_note_f1": 0.0,
            "accidental_accuracy": 0.0,
            "measure_balance_rate": 0.0,
            "metadata_presence_rate": 0.0,
            "voice_assignment_accuracy": 0.0,
            "key_time_accuracy": 0.0,
            "structural_f1": 0.0,
            "quality": quality_score({}),
        }
    total = len(values)
    aggregated = {
        "ser": sum(item.ser for item in values) / total,
        "pitch_accuracy": sum(item.pitch_accuracy for item in values) / total,
        "rhythm_accuracy": sum(item.rhythm_accuracy for item in values) / total,
        "note_event_accuracy": sum(item.note_event_accuracy for item in values) / total,
        "note_event_f1": sum(item.note_event_f1 for item in values) / total,
        "onset_accuracy": sum(item.onset_accuracy for item in values) / total,
        "onset_f1": sum(item.onset_f1 for item in values) / total,
        "chord_note_f1": sum(item.chord_note_f1 for item in values) / total,
        "accidental_accuracy": sum(item.accidental_accuracy for item in values) / total,
        "measure_balance_rate": sum(item.measure_balance_rate for item in values) / total,
        "metadata_presence_rate": sum(item.metadata_presence_rate for item in values) / total,
        "voice_assignment_accuracy": sum(item.voice_assignment_accuracy for item in values) / total,
        "key_time_accuracy": sum(item.key_time_accuracy for item in values) / total,
        "structural_f1": sum(item.structural_f1 for item in values) / total,
    }
    aggregated["quality"] = quality_score(aggregated)
    return aggregated


def default_ablation_matrix() -> List[Dict[str, str]]:
    return [
        {"id": "ablation-1", "name": "DoRA vs LoRA vs full", "question": "Does DoRA outperform alternatives?"},
        {"id": "ablation-2", "name": "Target module scope", "question": "All linear vs attention-only vs MLP-only"},
        {"id": "ablation-3", "name": "Tokenization strategy", "question": "Hybrid vs semantic vs character-level"},
        {"id": "ablation-4", "name": "Curriculum ordering", "question": "3-stage curriculum vs mixed-from-start"},
        {"id": "ablation-5", "name": "Constrained decoding", "question": "With vs without grammar FSA"},
        {"id": "ablation-6", "name": "Deformable block", "question": "Impact of deformable attention layer"},
        {"id": "ablation-7", "name": "Decoder position encoding", "question": "RoPE vs learned embeddings"},
    ]
