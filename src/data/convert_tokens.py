#!/usr/bin/env python3
"""Convert manifest source labels (MEI/Kern/MusicXML/semantic) into token sequences."""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from src.tokenizer.vocab import (
    CLEF_TOKENS,
    DYNAMIC_TOKENS,
    EXPRESSION_TOKENS,
    TEMPO_TOKENS,
    TIME_SIGNATURE_TOKENS,
    build_pitch_tokens,
    build_gracenote_tokens,
    build_key_signature_tokens,
)


DURATION_BY_NAME = {
    "whole": "_whole",
    "half": "_half",
    "quarter": "_quarter",
    "eighth": "_eighth",
    "sixteenth": "_sixteenth",
    "thirty_second": "_thirty_second",
    "sixty_fourth": "_sixty_fourth",
}

MEI_DURATION_TO_NAME = {
    "1": "whole",
    "2": "half",
    "4": "quarter",
    "8": "eighth",
    "16": "sixteenth",
    "32": "thirty_second",
    "64": "sixty_fourth",
}

KERN_DURATION_TO_NAME = dict(MEI_DURATION_TO_NAME)

MEI_NS = {"mei": "http://www.music-encoding.org/ns/mei"}

KEY_SIGNATURE_MAJOR_BY_FIFTHS = {
    -7: "Cb",
    -6: "Gb",
    -5: "Db",
    -4: "Ab",
    -3: "Eb",
    -2: "Bb",
    -1: "F",
    0: "C",
    1: "G",
    2: "D",
    3: "A",
    4: "E",
    5: "B",
    6: "F#",
    7: "C#",
}

ARTICULATION_CLASS_TO_TOKEN = {
    "Staccato": "staccato",
    "Accent": "accent",
    "Tenuto": "tenuto",
    "StrongAccent": "marcato",
    "Staccatissimo": "staccatissimo",
    "DetachedLegato": "portato",
    "SnapPizzicato": "snap_pizz",
}

EXPRESSION_CLASS_TO_TOKEN = {
    "Fermata": "fermata",
    "Trill": "trill",
    "Mordent": "mordent",
    "Turn": "turn",
}

DYNAMIC_VALUE_TO_TOKEN = {
    token[len("dynamic-") :].lower(): token for token in DYNAMIC_TOKENS if token.startswith("dynamic-")
}
TEMPO_TEXT_TO_TOKEN = {
    token[len("tempo-") :].replace("_", " ").lower(): token for token in TEMPO_TOKENS
}
EXPRESSION_TEXT_TO_TOKEN = {
    token[len("expr-") :].replace("_", " ").lower(): token for token in EXPRESSION_TOKENS
}
SUPPORTED_CLEF_TOKENS = set(CLEF_TOKENS)
SUPPORTED_KEY_SIGNATURE_TOKENS = set(build_key_signature_tokens())
SUPPORTED_TIME_SIGNATURE_TOKENS = set(TIME_SIGNATURE_TOKENS)
SUPPORTED_NOTE_TOKENS = {token for token in build_pitch_tokens() if token.startswith("note-")}
SUPPORTED_GRACE_TOKENS = set(build_gracenote_tokens())
MAX_SUPPORTED_VOICE_INDEX = 4


def relpath(project_root: Path, target: Path) -> str:
    return str(target.resolve().relative_to(project_root.resolve())).replace("\\", "/")


def load_manifest_entries(manifest_path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {manifest_path}:{line_no}") from exc
    return entries


def resolve_optional_path(project_root: Path, value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    return (project_root / Path(value)).resolve()


def _split_staff_sequences_for_validation(token_sequence: Sequence[str]) -> List[List[str]]:
    if not token_sequence:
        return []
    if token_sequence.count("<staff_start>") <= 1:
        return [list(token_sequence)]

    staff_sequences: List[List[str]] = []
    idx = 0
    while idx < len(token_sequence):
        if token_sequence[idx] != "<staff_start>":
            idx += 1
            continue
        end_idx = idx + 1
        while end_idx < len(token_sequence) and token_sequence[end_idx] != "<staff_end>":
            end_idx += 1
        if end_idx >= len(token_sequence):
            raise ValueError("Malformed token sequence: missing <staff_end>.")
        staff_sequences.append(["<bos>", *token_sequence[idx : end_idx + 1], "<eos>"])
        idx = end_idx + 1
    return staff_sequences


def _get_grammar_validator():
    from src.decoding.grammar_fsa import GrammarFSA

    validator = getattr(_get_grammar_validator, "_validator", None)
    if validator is None:
        validator = GrammarFSA()
        setattr(_get_grammar_validator, "_validator", validator)
    return validator


def validate_token_sequence(token_sequence: Sequence[str], strict: bool = True) -> None:
    sequences = _split_staff_sequences_for_validation(token_sequence)
    if not sequences:
        raise ValueError("Empty token sequence cannot be validated.")

    validator = _get_grammar_validator()
    for sequence in sequences:
        validator.validate_sequence(sequence, strict=strict)


def normalize_duration_name(raw: str) -> Tuple[str, int]:
    token = raw.strip().lower().replace("-", "_")
    prefix_aliases = {
        "quadruple_whole": "whole",
        "triple_whole": "whole",
        "double_whole": "whole",
        "breve": "whole",
        "longa": "whole",
        "maxima": "whole",
        "one_hundred_twenty_eighth": "hundred_twenty_eighth",
        "hundred_twenty_eighth": "sixty_fourth",
    }
    for source_prefix, target_prefix in prefix_aliases.items():
        if token.startswith(source_prefix):
            token = target_prefix + token[len(source_prefix) :]
            break
    aliases = {
        "16th": "sixteenth",
        "32nd": "thirty_second",
        "64th": "sixty_fourth",
        "128th": "hundred_twenty_eighth",
        "one_hundred_twenty_eighth": "sixty_fourth",
        "hundred_twenty_eighth": "sixty_fourth",
    }
    token = aliases.get(token, token)

    known = [
        "sixty_fourth",
        "thirty_second",
        "sixteenth",
        "quarter",
        "eighth",
        "whole",
        "half",
    ]
    for duration_name in known:
        if not (
            token == duration_name
            or token.startswith(duration_name + ".")
            or token.startswith(duration_name + "_")
        ):
            continue
        remainder = token[len(duration_name) :]
        dots = 0
        if remainder.startswith(".."):
            dots = 2
        elif remainder.startswith("."):
            dots = 1
        if "_double_dot" in remainder:
            dots = 2
        elif "_dot" in remainder:
            dots = max(dots, 1)
        return duration_name, dots

    return token, 0


def duration_tokens(duration_name: str, dots: int, is_rest: bool) -> List[str]:
    if duration_name not in DURATION_BY_NAME:
        raise ValueError(f"Unsupported duration '{duration_name}'")
    tokens = [DURATION_BY_NAME[duration_name]]
    if dots == 1:
        tokens.append("_dot")
    elif dots >= 2:
        tokens.append("_double_dot")
    return tokens


def parse_note_duration_token(token: str, prefix: str) -> Tuple[str, List[str]]:
    body = token[len(prefix) :]
    if prefix == "rest-":
        duration_name, dots = normalize_duration_name(body)
        return "rest", duration_tokens(duration_name, dots=dots, is_rest=True)
    if "_" not in body:
        raise ValueError(f"Expected duration separator in token '{token}'")
    pitch_or_rest, duration_raw = body.split("_", 1)
    duration_name, dots = normalize_duration_name(duration_raw)
    return pitch_or_rest, duration_tokens(duration_name, dots=dots, is_rest=False)


def convert_semantic_file(path: Path) -> List[str]:
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        raise ValueError(f"Semantic file is empty: {path}")
    semantic_tokens = raw.replace("\t", " ").split()

    converted: List[str] = ["<bos>", "<staff_start>"]
    measure_open = False

    def ensure_measure_open() -> None:
        nonlocal measure_open
        if not measure_open:
            converted.append("<measure_start>")
            measure_open = True

    for idx, token in enumerate(semantic_tokens):
        if token.startswith("clef-") or token.startswith("keySignature-") or token.startswith("timeSignature-"):
            if measure_open:
                converted.append("<measure_end>")
                measure_open = False
            normalized: Optional[str]
            if token.startswith("clef-"):
                normalized = _normalize_clef_token(token)
            elif token.startswith("keySignature-"):
                normalized = _normalize_key_signature_token(token)
            else:
                normalized = _normalize_time_signature_token(token)
            if normalized is not None:
                converted.append(normalized)
            continue

        if token == "barline":
            if measure_open:
                converted.append("<measure_end>")
                measure_open = False
            continue
        if token.startswith("multirest-"):
            count_raw = token[len("multirest-") :]
            try:
                count = max(1, int(count_raw))
            except ValueError:
                raise ValueError(f"Invalid multirest token '{token}'")
            for rest_idx in range(count):
                ensure_measure_open()
                converted.extend(["rest", "_whole", "<measure_end>"])
                measure_open = False
                is_last_measure = rest_idx == count - 1
                if not is_last_measure:
                    ensure_measure_open()
            continue
        if token.startswith("note-"):
            ensure_measure_open()
            pitch, duration = parse_note_duration_token(token, "note-")
            prefer_flats = "b" in pitch and "#" not in pitch
            normalized = _normalize_pitch_symbol(pitch, prefer_flats=prefer_flats)
            converted.append(f"note-{_normalize_note_pitch_symbol(normalized)}")
            converted.extend(duration)
            continue
        if token.startswith("gracenote-"):
            ensure_measure_open()
            pitch, duration = parse_note_duration_token(token, "gracenote-")
            converted.append(f"gracenote-{_normalize_grace_pitch_symbol(pitch)}")
            converted.extend(duration)
            continue
        if token.startswith("rest-"):
            ensure_measure_open()
            rest_token, duration = parse_note_duration_token(token, "rest-")
            converted.append(rest_token)
            converted.extend(duration)
            continue
        if token == "tie":
            ensure_measure_open()
            converted.append("tie_start")
            continue
        ensure_measure_open()
        converted.append(token)

    if measure_open:
        converted.append("<measure_end>")
    converted.extend(["<staff_end>", "<eos>"])
    return converted


def mei_key_signature_token(key_sig: Optional[str]) -> str:
    if not key_sig:
        return _normalize_key_signature_token("keySignature-CM")
    if key_sig == "0":
        return _normalize_key_signature_token("keySignature-CM")
    match = re.fullmatch(r"([0-7])([fs])", key_sig.strip())
    if not match:
        return _normalize_key_signature_token("keySignature-none")
    count = int(match.group(1))
    direction = 1 if match.group(2) == "s" else -1
    fifths = direction * count
    tonic = KEY_SIGNATURE_MAJOR_BY_FIFTHS.get(fifths, "C")
    return _normalize_key_signature_token(f"keySignature-{tonic}M")


def mei_note_pitch(note_element: ET.Element) -> str:
    pname = note_element.attrib.get("pname")
    octv = note_element.attrib.get("oct")
    if not pname or not octv:
        raise ValueError("MEI note missing pname/oct attributes")
    accid_value = note_element.attrib.get("accid", "")
    accid_map = {"s": "#", "ss": "##", "f": "b", "ff": "bb", "n": ""}
    accidental = accid_map.get(accid_value, "")
    return f"{pname.upper()}{accidental}{octv}"


def convert_mei_file(path: Path) -> List[str]:
    tree = ET.parse(path)
    root = tree.getroot()
    score_def = root.find(".//mei:scoreDef", MEI_NS)
    staff_def = root.find(".//mei:staffDef", MEI_NS)

    clef_shape = "G"
    clef_line = "2"
    if staff_def is not None:
        clef_shape = staff_def.attrib.get("clef.shape", clef_shape)
        clef_line = staff_def.attrib.get("clef.line", clef_line)
    key_sig = score_def.attrib.get("key.sig") if score_def is not None else None
    meter_count = score_def.attrib.get("meter.count") if score_def is not None else None
    meter_unit = score_def.attrib.get("meter.unit") if score_def is not None else None

    tokens: List[str] = [
        "<bos>",
        "<staff_start>",
        _normalize_clef_token(f"clef-{clef_shape}{clef_line}") or "clef-G2",
        _normalize_key_signature_token(mei_key_signature_token(key_sig)),
    ]
    if meter_count and meter_unit:
        tokens.append(_normalize_time_signature_token(f"timeSignature-{meter_count}/{meter_unit}"))

    measures = root.findall(".//mei:measure", MEI_NS)
    if not measures:
        raise ValueError(f"No measures found in MEI file: {path}")

    for measure in measures:
        tokens.append("<measure_start>")
        for element in measure.iter():
            tag = element.tag.split("}", 1)[-1]
            if tag == "note":
                duration_raw = element.attrib.get("dur")
                if not duration_raw:
                    raise ValueError("MEI note missing dur attribute")
                duration_name = MEI_DURATION_TO_NAME.get(duration_raw)
                if duration_name is None:
                    raise ValueError(f"Unsupported MEI duration '{duration_raw}'")
                dots = int(element.attrib.get("dots", "0"))
                mei_pitch = mei_note_pitch(element)
                prefer_flats = "b" in mei_pitch and "#" not in mei_pitch
                normalized = _normalize_pitch_symbol(mei_pitch, prefer_flats=prefer_flats)
                tokens.append(f"note-{_normalize_note_pitch_symbol(normalized)}")
                tokens.extend(duration_tokens(duration_name, dots=dots, is_rest=False))
            elif tag == "rest":
                duration_raw = element.attrib.get("dur")
                if not duration_raw:
                    raise ValueError("MEI rest missing dur attribute")
                duration_name = MEI_DURATION_TO_NAME.get(duration_raw)
                if duration_name is None:
                    raise ValueError(f"Unsupported MEI rest duration '{duration_raw}'")
                dots = int(element.attrib.get("dots", "0"))
                tokens.append("rest")
                tokens.extend(duration_tokens(duration_name, dots=dots, is_rest=True))
        tokens.append("<measure_end>")

    tokens.extend(["<staff_end>", "<eos>"])
    return tokens


def kern_clef_token(cell: str) -> Optional[str]:
    if not cell.startswith("*clef"):
        return None
    clef = cell[len("*clef") :]
    if not clef:
        return None
    return _normalize_clef_token(f"clef-{clef}")


def kern_key_signature_token(cell: str) -> Optional[str]:
    if not cell.startswith("*k["):
        return None
    inside = cell[cell.find("[") + 1 : cell.rfind("]")]
    if not inside:
        return "keySignature-CM"
    accidentals = re.findall(r"[A-Ga-g][#-]", inside)
    if not accidentals:
        return "keySignature-CM"
    accidental_symbols = {acc[-1] for acc in accidentals}
    if len(accidental_symbols) != 1:
        return "keySignature-none"
    symbol = accidental_symbols.pop()
    count = len(accidentals)
    if symbol == "#":
        fifths = count
    elif symbol == "-":
        fifths = -count
    else:
        return "keySignature-none"
    tonic = KEY_SIGNATURE_MAJOR_BY_FIFTHS.get(fifths, "C")
    return _normalize_key_signature_token(f"keySignature-{tonic}M")


def kern_time_signature_token(cell: str) -> Optional[str]:
    if not cell.startswith("*M"):
        return None
    meter = cell[len("*M") :]
    if not meter:
        return None
    return _normalize_time_signature_token(f"timeSignature-{meter}")


def kern_pitch_token(body: str) -> str:
    match = re.search(r"[A-Ga-g]+", body)
    if not match:
        raise ValueError(f"Could not parse Kern pitch from '{body}'")
    letters = match.group(0)
    base = letters[0]
    count = len(letters)
    octave = 4 + (count - 1) if base.islower() else 3 - (count - 1)

    accidental = ""
    if "#" in body:
        accidental = "#" * body.count("#")
    elif "-" in body:
        accidental = "b" * body.count("-")
    elif "n" in body:
        accidental = ""
    return f"{base.upper()}{accidental}{octave}"


def kern_duration_components(duration_num: int, dots: int, is_rest: bool) -> List[str]:
    if duration_num <= 0:
        raise ValueError(f"Unsupported Kern duration '{duration_num}'")

    if str(duration_num) in KERN_DURATION_TO_NAME:
        duration_name = KERN_DURATION_TO_NAME[str(duration_num)]
        return duration_tokens(duration_name, dots=dots, is_rest=is_rest)

    base_by_divisor = {
        3: "<tuplet_3>",
        5: "<tuplet_5>",
        6: "<tuplet_6>",
        7: "<tuplet_7>",
    }
    for divisor, tuplet_token in base_by_divisor.items():
        if duration_num % divisor != 0:
            continue
        base = duration_num // divisor
        base_name = KERN_DURATION_TO_NAME.get(str(base))
        if base_name is None:
            continue
        return [tuplet_token, *duration_tokens(base_name, dots=dots, is_rest=is_rest)]

    # GrandStaff includes a small tail of non-canonical durations (e.g. 22, 36).
    # Quantize these to the nearest representable duration in our fixed token scheme.
    target_ql = 4.0 / float(duration_num)
    candidates: List[Tuple[float, Optional[str], str]] = []
    for base_raw, base_name in KERN_DURATION_TO_NAME.items():
        base_num = int(base_raw)
        candidates.append((4.0 / float(base_num), None, base_name))
        for divisor, tuplet_token in base_by_divisor.items():
            quantized_num = base_num * divisor
            candidates.append((4.0 / float(quantized_num), tuplet_token, base_name))

    best_ql, best_tuplet, best_name = min(
        candidates, key=lambda item: abs(item[0] - target_ql)
    )
    quantized_duration = duration_tokens(best_name, dots=dots, is_rest=is_rest)
    if best_tuplet is None:
        return quantized_duration
    return [best_tuplet, *quantized_duration]


def parse_kern_event(
    event: str, fallback_duration: Optional[Tuple[int, int]]
) -> Tuple[List[str], List[str], bool, Tuple[int, int]]:
    match = re.match(r"^(\d+)(\.*)(.*)$", event)
    if match:
        duration_num, dot_group, body = match.groups()
        duration_value = int(duration_num)
        dots = len(dot_group)
    else:
        if fallback_duration is None:
            duration_value, dots = (4, 0)
        else:
            duration_value, dots = fallback_duration
        body = event
    duration_parts = kern_duration_components(duration_value, dots=dots, is_rest=False)

    if "r" in body.lower():
        rest_duration_parts = kern_duration_components(duration_value, dots=dots, is_rest=True)
        return ["rest"], rest_duration_parts, True, (duration_value, dots)

    prefer_flats = "-" in body and "#" not in body
    normalized = _normalize_pitch_symbol(kern_pitch_token(body), prefer_flats=prefer_flats)
    pitch = _normalize_note_pitch_symbol(normalized)
    return [f"note-{pitch}"], duration_parts, False, (duration_value, dots)


def parse_kern_cell(
    cell: str, fallback_duration: Optional[Tuple[int, int]]
) -> Tuple[List[str], Optional[Tuple[int, int]]]:
    events = [part for part in cell.split(" ") if part]
    if not events:
        return [], fallback_duration

    parsed = []
    active_duration = fallback_duration
    for event in events:
        pitches, duration, is_rest, active_duration = parse_kern_event(event, active_duration)
        parsed.append((pitches, duration, is_rest))

    if len(parsed) > 1:
        if any(is_rest for _, _, is_rest in parsed):
            output: List[str] = []
            for pitches, duration, _ in parsed:
                output.extend(pitches)
                output.extend(duration)
            return output, active_duration
        chord_tokens: List[str] = ["<chord_start>"]
        duration_tokens_out = parsed[0][1]
        for pitches, _, _ in parsed:
            chord_tokens.extend(pitches)
        chord_tokens.extend(["<chord_end>", *duration_tokens_out])
        return chord_tokens, active_duration

    pitches, duration, _ = parsed[0]
    return [*pitches, *duration], active_duration


def convert_kern_file(path: Path) -> List[str]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    tokens: List[str] = ["<bos>", "<staff_start>"]
    current_voice: Optional[int] = None
    duration_by_voice: Dict[int, Tuple[int, int]] = {}
    measure_open = False
    has_clef = False
    has_time = False

    def _ensure_header_defaults() -> None:
        nonlocal has_clef, has_time
        if not has_clef:
            tokens.append("clef-G2")
            has_clef = True
        if not has_time:
            tokens.append("timeSignature-4/4")
            has_time = True

    def ensure_measure_open() -> None:
        nonlocal measure_open, current_voice
        if measure_open:
            return
        _ensure_header_defaults()
        tokens.append("<measure_start>")
        measure_open = True
        current_voice = None

    def close_measure_if_open() -> None:
        nonlocal measure_open, current_voice
        if not measure_open:
            return
        if tokens[-1] != "<measure_end>":
            tokens.append("<measure_end>")
        measure_open = False
        current_voice = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        cells = line.split("\t")
        if any(cell.startswith("=") for cell in cells):
            close_measure_if_open()
            continue
        if line.startswith("*"):
            interpretation_tokens: List[str] = []
            for idx, cell in enumerate(cells, start=1):
                if cell in {"*", "*-"}:
                    continue
                for parser in (kern_clef_token, kern_key_signature_token, kern_time_signature_token):
                    parsed = parser(cell)
                    if parsed:
                        interpretation_tokens.append(parsed)
            if interpretation_tokens:
                close_measure_if_open()
                deduped = _dedupe_preserve(interpretation_tokens)
                for token in deduped:
                    if token.startswith("clef-"):
                        has_clef = True
                    elif token.startswith("timeSignature-"):
                        has_time = True
                tokens.extend(deduped)
            continue

        for idx, cell in enumerate(cells, start=1):
            if cell in {"", "."}:
                continue
            canonical_voice = min(idx, MAX_SUPPORTED_VOICE_INDEX)
            if canonical_voice < 1 or canonical_voice > MAX_SUPPORTED_VOICE_INDEX:
                continue
            if idx > MAX_SUPPORTED_VOICE_INDEX:
                # Our locked vocabulary supports four voices max; drop excess spines.
                continue

            cell_tokens, duration_state = parse_kern_cell(cell, duration_by_voice.get(canonical_voice))
            if not cell_tokens:
                continue
            ensure_measure_open()
            if current_voice != canonical_voice:
                tokens.append(f"<voice_{canonical_voice}>")
                current_voice = canonical_voice
            if duration_state is not None:
                duration_by_voice[canonical_voice] = duration_state
            tokens.extend(cell_tokens)

    close_measure_if_open()
    tokens.extend(["<staff_end>", "<eos>"])
    return tokens


_NATURAL_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SEMITONE_TO_SHARP = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
_SEMITONE_TO_FLAT = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
_PITCH_CLASS_TO_SEMITONE = {
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


def _normalize_pitch_symbol(symbol: str, prefer_flats: Optional[bool] = None) -> str:
    match = re.fullmatch(r"([A-G])([#b]*)(-?\d+)", symbol.strip())
    if not match:
        raise ValueError(f"Unsupported pitch symbol '{symbol}'")

    letter, accidental_group, octave_text = match.groups()
    octave = int(octave_text)
    semitone = _NATURAL_SEMITONES[letter] + accidental_group.count("#") - accidental_group.count("b")
    while semitone < 0:
        semitone += 12
        octave -= 1
    while semitone >= 12:
        semitone -= 12
        octave += 1

    if semitone in {1, 3, 6, 8, 10}:
        if prefer_flats is None:
            prefer_flats = "b" in accidental_group and "#" not in accidental_group
        pitch_class = _SEMITONE_TO_FLAT[semitone] if prefer_flats else _SEMITONE_TO_SHARP[semitone]
    else:
        pitch_class = _SEMITONE_TO_SHARP[semitone]
    return f"{pitch_class}{octave}"


def _normalize_grace_pitch_symbol(symbol: str) -> str:
    normalized = _normalize_pitch_symbol(symbol)
    match = re.fullmatch(r"([A-G])(?:[#b]?)(-?\d+)", normalized)
    if match is None:
        raise ValueError(f"Unsupported grace pitch symbol '{symbol}'")
    letter, octave_text = match.groups()
    octave = int(octave_text)
    octave = max(2, min(6, octave))
    candidate = f"{letter}{octave}"
    if f"gracenote-{candidate}" in SUPPORTED_GRACE_TOKENS:
        return candidate
    return "C4"


def _normalize_note_pitch_symbol(symbol: str) -> str:
    normalized = _normalize_pitch_symbol(symbol)
    match = re.fullmatch(r"([A-G](?:#|b)?)(-?\d+)", normalized)
    if match is None:
        raise ValueError(f"Unsupported pitch symbol '{symbol}'")
    pitch_class, octave_text = match.groups()
    octave = int(octave_text)
    candidate = f"{pitch_class}{octave}"
    if f"note-{candidate}" in SUPPORTED_NOTE_TOKENS:
        return candidate

    semitone = _PITCH_CLASS_TO_SEMITONE.get(pitch_class)
    if semitone is None:
        return "C4"
    target_midi = 12 * (octave + 1) + semitone

    best_symbol = "C4"
    best_key: Tuple[int, int, str] | None = None
    for note_token in SUPPORTED_NOTE_TOKENS:
        note_symbol = note_token[len("note-") :]
        parsed = re.fullmatch(r"([A-G](?:#|b)?)(-?\d+)", note_symbol)
        if parsed is None:
            continue
        cand_pitch_class, cand_octave_text = parsed.groups()
        cand_octave = int(cand_octave_text)
        cand_semitone = _PITCH_CLASS_TO_SEMITONE.get(cand_pitch_class)
        if cand_semitone is None:
            continue
        cand_midi = 12 * (cand_octave + 1) + cand_semitone
        key = (
            abs(cand_midi - target_midi),
            0 if cand_pitch_class == pitch_class else 1,
            note_symbol,
        )
        if best_key is None or key < best_key:
            best_key = key
            best_symbol = note_symbol
    return best_symbol


def token_from_music21_pitch(pitch) -> str:
    name = pitch.name.replace("-", "b")
    prefer_flats = "b" in name and "#" not in name
    normalized = _normalize_pitch_symbol(f"{name}{pitch.octave}", prefer_flats=prefer_flats)
    normalized = _normalize_note_pitch_symbol(normalized)
    return f"note-{normalized}"


def _dedupe_preserve(tokens: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def _normalize_lookup_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _lookup_text_tokens(text: str, mapping: Dict[str, str]) -> List[str]:
    normalized = _normalize_lookup_text(text)
    if not normalized:
        return []
    matched: List[str] = []
    for phrase, token in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
        if re.search(pattern, normalized):
            matched.append(token)
    return _dedupe_preserve(matched)


def _dynamic_token_from_music21(dynamic_obj) -> Optional[str]:
    raw = str(getattr(dynamic_obj, "value", "") or getattr(dynamic_obj, "name", "")).strip().lower()
    if not raw:
        return None
    raw = raw.replace("z", "z")
    raw = re.sub(r"[^a-z]+", "", raw)
    return DYNAMIC_VALUE_TO_TOKEN.get(raw)


def _tempo_tokens_from_music21(mark_obj) -> List[str]:
    text = str(getattr(mark_obj, "text", "") or "").strip()
    tokens = _lookup_text_tokens(text, TEMPO_TEXT_TO_TOKEN)
    if tokens:
        return tokens
    number = getattr(mark_obj, "number", None)
    if number is None:
        return []
    try:
        bpm = float(number)
    except (TypeError, ValueError):
        return []
    if bpm < 52:
        return ["tempo-Largo"]
    if bpm < 60:
        return ["tempo-Lento"]
    if bpm < 72:
        return ["tempo-Adagio"]
    if bpm < 92:
        return ["tempo-Andante"]
    if bpm < 108:
        return ["tempo-Moderato"]
    if bpm < 140:
        return ["tempo-Allegro"]
    if bpm < 172:
        return ["tempo-Vivace"]
    if bpm < 196:
        return ["tempo-Presto"]
    return ["tempo-Prestissimo"]


def _expression_tokens_from_music21(text_expression_obj) -> List[str]:
    raw_text = str(
        getattr(text_expression_obj, "content", "")
        or getattr(text_expression_obj, "text", "")
        or text_expression_obj
    ).strip()
    if not raw_text:
        return []
    return _dedupe_preserve(
        [
            *_lookup_text_tokens(raw_text, TEMPO_TEXT_TO_TOKEN),
            *_lookup_text_tokens(raw_text, EXPRESSION_TEXT_TO_TOKEN),
        ]
    )


def _event_articulation_tokens(event_obj) -> List[str]:
    tokens: List[str] = []
    for articulation in getattr(event_obj, "articulations", []):
        token = ARTICULATION_CLASS_TO_TOKEN.get(type(articulation).__name__)
        if token is not None:
            tokens.append(token)
    for expression in getattr(event_obj, "expressions", []):
        token = EXPRESSION_CLASS_TO_TOKEN.get(type(expression).__name__)
        if token is not None:
            tokens.append(token)
    return _dedupe_preserve(tokens)


def _event_tie_tokens(event_obj) -> List[str]:
    tie_types: set[str] = set()
    tie_obj = getattr(event_obj, "tie", None)
    if tie_obj is not None:
        tie_types.add(str(getattr(tie_obj, "type", "")).lower())
    for chord_note in getattr(event_obj, "notes", []):
        chord_tie = getattr(chord_note, "tie", None)
        if chord_tie is not None:
            tie_types.add(str(getattr(chord_tie, "type", "")).lower())

    tokens: List[str] = []
    if "stop" in tie_types or "continue" in tie_types:
        tokens.append("tie_end")
    if "start" in tie_types or "continue" in tie_types:
        tokens.append("tie_start")
    return tokens


_CLEF_FALLBACK_MAP: Dict[str, str] = {
    # KERN ottava/transposing variants → closest supported clef
    "clef-Gv2": "clef-G2_8vb",
    "clef-G^2": "clef-G2_8va",
    "clef-GV2": "clef-G2_8vb",
    "clef-Fv4": "clef-F4",
    "clef-F^4": "clef-F4",
    "clef-C2": "clef-C3",       # mezzo-soprano → alto (closest)
    "clef-C5": "clef-C4",       # baritone C → tenor (closest)
    "clef-F3": "clef-F4",       # baritone F → bass
    "clef-F5": "clef-F4",       # sub-bass → bass
    "clef-G1": "clef-G2",       # French violin → treble
    "clef-G3": "clef-G2",       # uncommon → treble
}


def _normalize_clef_token(token: str) -> Optional[str]:
    if token in SUPPORTED_CLEF_TOKENS:
        return token
    if token in _CLEF_FALLBACK_MAP:
        return _CLEF_FALLBACK_MAP[token]
    return None


def _normalize_key_signature_token(token: str) -> str:
    if token in SUPPORTED_KEY_SIGNATURE_TOKENS:
        return token
    return "keySignature-none"


def _normalize_time_signature_token(token: str) -> str:
    if token in SUPPORTED_TIME_SIGNATURE_TOKENS:
        return token
    return "timeSignature-other"


def _build_slur_token_maps(part, spanner_module) -> Tuple[Dict[int, int], Dict[int, int]]:
    starts: Dict[int, int] = {}
    ends: Dict[int, int] = {}
    for slur in part.recurse().getElementsByClass(spanner_module.Slur):
        first = slur.getFirst()
        last = slur.getLast()
        if first is not None:
            starts[id(first)] = starts.get(id(first), 0) + 1
        if last is not None:
            ends[id(last)] = ends.get(id(last), 0) + 1
    return starts, ends


def _barline_token_for_measure(measure, bar_module) -> Optional[str]:
    left = measure.leftBarline
    right = measure.rightBarline
    left_repeat = isinstance(left, bar_module.Repeat) and str(getattr(left, "direction", "")).lower() == "start"
    right_repeat = isinstance(right, bar_module.Repeat) and str(getattr(right, "direction", "")).lower() == "end"
    if left_repeat and right_repeat:
        return "repeat_both"
    if left_repeat:
        return "repeat_start"
    if right_repeat:
        return "repeat_end"
    right_type = str(getattr(right, "type", "")).lower() if right is not None else ""
    if right_type == "double":
        return "double_barline"
    if right_type == "final":
        return "final_barline"
    return None


def duration_from_quarter_length(quarter_length: float) -> str:
    duration_by_ql = {
        4.0: "whole",
        2.0: "half",
        1.0: "quarter",
        0.5: "eighth",
        0.25: "sixteenth",
        0.125: "thirty_second",
        0.0625: "sixty_fourth",
    }
    for ql, name in duration_by_ql.items():
        if abs(quarter_length - ql) < 1e-6:
            return name
    nearest = min(duration_by_ql.items(), key=lambda item: abs(item[0] - quarter_length))
    return nearest[1]


def music21_duration_tokens(duration_obj, is_rest: bool) -> List[str]:
    type_to_name = {
        "whole": "whole",
        "half": "half",
        "quarter": "quarter",
        "eighth": "eighth",
        "16th": "sixteenth",
        "32nd": "thirty_second",
        "64th": "sixty_fourth",
    }
    duration_type = str(getattr(duration_obj, "type", "")).lower()
    duration_name = type_to_name.get(duration_type)
    if duration_name is None:
        duration_name = duration_from_quarter_length(float(duration_obj.quarterLength))

    dots = int(getattr(duration_obj, "dots", 0))
    tokens: List[str] = []
    tuplets = list(getattr(duration_obj, "tuplets", []))
    for tuplet in tuplets:
        actual = int(getattr(tuplet, "numberNotesActual", 0) or 0)
        if actual in {3, 5, 6, 7}:
            tokens.append(f"<tuplet_{actual}>")

    tokens.extend(duration_tokens(duration_name, dots=dots, is_rest=is_rest))
    return tokens


def convert_musicxml_file(path: Path) -> List[str]:
    try:
        from music21 import bar, chord, converter, dynamics, expressions, key, meter, note, spanner, stream, tempo
    except ImportError as exc:  # pragma: no cover - dependency based
        raise RuntimeError(
            "music21 is required for MusicXML/MSCX conversion. Install with: pip install music21"
        ) from exc

    score = converter.parse(str(path))
    if not score.parts:
        raise ValueError(f"No parts found in score: {path}")

    tokens: List[str] = ["<bos>"]
    converted_parts = 0
    for part in score.parts:
        measures = list(part.getElementsByClass("Measure"))
        if not measures:
            continue
        converted_parts += 1
        slur_starts, slur_ends = _build_slur_token_maps(part, spanner)

        tokens.append("<staff_start>")
        clef_obj = part.recurse().getElementsByClass("Clef").first()
        if clef_obj is not None:
            sign = getattr(clef_obj, "sign", "G")
            line = getattr(clef_obj, "line", 2)
            clef_token = _normalize_clef_token(f"clef-{sign}{line}")
            if clef_token is not None:
                tokens.append(clef_token)
        key_obj = part.recurse().getElementsByClass(key.KeySignature).first()
        if key_obj is not None:
            fifths = int(key_obj.sharps or 0)
            tonic = KEY_SIGNATURE_MAJOR_BY_FIFTHS.get(fifths, "C")
            mode_value = getattr(key_obj, "mode", None)
            if mode_value is None:
                try:
                    mode_value = key_obj.asKey().mode
                except (AttributeError, TypeError, ValueError):
                    mode_value = "major"
            mode_text = str(mode_value).strip().lower()
            mode = "m" if (mode_text == "m" or mode_text.startswith("min")) else "M"
            tokens.append(_normalize_key_signature_token(f"keySignature-{tonic}{mode}"))
        ts_obj = part.recurse().getElementsByClass(meter.TimeSignature).first()
        if ts_obj is not None:
            tokens.append(_normalize_time_signature_token(f"timeSignature-{ts_obj.ratioString}"))

        for measure in measures:
            tokens.append("<measure_start>")
            annotation_tokens: List[Tuple[float, str]] = []
            for dynamic_obj in measure.recurse().getElementsByClass(dynamics.Dynamic):
                dynamic_token = _dynamic_token_from_music21(dynamic_obj)
                if dynamic_token is not None:
                    annotation_tokens.append((float(getattr(dynamic_obj, "offset", 0.0)), dynamic_token))
            for tempo_obj in measure.recurse().getElementsByClass(tempo.MetronomeMark):
                offset = float(getattr(tempo_obj, "offset", 0.0))
                for tempo_token in _tempo_tokens_from_music21(tempo_obj):
                    annotation_tokens.append((offset, tempo_token))
            for text_expression_obj in measure.recurse().getElementsByClass(expressions.TextExpression):
                offset = float(getattr(text_expression_obj, "offset", 0.0))
                for expression_token in _expression_tokens_from_music21(text_expression_obj):
                    annotation_tokens.append((offset, expression_token))
            for _, token_value in sorted(annotation_tokens, key=lambda item: (item[0], item[1])):
                tokens.append(token_value)

            voices = list(measure.getElementsByClass(stream.Voice))
            if len(voices) > 1:
                voice_streams: List[Tuple[Optional[int], object]] = [
                    (idx + 1, voice) for idx, voice in enumerate(voices)
                ]
            else:
                voice_streams = [(None, measure)]

            for voice_index, source in voice_streams:
                if voice_index is not None:
                    tokens.append(f"<voice_{min(voice_index, 4)}>")
                events = sorted(
                    list(source.notesAndRests),
                    key=lambda event: (
                        float(getattr(event, "offset", 0.0)),
                        1 if isinstance(event, note.Rest) else 0,
                    ),
                )
                occupied_offsets = {
                    float(getattr(event, "offset", 0.0))
                    for event in events
                    if not isinstance(event, note.Rest)
                }
                consumed_until = 0.0
                for element in events:
                    element_offset = float(getattr(element, "offset", 0.0))
                    if isinstance(element, note.Rest) and element_offset in occupied_offsets:
                        continue
                    if element_offset < consumed_until - 1e-6:
                        continue
                    for _ in range(slur_ends.get(id(element), 0)):
                        tokens.append("slur_end")
                    for _ in range(slur_starts.get(id(element), 0)):
                        tokens.append("slur_start")
                    tokens.extend(_event_tie_tokens(element))
                    tokens.extend(_event_articulation_tokens(element))

                    if isinstance(element, note.Rest):
                        tokens.append("rest")
                        tokens.extend(music21_duration_tokens(element.duration, is_rest=True))
                    elif isinstance(element, chord.Chord):
                        if bool(getattr(element.duration, "isGrace", False)):
                            for chord_pitch in element.pitches:
                                grace_pitch = _normalize_grace_pitch_symbol(
                                    token_from_music21_pitch(chord_pitch)[len("note-") :]
                                )
                                tokens.append(f"gracenote-{grace_pitch}")
                                tokens.extend(music21_duration_tokens(element.duration, is_rest=False))
                        else:
                            tokens.append("<chord_start>")
                            for chord_pitch in element.pitches:
                                tokens.append(token_from_music21_pitch(chord_pitch))
                            tokens.append("<chord_end>")
                            tokens.extend(music21_duration_tokens(element.duration, is_rest=False))
                    else:
                        pitch_token = token_from_music21_pitch(element.pitch)
                        if bool(getattr(element.duration, "isGrace", False)):
                            grace_pitch = _normalize_grace_pitch_symbol(pitch_token[len("note-") :])
                            tokens.append(f"gracenote-{grace_pitch}")
                        else:
                            tokens.append(pitch_token)
                        tokens.extend(music21_duration_tokens(element.duration, is_rest=False))

                    if not bool(getattr(element.duration, "isGrace", False)):
                        consumed_until = max(
                            consumed_until,
                            element_offset + float(getattr(element, "quarterLength", 0.0)),
                        )

            barline_token = _barline_token_for_measure(measure, bar)
            if barline_token is not None:
                tokens.append(barline_token)
            tokens.append("<measure_end>")
        tokens.append("<staff_end>")

    if converted_parts == 0:
        raise ValueError(f"No measures found in score parts: {path}")
    tokens.append("<eos>")
    return tokens


def pick_converter(entry: Dict[str, object]) -> Tuple[str, str]:
    if entry.get("semantic_path"):
        return "semantic", str(entry["semantic_path"])
    if entry.get("krn_path"):
        return "kern", str(entry["krn_path"])
    if entry.get("musicxml_path"):
        return "musicxml", str(entry["musicxml_path"])
    if entry.get("mscx_path"):
        return "musicxml", str(entry["mscx_path"])
    if entry.get("mei_path"):
        return "mei", str(entry["mei_path"])
    raise ValueError(f"No supported source path for sample {entry.get('sample_id')}")


def convert_entry(project_root: Path, entry: Dict[str, object]) -> Tuple[str, List[str]]:
    source_format, source_relpath = pick_converter(entry)
    source_path = resolve_optional_path(project_root, source_relpath)
    if source_path is None or not source_path.exists():
        raise FileNotFoundError(f"Source file not found: {source_relpath}")

    if source_format == "semantic":
        return source_format, convert_semantic_file(source_path)
    if source_format == "kern":
        return source_format, convert_kern_file(source_path)
    if source_format == "musicxml":
        return source_format, convert_musicxml_file(source_path)
    if source_format == "mei":
        return source_format, convert_mei_file(source_path)
    raise ValueError(f"Unknown source format '{source_format}'")


def write_token_manifest(
    project_root: Path,
    manifest_entries: Sequence[Dict[str, object]],
    output_path: Path,
    summary_path: Path,
    max_samples: Optional[int],
    allow_failures: bool,
    allow_relaxed_validation: bool,
    datasets_filter: Optional[set[str]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    converted = 0
    failures: List[Dict[str, str]] = []
    converted_by_dataset: Counter[str] = Counter()
    converted_by_source: Counter[str] = Counter()
    converted_by_split: Dict[str, Counter[str]] = defaultdict(Counter)
    relaxed_validation_samples = 0
    relaxed_validation_by_source: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8") as output_file:
        for entry in manifest_entries:
            dataset = str(entry.get("dataset", "")).lower()
            if datasets_filter and dataset not in datasets_filter:
                continue
            total += 1
            if max_samples is not None and converted >= max_samples:
                break
            try:
                source_format, sequence = convert_entry(project_root, entry)
                strict_validation_error: Optional[Exception] = None
                try:
                    validate_token_sequence(sequence, strict=True)
                except Exception as exc:
                    strict_validation_error = exc

                if strict_validation_error is not None:
                    can_relax = allow_relaxed_validation and source_format in {
                        "semantic",
                        "kern",
                        "musicxml",
                        "mei",
                    }
                    if not can_relax:
                        raise strict_validation_error
                    validate_token_sequence(sequence, strict=False)
                    relaxed_validation_samples += 1
                    relaxed_validation_by_source[source_format] += 1
                converted_entry = {
                    "sample_id": entry["sample_id"],
                    "dataset": entry["dataset"],
                    "split": entry["split"],
                    "image_path": entry.get("image_path"),
                    "source_format": source_format,
                    "source_path": pick_converter(entry)[1],
                    "token_sequence": sequence,
                    "token_count": len(sequence),
                }
                output_file.write(json.dumps(converted_entry, ensure_ascii=False) + "\n")
                converted += 1
                converted_by_dataset[dataset] += 1
                converted_by_source[source_format] += 1
                converted_by_split[dataset][str(entry.get("split", "train"))] += 1
            except Exception as exc:  # explicit capture with report
                failures.append(
                    {
                        "sample_id": str(entry.get("sample_id", "")),
                        "dataset": dataset,
                        "error": str(exc),
                    }
                )

    summary = {
        "input_samples_seen": total,
        "converted_samples": converted,
        "failed_samples": len(failures),
        "output_manifest": relpath(project_root, output_path),
        "converted_by_dataset": dict(converted_by_dataset),
        "converted_by_source": dict(converted_by_source),
        "converted_by_split": {
            dataset: dict(split_counter) for dataset, split_counter in converted_by_split.items()
        },
        "relaxed_validation_samples": relaxed_validation_samples,
        "relaxed_validation_by_source": dict(relaxed_validation_by_source),
        "failures_preview": failures[:20],
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if failures and not allow_failures:
        first = failures[0]
        raise RuntimeError(
            f"Conversion failed for {len(failures)} sample(s); "
            f"first failure sample_id={first['sample_id']} error={first['error']}"
        )


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Convert source labels in manifest into token sequences."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help="Repository root path.",
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "master_manifest.jsonl",
        help="Input canonical manifest JSONL path.",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "token_manifest.jsonl",
        help="Output tokenized manifest JSONL path.",
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=project_root / "src" / "data" / "manifests" / "token_manifest_summary.json",
        help="Output summary JSON path.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for conversion samples.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset filter (e.g., primus,cameraprimus).",
    )
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Do not fail the run when a sample conversion fails.",
    )
    parser.add_argument(
        "--allow-relaxed-validation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use relaxed grammar fallback when strict validation fails (default: enabled).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_entries = load_manifest_entries(args.input_manifest)
    datasets_filter = (
        {item.strip().lower() for item in args.datasets.split(",") if item.strip()}
        if args.datasets
        else None
    )
    write_token_manifest(
        project_root=args.project_root,
        manifest_entries=manifest_entries,
        output_path=args.output_manifest,
        summary_path=args.output_summary,
        max_samples=args.max_samples,
        allow_failures=args.allow_failures,
        allow_relaxed_validation=args.allow_relaxed_validation,
        datasets_filter=datasets_filter,
    )
    print(f"Token manifest written to {args.output_manifest}")
    print(f"Summary written to {args.output_summary}")


if __name__ == "__main__":
    main()
