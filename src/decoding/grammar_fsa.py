#!/usr/bin/env python3
"""Finite-state grammar constraints for OMR decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set

from src.tokenizer.vocab import OMRVocabulary, build_default_vocabulary


DURATION_TO_BEATS: Dict[str, float] = {
    "_whole": 4.0,
    "_half": 2.0,
    "_quarter": 1.0,
    "_eighth": 0.5,
    "_sixteenth": 0.25,
    "_thirty_second": 0.125,
    "_sixty_fourth": 0.0625,
}

TUPLET_BEAT_SCALE: Dict[str, float] = {
    "<tuplet_3>": 2.0 / 3.0,
    "<tuplet_5>": 4.0 / 5.0,
    "<tuplet_6>": 5.0 / 6.0,
    "<tuplet_7>": 6.0 / 7.0,
}

MIN_DURATION_BEAT = min(DURATION_TO_BEATS.values())


@dataclass
class GrammarState:
    begun: bool = False
    ended: bool = False
    staff_open: bool = False
    staff_closed: bool = False
    measure_open: bool = False
    chord_open: bool = False
    expecting_duration: bool = False
    allow_duration_modifier: bool = False
    last_token: str | None = None
    beats_per_measure: float = 4.0
    beat_position: float = 0.0
    last_duration_base: float = 0.0
    pending_tuplet_scale: float = 1.0
    pending_duration_kind: str | None = None
    pending_tie_start: bool = False
    current_voice: str | None = None
    voice_switches_in_measure: int = 0
    voice_beat_positions: Dict[str, float] = field(default_factory=dict)
    header_has_clef: bool = False
    header_has_time: bool = False
    measure_index_in_staff: int = 0
    measure_has_content: bool = False


def _beats_from_time_signature(token: str) -> Optional[float]:
    if not token.startswith("timeSignature-"):
        return None
    symbol = token.split("-", 1)[1]
    if symbol == "C":
        return 4.0
    if symbol == "C/":
        return 4.0
    if symbol == "other":
        return 4.0
    if "/" not in symbol:
        return None
    numerator_text, denominator_text = symbol.split("/", 1)
    if not numerator_text.isdigit() or not denominator_text.isdigit():
        return None
    numerator = int(numerator_text)
    denominator = int(denominator_text)
    if denominator <= 0:
        return None
    return numerator * (4.0 / denominator)


class GrammarFSA:
    def __init__(self, vocabulary: OMRVocabulary | None = None) -> None:
        self.vocab = vocabulary or build_default_vocabulary()
        self.state = GrammarState()

    def reset(self) -> None:
        self.state = GrammarState()

    def _remaining_beats(self) -> float:
        return max(0.0, self.state.beats_per_measure - self.state.beat_position)

    def _allowed_duration_tokens(self) -> Set[str]:
        remaining = self._remaining_beats()
        scale = self.state.pending_tuplet_scale
        duration_kind = self.state.pending_duration_kind
        if duration_kind == "grace":
            return {token for token in self.vocab.base_duration_tokens if not token.endswith("_rest")}
        # In multi-voice measures, per-voice beat tracking drifts due to
        # tuplet rounding and independent voice durations.  Relax duration
        # constraints when voices are active and beats have overflowed.
        if self.state.voice_switches_in_measure > 0 and remaining <= 1e-6:
            return {
                token
                for token in self.vocab.base_duration_tokens
                if token in DURATION_TO_BEATS and not token.endswith("_rest")
            }
        fitting_durations = {
            token
            for token in self.vocab.base_duration_tokens
            if token in DURATION_TO_BEATS and (DURATION_TO_BEATS[token] * scale) <= remaining + 1e-6
        }
        durations = set(fitting_durations)
        if duration_kind in {"note", "rest"}:
            durations = {token for token in durations if not token.endswith("_rest")}
        if duration_kind in {"note", "rest"} and remaining > 1e-6:
            # Allow a minimal overflow option: notes may need ties, and rests may
            # legitimately overflow in multi-voice KERN data where per-voice beats
            # do not sum exactly to the time signature.
            overflow = [
                token
                for token in self.vocab.base_duration_tokens
                if token in DURATION_TO_BEATS
                and not token.endswith("_rest")
                and (DURATION_TO_BEATS[token] * scale) > remaining + 1e-6
            ]
            if overflow:
                if self.state.voice_switches_in_measure > 0:
                    # In multi-voice measures, per-voice beat tracking drifts
                    # due to tuplets and independent voice durations.  Allow
                    # all overflow durations rather than just the minimum.
                    durations |= set(overflow)
                else:
                    min_overflow_beats = min(DURATION_TO_BEATS[token] for token in overflow)
                    durations |= {
                        token
                        for token in overflow
                        if abs(DURATION_TO_BEATS[token] - min_overflow_beats) <= 1e-6
                    }
        if (
            duration_kind == "rest"
            and self.state.beat_position <= 1e-6
            and "_whole" in self.vocab.base_duration_tokens
        ):
            # Whole-rest glyphs are used as full-measure rests across meters.
            durations.add("_whole")
        if durations:
            return durations
        if duration_kind in {"rest", "note"}:
            for token in ("_eighth", "_quarter", "_half", "_whole"):
                if token in self.vocab.base_duration_tokens:
                    return {token}
        return {token for token in self.vocab.base_duration_tokens if not token.endswith("_rest")}

    def _allowed_voice_tokens(self) -> Set[str]:
        if self.state.expecting_duration or self.state.chord_open:
            return set()
        if self.state.current_voice is None:
            return set(self.vocab.voice_tokens)
        return {token for token in self.vocab.voice_tokens if token != self.state.current_voice}

    def _measure_end_forced(self) -> bool:
        if not self.state.measure_open:
            return False
        if self.state.expecting_duration or self.state.chord_open:
            return False
        # In multi-voice measures, per-voice beat positions may not align with
        # the global beat counter.  Do not force measure end when voices are
        # active — the converter will emit <measure_end> explicitly.
        if self.state.voice_switches_in_measure > 0:
            return False
        remaining = self._remaining_beats()
        return remaining <= 1e-6 or remaining < MIN_DURATION_BEAT

    def _contextual_allowed(self) -> Set[str]:
        state = self.state
        if not state.begun:
            return {"<bos>"}
        if state.ended:
            return set()
        if state.staff_closed:
            return {"<eos>", "<staff_start>"}
        if not state.staff_open:
            return {"<staff_start>"}
        if state.chord_open:
            return set(self.vocab.note_tokens) | {"<chord_end>"}
        if state.expecting_duration:
            allowed = self._allowed_duration_tokens()
            if state.pending_tuplet_scale == 1.0:
                allowed |= set(TUPLET_BEAT_SCALE.keys())
            return allowed

        if not state.measure_open:
            allow_measure_start = (
                state.measure_index_in_staff > 0
                or (state.header_has_clef and state.header_has_time)
            )
            allowed = (
                set(self.vocab.clef_tokens)
                | set(self.vocab.key_signature_tokens)
                | set(self.vocab.time_signature_tokens)
                | {"<staff_end>"}
            )
            if allow_measure_start:
                allowed.add("<measure_start>")
            return allowed

        if self._measure_end_forced():
            forced_tokens = {"<measure_end>"} | set(self.vocab.in_measure_attribute_tokens)
            forced_tokens |= set(self.vocab.grace_tokens)
            if state.current_voice is not None:
                forced_tokens |= self._allowed_voice_tokens()
            return forced_tokens

        in_measure = (
            self._allowed_voice_tokens()
            | set(self.vocab.note_tokens)
            | set(self.vocab.grace_tokens)
            | set(self.vocab.in_measure_attribute_tokens)
            | {"<chord_start>", "<measure_end>"}
        )
        if (
            state.measure_index_in_staff == 1
            and not state.measure_has_content
            and state.beat_position <= 1e-6
        ):
            in_measure.discard("<measure_end>")
        if state.allow_duration_modifier:
            in_measure |= set(self.vocab.duration_modifier_tokens)
        return in_measure

    def valid_next_tokens(self) -> Set[str]:
        return self._contextual_allowed()

    def valid_next_token_ids(self) -> Set[int]:
        return {
            self.vocab.token_to_id[token]
            for token in self.valid_next_tokens()
            if token in self.vocab.token_to_id
        }

    def binary_mask(self) -> List[int]:
        mask = [0] * self.vocab.size
        for token_id in self.valid_next_token_ids():
            mask[token_id] = 1
        return mask

    def _apply_duration_token(self, token: str) -> None:
        state = self.state
        if token in DURATION_TO_BEATS:
            if state.pending_duration_kind == "grace":
                state.last_duration_base = 0.0
                state.pending_tie_start = False
            else:
                scaled_beats = DURATION_TO_BEATS[token] * state.pending_tuplet_scale
                remaining_before = max(0.0, state.beats_per_measure - state.beat_position)
                if (
                    state.pending_duration_kind == "rest"
                    and token == "_whole"
                    and state.beat_position <= 1e-6
                    and scaled_beats > state.beats_per_measure + 1e-6
                ):
                    state.last_duration_base = state.beats_per_measure
                    state.beat_position = state.beats_per_measure
                    state.pending_tie_start = False
                else:
                    state.last_duration_base = scaled_beats
                    state.beat_position += state.last_duration_base
                    state.pending_tie_start = (
                        state.pending_duration_kind == "note"
                        and scaled_beats > remaining_before + 1e-6
                    )
            state.expecting_duration = False
            state.allow_duration_modifier = state.pending_duration_kind != "grace"
            state.pending_tuplet_scale = 1.0
            state.pending_duration_kind = None
            if state.current_voice is not None:
                state.voice_beat_positions[state.current_voice] = state.beat_position
            return
        if token == "_dot" and state.last_duration_base > 0:
            state.beat_position += state.last_duration_base * 0.5
            state.allow_duration_modifier = False
            if state.current_voice is not None:
                state.voice_beat_positions[state.current_voice] = state.beat_position
            return
        if token == "_double_dot" and state.last_duration_base > 0:
            state.beat_position += state.last_duration_base * 0.75
            state.allow_duration_modifier = False
            if state.current_voice is not None:
                state.voice_beat_positions[state.current_voice] = state.beat_position
            return
        state.allow_duration_modifier = False

    def step(self, token: str, strict: bool = True) -> None:
        allowed = self.valid_next_tokens()
        if strict and token not in allowed:
            raise ValueError(f"Invalid token '{token}' for current state.")

        state = self.state
        state.last_token = token

        if token == "<bos>":
            state.begun = True
            return
        if token == "<eos>":
            state.ended = True
            state.staff_open = False
            return
        if token == "<staff_start>":
            state.staff_open = True
            state.staff_closed = False
            state.measure_open = False
            state.chord_open = False
            state.expecting_duration = False
            state.allow_duration_modifier = False
            state.beat_position = 0.0
            state.pending_tuplet_scale = 1.0
            state.pending_duration_kind = None
            state.pending_tie_start = False
            state.current_voice = None
            state.voice_beat_positions.clear()
            state.header_has_clef = False
            state.header_has_time = False
            state.measure_index_in_staff = 0
            state.measure_has_content = False
            return
        if token == "<staff_end>":
            state.staff_open = False
            state.staff_closed = True
            state.measure_open = False
            state.chord_open = False
            state.expecting_duration = False
            state.allow_duration_modifier = False
            state.beat_position = 0.0
            state.pending_tuplet_scale = 1.0
            state.pending_duration_kind = None
            state.pending_tie_start = False
            state.current_voice = None
            state.voice_beat_positions.clear()
            state.header_has_clef = False
            state.header_has_time = False
            state.measure_index_in_staff = 0
            state.measure_has_content = False
            return
        if token == "<measure_start>":
            if strict and state.measure_index_in_staff == 0 and not (
                state.header_has_clef and state.header_has_time
            ):
                raise ValueError("First measure requires clef and time signature before <measure_start>.")
            state.measure_open = True
            state.beat_position = 0.0
            state.voice_switches_in_measure = 0
            state.current_voice = None
            state.voice_beat_positions.clear()
            state.allow_duration_modifier = False
            state.pending_tuplet_scale = 1.0
            state.pending_duration_kind = None
            state.pending_tie_start = False
            state.measure_index_in_staff += 1
            state.measure_has_content = False
            return
        if token == "<measure_end>":
            state.measure_open = False
            state.chord_open = False
            state.expecting_duration = False
            state.allow_duration_modifier = False
            state.beat_position = 0.0
            state.current_voice = None
            state.last_duration_base = 0.0
            state.pending_tuplet_scale = 1.0
            state.pending_duration_kind = None
            state.pending_tie_start = False
            state.voice_beat_positions.clear()
            state.measure_has_content = False
            return
        if token == "<chord_start>":
            state.chord_open = True
            state.allow_duration_modifier = False
            return
        if token == "<chord_end>":
            state.chord_open = False
            state.expecting_duration = True
            state.allow_duration_modifier = False
            state.pending_duration_kind = "note"
            return

        if token in self.vocab.clef_tokens and not state.measure_open:
            state.header_has_clef = True
            return

        parsed_beats = _beats_from_time_signature(token)
        if parsed_beats is not None and not state.measure_open:
            state.beats_per_measure = parsed_beats
            state.header_has_time = True
            return

        if token in self.vocab.voice_tokens:
            if state.current_voice == token and strict:
                raise ValueError(f"Voice token '{token}' repeated without alternation.")
            if state.current_voice is not None:
                state.voice_beat_positions[state.current_voice] = state.beat_position
            state.current_voice = token
            state.beat_position = state.voice_beat_positions.get(token, 0.0)
            state.voice_switches_in_measure += 1
            state.allow_duration_modifier = False
            return

        if token in TUPLET_BEAT_SCALE:
            if not state.expecting_duration and strict:
                raise ValueError(f"Tuplet token '{token}' is only valid before a duration token.")
            state.pending_tuplet_scale = TUPLET_BEAT_SCALE[token]
            state.allow_duration_modifier = False
            return

        if token in self.vocab.grace_tokens:
            state.expecting_duration = True
            state.allow_duration_modifier = False
            state.pending_duration_kind = "grace"
            state.measure_has_content = True
            return

        if token in self.vocab.note_tokens:
            state.expecting_duration = True
            state.allow_duration_modifier = False
            state.pending_duration_kind = "rest" if token == "rest" else "note"
            state.measure_has_content = True
            return

        if token in self.vocab.base_duration_tokens or token in self.vocab.duration_modifier_tokens:
            self._apply_duration_token(token)
            return

        if token == "tie_start" and state.pending_tie_start:
            state.pending_tie_start = False
        elif state.pending_tie_start:
            state.pending_tie_start = False

        state.allow_duration_modifier = False

    def validate_sequence(self, sequence: Iterable[str], strict: bool = True) -> None:
        self.reset()
        for token in sequence:
            self.step(token, strict=strict)

