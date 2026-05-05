#!/usr/bin/env python3
"""Locked token vocabulary utilities for OMR."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


STRUCTURAL_TOKENS = [
    "<bos>",
    "<eos>",
    "<pad>",
    "<staff_start>",
    "<staff_end>",
    "<measure_start>",
    "<measure_end>",
    "<voice_1>",
    "<voice_2>",
    "<voice_3>",
    "<voice_4>",
    "<chord_start>",
    "<chord_end>",
    "<tuplet_3>",
    "<tuplet_5>",
    "<tuplet_6>",
    "<tuplet_7>",
]

STAFF_INDEX_MARKER_TOKENS = [f"<staff_idx_{i}>" for i in range(8)]

# v3 enharmonic extension: Cb/Fb/B#/E# × octaves 2-6 = 20 tokens.
# These would otherwise collapse to B/E/C/F via semitone normalisation, losing
# spelling fidelity that music21 preserves when parsing kern files.
# APPEND-ONLY: keep after STAFF_INDEX_MARKER_TOKENS to preserve all prior IDs.
EXTENDED_NOTE_TOKENS: list[str] = (
    [f"note-Cb{o}" for o in range(2, 7)]
    + [f"note-Fb{o}" for o in range(2, 7)]
    + [f"note-B#{o}" for o in range(2, 7)]
    + [f"note-E#{o}" for o in range(2, 7)]
)

# v3 phase 2: octave-1 sub-bass notes (F1, G1, A1, Bb1 etc.) were previously
# absent from the vocab, causing _normalize_note_pitch_symbol to clamp them all
# to note-C2 (the lowest available token).  Add the full 21-token set covering
# the same pitch/accidental combinations as octaves 2-6.
# APPEND-ONLY: keep after EXTENDED_NOTE_TOKENS to preserve all prior IDs.
OCTAVE_1_NOTE_TOKENS: list[str] = [
    "note-A1", "note-Ab1", "note-A#1",
    "note-B1", "note-Bb1", "note-B#1",
    "note-C1", "note-Cb1", "note-C#1",
    "note-D1", "note-Db1", "note-D#1",
    "note-E1", "note-Eb1", "note-E#1",
    "note-F1", "note-Fb1", "note-F#1",
    "note-G1", "note-Gb1", "note-G#1",
]

DURATION_TOKENS = [
    "_whole",
    "_half",
    "_quarter",
    "_eighth",
    "_sixteenth",
    "_thirty_second",
    "_sixty_fourth",
    "_dot",
    "_double_dot",
]

LEGACY_TOKEN_ALIASES = {
    "_whole_rest": "_whole",
    "_half_rest": "_half",
    "_quarter_rest": "_quarter",
    "_eighth_rest": "_eighth",
}

CLEF_TOKENS = [
    "clef-G2",
    "clef-F4",
    "clef-C3",
    "clef-C4",
    "clef-C1",
    "clef-G2_8vb",
    "clef-G2_8va",
]

TIME_SIGNATURE_TOKENS = [
    "timeSignature-4/4",
    "timeSignature-3/4",
    "timeSignature-2/4",
    "timeSignature-6/8",
    "timeSignature-2/2",
    "timeSignature-3/8",
    "timeSignature-9/8",
    "timeSignature-12/8",
    "timeSignature-C",
    "timeSignature-C/",
    "timeSignature-5/4",
    "timeSignature-7/8",
    "timeSignature-6/4",
    "timeSignature-3/2",
    "timeSignature-other",
]

BARLINE_TOKENS = [
    "barline",
    "double_barline",
    "final_barline",
    "repeat_start",
    "repeat_end",
    "repeat_both",
]

ARTICULATION_TOKENS = [
    "staccato",
    "accent",
    "tenuto",
    "marcato",
    "fermata",
    "trill",
    "mordent",
    "turn",
    "staccatissimo",
    "portato",
    "sforzando",
    "snap_pizz",
]

DYNAMIC_TOKENS = [
    "dynamic-ppp",
    "dynamic-pp",
    "dynamic-p",
    "dynamic-mp",
    "dynamic-mf",
    "dynamic-f",
    "dynamic-ff",
    "dynamic-fff",
    "dynamic-sfz",
    "dynamic-fp",
]

CONNECTION_TOKENS = [
    "tie_start",
    "tie_end",
    "slur_start",
    "slur_end",
    "cresc_start",
    "cresc_end",
    "decresc_start",
    "decresc_end",
]

OTHER_NOTATION_TOKENS = [
    "ottava_8va_start",
    "ottava_8va_end",
    "ottava_8vb_start",
    "ottava_8vb_end",
    "pedal_start",
    "pedal_end",
    "trem_single",
    "trem_double",
    "trem_triple",
    "coda",
    "segno",
    "breath_mark",
    "caesura",
    "arco",
    "pizz",
]

TEMPO_TOKENS = [
    "tempo-Largo",
    "tempo-Larghetto",
    "tempo-Lento",
    "tempo-Adagio",
    "tempo-Andante",
    "tempo-Andantino",
    "tempo-Moderato",
    "tempo-Allegretto",
    "tempo-Allegro",
    "tempo-Vivace",
    "tempo-Presto",
    "tempo-Prestissimo",
    "tempo-Grave",
    "tempo-Grazioso",
    "tempo-Maestoso",
    "tempo-Animato",
    "tempo-Agitato",
    "tempo-Con_brio",
    "tempo-Tempo_giusto",
    "tempo-A_tempo",
    "tempo-Rubato",
    "tempo-Ritenuto",
    "tempo-Ritardando",
    "tempo-Accelerando",
    "tempo-Poco_a_poco",
    "tempo-Piu_mosso",
    "tempo-Meno_mosso",
    "tempo-Ad_libitum",
    "tempo-Sostenuto",
    "tempo-Non_troppo",
    "tempo-Assai",
    "tempo-Comodo",
    "tempo-Giocoso",
    "tempo-Stringendo",
    "tempo-Mosso",
    "tempo-Calando",
    "tempo-Senza_misura",
    "tempo-Liberamente",
    "tempo-Tempo_primo",
    "tempo-In_tempo",
    "tempo-L_istesso_tempo",
    "tempo-Meno_vivo",
    "tempo-Piu_vivo",
    "tempo-Veloce",
    "tempo-Quasi",
    "tempo-Tranquillo",
    "tempo-Con_moto",
    "tempo-Senza_tempo",
    "tempo-Sempre",
    "tempo-Pesante",
]

EXPRESSION_TOKENS = [
    "expr-dolce",
    "expr-cantabile",
    "expr-espressivo",
    "expr-legato",
    "expr-marcato",
    "expr-a_tempo",
    "expr-rit",
    "expr-accel",
    "expr-rall",
    "expr-poco_a_poco",
    "expr-molto",
    "expr-sempre",
    "expr-subito",
    "expr-piu",
    "expr-meno",
    "expr-tenuto",
    "expr-sostenuto",
    "expr-con_brio",
    "expr-giocoso",
    "expr-grandioso",
    "expr-maestoso",
    "expr-delicato",
    "expr-brillante",
    "expr-tranquillo",
    "expr-con_fuoco",
    "expr-vigoroso",
    "expr-appassionato",
    "expr-misterioso",
    "expr-lamentoso",
    "expr-scherzando",
    "expr-risoluto",
    "expr-pesante",
    "expr-leggiero",
    "expr-stretto",
    "expr-dim",
    "expr-cresc",
    "expr-senza_pedale",
    "expr-con_pedale",
    "expr-con_sordino",
    "expr-senza_sordino",
    "expr-pizz",
    "expr-arco",
    "expr-morendo",
    "expr-smorzando",
    "expr-mezzo_voce",
    "expr-sotto_voce",
    "expr-alla_breve",
    "expr-sostenendo",
    "expr-energico",
    "expr-con_anima",
    "expr-con_espressione",
    "expr-con_grazia",
    "expr-alla_marcia",
    "expr-allargando",
    "expr-rubato",
    "expr-tenerezza",
    "expr-ma_non_troppo",
    "expr-non_legato",
    "expr-ben_marcato",
    "expr-sf",
    "expr-ffz",
    "expr-subito_p",
    "expr-subito_f",
    "expr-ritenuto",
    "expr-vivo",
    "expr-con_moto",
    "expr-lusingando",
    "expr-brio",
    "expr-furioso",
    "expr-dolcissimo",
    "expr-appena",
    "expr-mesto",
    "expr-risolvendo",
    "expr-flebile",
    "expr-con_affetto",
    "expr-animando",
    "expr-con_slancio",
    "expr-senza_rall",
]


def _dedupe(tokens: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    output: List[str] = []
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        output.append(token)
    return output


def build_pitch_tokens() -> List[str]:
    pitch_classes = [
        "C",
        "C#",
        "Db",
        "D",
        "D#",
        "Eb",
        "E",
        "F",
        "F#",
        "Gb",
        "G",
        "G#",
        "Ab",
        "A",
        "A#",
        "Bb",
        "B",
    ]
    note_tokens: List[str] = []
    for octave in range(2, 7):
        for pitch_class in pitch_classes:
            note_tokens.append(f"note-{pitch_class}{octave}")
    note_tokens.append("note-C7")
    note_tokens.append("rest")
    return note_tokens


def build_gracenote_tokens() -> List[str]:
    # 35 natural grace-note variants (7 pitch classes x 5 octaves).
    pitch_classes = ["C", "D", "E", "F", "G", "A", "B"]
    grace_tokens: List[str] = []
    for octave in range(2, 7):
        for pitch_class in pitch_classes:
            grace_tokens.append(f"gracenote-{pitch_class}{octave}")
    return grace_tokens


def build_key_signature_tokens() -> List[str]:
    major = [
        "CM",
        "GM",
        "DM",
        "AM",
        "EM",
        "BM",
        "F#M",
        "C#M",
        "FM",
        "BbM",
        "EbM",
        "AbM",
        "DbM",
        "GbM",
        "CbM",
    ]
    minor = [
        "Am",
        "Em",
        "Bm",
        "F#m",
        "C#m",
        "G#m",
        "D#m",
        "A#m",
        "Dm",
        "Gm",
        "Cm",
        "Fm",
        "Bbm",
        "Ebm",
        "Abm",
    ]
    tokens = [f"keySignature-{key}" for key in major + minor]
    tokens.append("keySignature-none")
    return tokens


def build_default_token_list() -> List[str]:
    return _dedupe(
        [
            *STRUCTURAL_TOKENS,
            *build_pitch_tokens(),
            *build_gracenote_tokens(),
            *DURATION_TOKENS,
            *CLEF_TOKENS,
            *build_key_signature_tokens(),
            *TIME_SIGNATURE_TOKENS,
            *BARLINE_TOKENS,
            *ARTICULATION_TOKENS,
            *DYNAMIC_TOKENS,
            *CONNECTION_TOKENS,
            *OTHER_NOTATION_TOKENS,
            *TEMPO_TOKENS,
            *EXPRESSION_TOKENS,
            *STAFF_INDEX_MARKER_TOKENS,
            # v3 extension: enharmonic spelling tokens (Cb/Fb/B#/E# × octaves 2-6)
            *EXTENDED_NOTE_TOKENS,
            # v3 phase 2: octave-1 sub-bass notes — always appended last
            *OCTAVE_1_NOTE_TOKENS,
        ]
    )


@dataclass(frozen=True)
class OMRVocabulary:
    tokens: Sequence[str]
    token_to_id: Dict[str, int]
    id_to_token: Dict[int, str]
    structural_tokens: Set[str]
    pitch_tokens: Set[str]
    note_tokens: Set[str]
    grace_tokens: Set[str]
    duration_tokens: Set[str]
    base_duration_tokens: Set[str]
    duration_modifier_tokens: Set[str]
    voice_tokens: Set[str]
    clef_tokens: Set[str]
    key_signature_tokens: Set[str]
    time_signature_tokens: Set[str]
    barline_tokens: Set[str]
    in_measure_attribute_tokens: Set[str]

    @property
    def size(self) -> int:
        return len(self.tokens)

    def encode(self, token_sequence: Sequence[str], strict: bool = True) -> List[int]:
        encoded: List[int] = []
        for token in token_sequence:
            normalized = LEGACY_TOKEN_ALIASES.get(token, token)
            token_id = self.token_to_id.get(normalized)
            if token_id is None:
                if strict:
                    raise KeyError(f"Unknown token: {token}")
                continue
            encoded.append(token_id)
        return encoded

    def decode(self, ids: Sequence[int]) -> List[str]:
        decoded: List[str] = []
        for token_id in ids:
            token = self.id_to_token.get(token_id)
            if token is None:
                raise KeyError(f"Unknown token id: {token_id}")
            decoded.append(token)
        return decoded


def build_default_vocabulary() -> OMRVocabulary:
    tokens = build_default_token_list()
    token_to_id = {token: idx for idx, token in enumerate(tokens)}
    id_to_token = {idx: token for token, idx in token_to_id.items()}

    note_tokens = {token for token in tokens if token.startswith("note-")} | {"rest"}
    grace_tokens = {token for token in tokens if token.startswith("gracenote-")}
    pitch_tokens = (note_tokens - {"rest"}) | grace_tokens
    duration_tokens = set(DURATION_TOKENS)
    base_duration_tokens = duration_tokens - {"_dot", "_double_dot"}
    duration_modifier_tokens = {"_dot", "_double_dot"}
    voice_tokens = {token for token in tokens if token.startswith("<voice_")}

    in_measure_attribute_tokens = set(
        [
            *BARLINE_TOKENS,
            *ARTICULATION_TOKENS,
            *DYNAMIC_TOKENS,
            *CONNECTION_TOKENS,
            *OTHER_NOTATION_TOKENS,
            *TEMPO_TOKENS,
            *EXPRESSION_TOKENS,
        ]
    )

    return OMRVocabulary(
        tokens=tokens,
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        structural_tokens=set(STRUCTURAL_TOKENS) | set(STAFF_INDEX_MARKER_TOKENS),
        pitch_tokens=pitch_tokens,
        note_tokens=note_tokens,
        grace_tokens=grace_tokens,
        duration_tokens=duration_tokens,
        base_duration_tokens=base_duration_tokens,
        duration_modifier_tokens=duration_modifier_tokens,
        voice_tokens=voice_tokens,
        clef_tokens=set(CLEF_TOKENS),
        key_signature_tokens={token for token in tokens if token.startswith("keySignature-")},
        time_signature_tokens=set(TIME_SIGNATURE_TOKENS),
        barline_tokens=set(BARLINE_TOKENS),
        in_measure_attribute_tokens=in_measure_attribute_tokens,
    )
