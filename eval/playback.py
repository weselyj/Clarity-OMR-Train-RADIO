"""Playback-equivalent note-level accuracy via mir_eval.transcription.

**Primary Phase 0 leaderboard metric.**

Renders both MusicXML files to note intervals (onset_sec, offset_sec) plus pitch-in-Hz
at a fixed nominal tempo (120 BPM -> 1 quarter = 0.5 s), then scores with
mir_eval.transcription.precision_recall_f1_overlap. Tolerant of enharmonic
differences (canonicalize erases them) and of small rhythm drift (onset
tolerance 50 ms is mir_eval's default for piano transcription).

Part/staff assignment is ignored -- two scores with pitches in swapped hands
will score identically. This matches the spec's framing of playback as
"how does it sound" rather than "how is it notated".

offset_ratio=None -> onset-only matching (per Phase 0 spec).
"""
from pathlib import Path
from typing import Union
import numpy as np
from music21 import stream
import mir_eval
from eval.canonicalize import canonicalize, extract_midi_sequence

NOMINAL_TEMPO_BPM = 120.0
ONSET_TOLERANCE_SEC = 0.05


def playback_f(
    pred: Union[Path, str, stream.Score],
    gt: Union[Path, str, stream.Score],
    tempo_bpm: float = NOMINAL_TEMPO_BPM,
) -> dict[str, float]:
    """Return {'precision', 'recall', 'f'} scored at the given tempo."""
    pred_intervals, pred_hz = _to_mir_eval(pred, tempo_bpm)
    gt_intervals, gt_hz = _to_mir_eval(gt, tempo_bpm)

    precision, recall, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_intervals=gt_intervals,
        ref_pitches=gt_hz,
        est_intervals=pred_intervals,
        est_pitches=pred_hz,
        onset_tolerance=ONSET_TOLERANCE_SEC,
        offset_ratio=None,
    )
    return {"precision": float(precision), "recall": float(recall), "f": float(f)}


def _to_mir_eval(source, tempo_bpm: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert a MusicXML source to (intervals_Nx2_sec, pitches_N_hz) for mir_eval.

    Uses canonicalize() + extract_midi_sequence() to share the offset/enharmonic
    normalization with the other scorers.
    """
    events = extract_midi_sequence(canonicalize(source))
    if not events:
        return np.zeros((0, 2)), np.zeros(0)

    qps = tempo_bpm / 60.0  # quarters per second
    intervals = []
    pitches_hz = []
    for midi_pitch, onset_q, duration_q in events:
        start_s = onset_q / qps
        end_s = start_s + max(duration_q / qps, 1e-3)
        intervals.append([start_s, end_s])
        # mir_eval expects pitch in Hz: A4 (MIDI 69) = 440 Hz
        pitches_hz.append(440.0 * (2 ** ((midi_pitch - 69) / 12)))

    return np.array(intervals), np.array(pitches_hz)
