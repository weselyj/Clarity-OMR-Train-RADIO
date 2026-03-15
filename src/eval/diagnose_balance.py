#!/usr/bin/env python3
"""Diagnose why measure_balance_rate is low in eval predictions."""
import json, math, sys

BASE_DURATION_BEATS = {
    '_whole': 4.0, '_half': 2.0, '_quarter': 1.0, '_eighth': 0.5,
    '_sixteenth': 0.25, '_thirty_second': 0.125, '_sixty_fourth': 0.0625,
}
TUPLET_SCALE = {'<tuplet_3>': 2/3, '<tuplet_5>': 4/5, '<tuplet_6>': 4/6, '<tuplet_7>': 4/7}
VOICE_TOKENS = {'<voice_1>', '<voice_2>', '<voice_3>', '<voice_4>'}

def parse_time_sig(t):
    if not t or not t.startswith('timeSignature-'):
        return None
    parts = t[len('timeSignature-'):].split('/')
    if len(parts) == 2:
        try:
            return float(parts[0]) * (4.0 / float(parts[1]))
        except Exception:
            return None
    return None

def analyze(path):
    reasons = {'whole_rest_non4': 0, 'wrong_time_sig': 0, 'voice_imbalance': 0, 'other': 0}
    total_measures = 0
    unbalanced = 0
    examples = []

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            pred = row['pred_tokens']
            gt = row.get('gt_tokens', [])

            pred_ts = next((t for t in pred if t.startswith('timeSignature-')), None)
            gt_ts = next((t for t in gt if t.startswith('timeSignature-')), None)
            bpm = parse_time_sig(pred_ts) or 4.0

            in_measure = False
            voice_pos = {}
            cur_voice = '<voice_1>'
            beat = 0.0
            tuplet = 1.0
            measure_tokens = []

            for i, tok in enumerate(pred):
                if tok == '<measure_start>':
                    if in_measure:
                        voice_pos[cur_voice] = beat
                        vals = [v for v in voice_pos.values() if v is not None]
                        max_beat = max(vals) if vals else 0.0
                        tol = max(0.2, 0.03 * max(1.0, bpm))
                        total_measures += 1
                        if not (math.isfinite(max_beat) and abs(max_beat - bpm) <= tol):
                            unbalanced += 1
                            has_whole_rest = 'rest' in measure_tokens and '_whole' in measure_tokens
                            if has_whole_rest and abs(bpm - 4.0) > 0.01:
                                reasons['whole_rest_non4'] += 1
                            elif pred_ts != gt_ts:
                                reasons['wrong_time_sig'] += 1
                            elif len(voice_pos) > 1:
                                reasons['voice_imbalance'] += 1
                            else:
                                reasons['other'] += 1
                                if len(examples) < 8:
                                    examples.append({
                                        'sample': row['sample_id'][:60],
                                        'dataset': row.get('dataset', '?'),
                                        'bpm': bpm,
                                        'max_beat': round(max_beat, 3),
                                        'diff': round(max_beat - bpm, 3),
                                        'mtoks': measure_tokens[:25],
                                    })
                    in_measure = True
                    cur_voice = '<voice_1>'
                    voice_pos = {cur_voice: 0.0}
                    beat = 0.0
                    tuplet = 1.0
                    measure_tokens = []
                    continue

                if tok == '<measure_end>':
                    if in_measure:
                        voice_pos[cur_voice] = beat
                        vals = [v for v in voice_pos.values() if v is not None]
                        max_beat = max(vals) if vals else 0.0
                        tol = max(0.2, 0.03 * max(1.0, bpm))
                        total_measures += 1
                        if not (math.isfinite(max_beat) and abs(max_beat - bpm) <= tol):
                            unbalanced += 1
                            has_whole_rest = 'rest' in measure_tokens and '_whole' in measure_tokens
                            if has_whole_rest and abs(bpm - 4.0) > 0.01:
                                reasons['whole_rest_non4'] += 1
                            elif pred_ts != gt_ts:
                                reasons['wrong_time_sig'] += 1
                            elif len(voice_pos) > 1:
                                reasons['voice_imbalance'] += 1
                            else:
                                reasons['other'] += 1
                                if len(examples) < 8:
                                    examples.append({
                                        'sample': row['sample_id'][:60],
                                        'dataset': row.get('dataset', '?'),
                                        'bpm': bpm,
                                        'max_beat': round(max_beat, 3),
                                        'diff': round(max_beat - bpm, 3),
                                        'mtoks': measure_tokens[:25],
                                    })
                    in_measure = False
                    continue

                if tok in VOICE_TOKENS and in_measure:
                    voice_pos[cur_voice] = beat
                    cur_voice = tok
                    beat = voice_pos.get(cur_voice, 0.0)
                    measure_tokens.append(tok)
                    continue

                if tok in TUPLET_SCALE:
                    tuplet = TUPLET_SCALE[tok]
                    measure_tokens.append(tok)
                    continue

                if tok in BASE_DURATION_BEATS and in_measure:
                    b = BASE_DURATION_BEATS[tok] * tuplet
                    if i + 1 < len(pred) and pred[i + 1] == '_dot':
                        b *= 1.5
                    elif i + 1 < len(pred) and pred[i + 1] == '_double_dot':
                        b *= 1.75
                    if i > 0 and pred[i - 1].startswith('gracenote-'):
                        b = 0.0
                    beat += b
                    voice_pos[cur_voice] = beat
                    tuplet = 1.0

                if tok.startswith('timeSignature-'):
                    new_bpm = parse_time_sig(tok)
                    if new_bpm:
                        bpm = new_bpm

                measure_tokens.append(tok)

    print(f'Total measures: {total_measures}')
    print(f'Unbalanced:     {unbalanced} ({unbalanced / max(1, total_measures) * 100:.1f}%)')
    print(f'Balance rate:   {(total_measures - unbalanced) / max(1, total_measures) * 100:.1f}%')
    print()
    print('Reasons for imbalance:')
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        pct = v / max(1, unbalanced) * 100
        print(f'  {k:25s} {v:4d}  ({pct:.1f}%)')
    print()
    print('Example unbalanced (other category):')
    for e in examples:
        print(f'  [{e["dataset"]}] bpm={e["bpm"]} max_beat={e["max_beat"]} diff={e["diff"]}')
        print(f'    sample: {e["sample"]}')
        print(f'    tokens: {" ".join(e["mtoks"][:20])}')
        print()

if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else 'src/eval/checkpoint_eval_finetune_conservative_600/stageb_eval_predictions_eval.jsonl'
    analyze(path)
