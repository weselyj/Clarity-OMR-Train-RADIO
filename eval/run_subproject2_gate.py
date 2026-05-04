#!/usr/bin/env python3
"""Run Subproject 2 gate evaluation against a Stage 2 systems checkpoint.

Three gate conditions:
  1. val_loss <= 0.10
  2. Marker correctness >= 18/20 (random val pages)
  3. Per-staff syntactic validity >= 95%

Output: a printed summary plus a JSON file at eval/subproject2_gate_results.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.tokenizer.vocab import build_default_vocabulary
from src.decoding.grammar_fsa import GrammarFSA


def _check_marker_structure(tokens: List[str]) -> bool:
    """A decoded sequence is marker-correct iff:
      - For each <staff_start>, the next token is <staff_idx_N>
      - <staff_idx_N> values are strictly ascending and start at 0
      - Number of distinct markers equals number of <staff_start>
    """
    starts = [i for i, t in enumerate(tokens) if t == "<staff_start>"]
    if not starts:
        return True  # nothing to check
    seen = []
    for s in starts:
        if s + 1 >= len(tokens):
            return False
        nxt = tokens[s + 1]
        if not nxt.startswith("<staff_idx_"):
            return False
        try:
            idx = int(nxt[len("<staff_idx_"):-1])
        except ValueError:
            return False
        seen.append(idx)
    return seen == list(range(len(seen)))


def _check_syntactic_validity(tokens: List[str]) -> bool:
    """Run the predicted token sequence through the grammar FSA in strict mode."""
    fsa = GrammarFSA()
    try:
        fsa.validate_sequence(tokens, strict=True)
    except Exception:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--num-spotcheck", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("eval/subproject2_gate_results.json"))
    parser.add_argument("--val-loss", type=float, default=None,
                        help="If provided, use this val_loss instead of recomputing")
    args = parser.parse_args()

    # 1) val_loss — if not provided via CLI, run the trainer's val pass.
    if args.val_loss is not None:
        val_loss = float(args.val_loss)
    else:
        # Defer to the trainer's own validation logic. Caller should pass --val-loss
        # captured from the training run's logs to keep this script lightweight.
        raise SystemExit("Pass --val-loss explicitly (extract from training log).")

    # 2) Marker correctness + 3) syntactic validity on random val sample.
    # Decoding requires loading the model; for plan simplicity we read predicted
    # token sequences from a sidecar file produced by an inference script.
    pred_path = args.val_manifest.parent / "stage2_systems_predictions.jsonl"
    if not pred_path.exists():
        raise SystemExit(f"Missing predictions sidecar: {pred_path}. Run the prediction script first.")

    rng = random.Random(args.seed)
    all_preds: List[Dict[str, object]] = []
    with pred_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_preds.append(json.loads(line))
    rng.shuffle(all_preds)
    sample = all_preds[: args.num_spotcheck]

    marker_ok = sum(1 for p in sample if _check_marker_structure(p["predicted_tokens"]))
    syntax_ok = sum(1 for p in all_preds if _check_syntactic_validity(p["predicted_tokens"]))
    syntax_rate = syntax_ok / max(1, len(all_preds))

    results = {
        "val_loss": val_loss,
        "val_loss_pass": val_loss <= 0.10,
        "marker_correct_count": marker_ok,
        "marker_total": len(sample),
        "marker_correct_pass": marker_ok >= 18,
        "syntax_valid_count": syntax_ok,
        "syntax_total": len(all_preds),
        "syntax_rate": syntax_rate,
        "syntax_valid_pass": syntax_rate >= 0.95,
    }
    overall = all([results["val_loss_pass"], results["marker_correct_pass"], results["syntax_valid_pass"]])
    results["overall_gate_pass"] = overall

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 60)
    print(f"  val_loss:       {val_loss:.4f}  ({'PASS' if results['val_loss_pass'] else 'FAIL'} <= 0.10)")
    print(f"  marker correct: {marker_ok}/{len(sample)}  ({'PASS' if results['marker_correct_pass'] else 'FAIL'} >= 18)")
    print(f"  syntax valid:   {syntax_ok}/{len(all_preds)} = {syntax_rate:.3f}  ({'PASS' if results['syntax_valid_pass'] else 'FAIL'} >= 0.95)")
    print(f"  OVERALL:        {'GATE PASS' if overall else 'GATE FAIL'}")
    print("=" * 60)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
