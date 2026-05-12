# Stage 3 v3 — train-vs-inference gap diagnostic (revised post-lieder)

**Date:** 2026-05-12
**Status:** original conclusions revised — lieder eval showed demo result was not representative
**Parent:** [data-pipeline-and-capacity-audit](2026-05-12-data-pipeline-and-capacity-audit.md)
**Supersedes:** the original demo-pessimistic framing below
**Final results report:** [stage3-v3-data-rebalance-results](2026-05-12-stage3-v3-data-rebalance-results.md)

## Revised TL;DR

This document was originally written when only the 4-piece HF demo eval had completed (mean onset_f1 0.0510, similar to v2's 0.0589). At that point it looked like the data-rebalance fix dramatically improved training metrics but didn't translate to inference — the same "training improves, inference flat" pattern that disproved the encoder-drift hypothesis (PR #49). The leading hypothesis was that a cache-vs-live encoder feature mismatch (A2 fail) was creating a train-inference distribution shift that small training-data fixes can't close.

The 139-piece lieder eval, which completed an hour later, falsified the demo-pessimistic framing. **Lieder corpus mean onset_f1 = 0.2398** — essentially at the project ship-gate (0.241) and roughly 3× v2's corpus mean (0.0819). The model generalizes. The 4 demo pieces all happened to land in the bottom half of the lieder distribution — a small-sample artifact, not a representative signal.

The cache-vs-live mismatch (A2 fail) is real but is not the urgent bottleneck. The data-rebalance and encoder-freeze fixes DID transfer to inference, just unevenly across pieces. The new diagnostic question is no longer "why didn't training metrics transfer" but "why is the distribution bimodal — why do 24% of pieces score below 0.10 onset_f1 while 11% score above 0.50?"

## Lieder distribution snapshot (139 pieces, post-rebalance)

```
mean: 0.2398  median: 0.1830  stdev: 0.2003  p25: 0.10  p75: 0.30
min: 0.0268   max: 0.8697

  [0.00, 0.05)  n=  8   (5.8%)
  [0.05, 0.10)  n= 25  (18.0%)
  [0.10, 0.20)  n= 46  (33.1%)  <- median band
  [0.20, 0.30)  n= 25  (18.0%)  <- mean lands here
  [0.30, 0.50)  n= 19  (13.7%)
  [0.50, 0.70)  n=  7   (5.0%)
  [0.70, 1.01)  n=  9   (6.5%)
```

The 33 pieces below 0.10 are the new target for any next sub-project; they likely reveal a structural failure mode (specific time signatures? specific layout? specific instrumentation?) that the rebalance didn't address.

## What still holds from the original doc

- **Encoder freeze is verified working.** All 774 encoder keys byte-identical between Stage 2 v2 best.pt and Stage 3 v3 best.pt with zero drift.
- **A2 still fails with max_abs ≈ 514**, and the root cause is preprocessing-mismatch between cache and live encoder (not encoder drift). The cache was built from neither S2v2 nor any encoder we currently use.
- **Frankenstein's verdict still holds**: encoder-side swaps don't fix the residual gap, because the gap was never about encoder weights.
- **The cache-vs-live mismatch is a real source of train-inference shift** — just not the dominant one. The decoder evidently learns enough cache-invariant structure to generalize despite the shift.

## What changed

The framing of the original doc — "training metrics improve, inference doesn't move" — assumed demo was representative. It wasn't. Lieder makes clear that inference DID move (3× corpus mean vs v2). The bottom of the lieder distribution still failing is a more interesting question than the original framing.

## Revised next-experiment priorities

The original document proposed three experiments in priority order. Lieder lowered the priority of all three:

| Experiment | Original priority | Revised priority | Why |
|---|---|---|---|
| 1. Live-encoder inference on a training piece | high | low | A3 already shows the decoder reproduces training data well; the question wasn't whether the decoder works, it was about distribution shift. Lieder partially answered: shift exists but doesn't kill generalization. |
| 2. Rebuild cache from S2v2 encoder + retrain | high | low | 6h compute for an experiment that may yield ~marginal gain; not worth running unless a future sub-project specifically needs the cache parity to be exact. |
| 3. Train without cache (live encoding) | medium | low | Same logic. The cache works well enough; the bottleneck is elsewhere. |

A more useful next experiment, given lieder: **stratified failure-mode analysis on the bottom ~33 lieder pieces** (those scoring < 0.10 onset_f1). Cluster them on observable features (time signature, key signature, staff count, page layout, instrument set, token-sequence length) and find the consistent driver. That gives the next sub-project a concrete target instead of a generic "improve the model."

## Recommendation

Treat this document as historical context — it captures the false-start where demo looked dispositive. The final results report supersedes it. Any reader pointing at this for next-step direction should read the final results report first.
