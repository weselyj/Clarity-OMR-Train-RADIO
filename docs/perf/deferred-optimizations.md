# Deferred Performance Optimizations

> Items considered during the cu132 + DataLoader work session (2026-04-27) that we
> measured-then-deferred, not items we forgot. Each entry has a **trigger condition**
> describing when it becomes worth re-evaluating, plus a pointer to the original
> review/spec that motivated it.

The closure rationale lives on the original issues: #2 (Pipeline throughput tracker) was closed when this file was created. If a trigger fires, file a fresh focused issue rather than re-opening the closed tracker.

---

## RAM cache for resized `uint8` image tensors

**Original spec:** issue #2 Tier 3 #4. Codex's PyTorch-throughput review (2026-04-26).

**Why deferred:** the [cu132 rollout](2026-04-27-cu132-rollout.md) bench showed CPU pipeline = 3.6% of wall, and the DataLoader rewrite (PR #14) drops that further by overlapping it with GPU compute via worker prefetch. There is no pressure on the I/O side that a RAM cache would relieve. The 32 GB host has plenty of OS-level page cache covering the working set already.

**Triggers — revisit if any are true:**

1. **Memory budget shrinks.** A rented GPU box with less RAM (e.g. 16 GB) where OS page cache can't absorb the dataset's hot subset. Symptom in the profile data: `cpu_h2d_ms` p95 climbs above the current ~1 ms baseline, suggesting fresh disk reads instead of cache hits.
2. **Repeated-read pattern in profile data.** If `cpu_encode_ms` p95 climbs while `cpu_augment_ms` stays flat, the worker is re-doing image decode that a cache could absorb. Capture this with `--profile-step-timing` and compare against the cu132 baseline in `2026-04-27-cu132-rollout.md`.
3. **Curriculum that resamples a small subset frequently.** A future stage that weights 100-200 samples at 80% of the mix would benefit from caching those specific items.

**What to do when triggered:** start small — bound the cache at 24-48 GB of CPU RAM, store images as `uint8` (not `float32`), use lazy population per worker, and instrument hit/miss. The original spec at issue #2 Tier 3 #4 has the design.

**Estimated effort:** 1-2 days including a regression bench against the cu132 baseline.

---

## Selective activation checkpointing

**Original spec:** issue #2 Tier 3 #8. Codex's PyTorch-throughput review (2026-04-26).

**Why deferred:** at the current Stage 2 config (batch=2, max_seq_len=384, bf16, RTX 5090), the model fits with healthy memory margin. There is no forcing function. The real win from activation checkpointing is enabling a *larger* batch (3 or 4) at the cost of recomputing some forward activations during backward — and the encoder forward already dominates wall time, so trading more compute for less memory has a poor ratio when memory isn't the bottleneck.

**Triggers — revisit if any are true:**

1. **`torch.OutOfMemoryError` shows up in training logs.** Even once. The current batch=2 has margin, but a future model variant or longer sequence could push past it.
2. **Need to move from batch=2 to batch=3 or 4.** Either because a new stage's loss curve demands more samples per gradient step, or because we're benchmarking against a paper that quotes batch=4. Trade-off worth re-evaluating with current toolchain timings.
3. **Switching to a longer max_sequence_length.** Stage 3 stays at 512. If a future stage goes to 768 or 1024 (e.g. for full-page polyphonic with more measures per crop), activations grow quadratically with seq, and checkpointing becomes the cheap fix.

**What to do when triggered:** start with `torch.utils.checkpoint.checkpoint_sequential` on the decoder layers (small, stable surface) before considering the encoder (RADIO is `trust_remote_code` — risky). The original spec at issue #2 Tier 3 #8 has more detail.

**Estimated effort:** 1 day for decoder-only, 2-3 days if encoder needs it (mostly debugging RADIO's wrapped state).

---

## Maintenance

When adding a new entry to this doc, follow the same shape:

- **Original spec** — link to the issue/PR/review where the idea originated.
- **Why deferred** — the measured evidence for the decision. Profile numbers, memory headroom, etc. Not "we'll get to it later."
- **Triggers** — observable conditions, not vague hopes. Each trigger should be either a profile metric we already capture or a system event (OOM, new stage spec).
- **What to do when triggered** — a starting point, not a full plan. The plan should be written when the trigger fires, against the data at that time.
- **Estimated effort** — useful for triaging when something else breaks alongside it.

If a trigger fires, file a fresh focused issue rather than re-opening the closed tracker. Reference this file in the new issue so future maintainers see the deferral context.
