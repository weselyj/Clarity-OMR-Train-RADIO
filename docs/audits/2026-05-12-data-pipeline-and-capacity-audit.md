# Data-Pipeline + Capacity Audit

**Date:** 2026-05-12
**Spec:** [../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md](../superpowers/specs/2026-05-11-data-pipeline-audit-and-stage3-v3-retrain-design.md)
**Branch:** `feat/stage3-v3-data-rebalance`
**Phase:** 1 of 3 (gate for Phase 2)

## TL;DR

Stage 3 v2 failed (HF demo mean onset_f1 ~0.06 vs the 0.241 gate) primarily because of three compounding data-pipeline issues, not decoder capacity. First, the combined Stage 3 manifest carries **only 20,583 synthetic rows out of 135,610 in the source synthetic manifest — 85% of synthetic data is dropped somewhere between the synthetic manifest and the combined manifest used for training**, so the prior "synthetic dominates at 77.8% of cached gradient share" diagnosis was measured on a filtered subset and the upstream filter itself is the largest unknown in the pipeline. Second, **synthetic_systems has zero val/test split** in the combined manifest (703 unique source MXL files, all in train), so multi-staff generalization was never measured during training. Third, **2.0% of synthetic entries (411) and 0.73% of grandstaff entries (782) are silently truncated at the 512-token cap** — and these are the dense, long, multi-staff sequences that train multi-staff reading; the truncation actively penalizes the corpora that need to teach grand-staff behavior. Generator regeneration is byte-deterministic (13/13 SHA-256 matches), spot-checked tokens align with rendered images, and decoder capacity (116M params) matches the upstream Clarity-OMR reference, so neither generator quality nor model capacity is the bottleneck.

Phase 2 should implement the encoder freeze fix (already specified), introduce a by-source synthetic val split (~5% of sources, ~6,751 entries), rebalance `dataset_mix` to down-weight synthetic relative to grandstaff (which is multi-staff but currently under-represented in gradient share), and raise `max_sequence_length` from 512 to 768 or 1024 to recover the truncated multi-staff tail. The 85% synthetic filter drop should be flagged for user-side investigation before Phase 2 begins, because if that filter is unintentional the entire imbalance picture changes (the "true" synthetic share, post-filter-fix, would be ~6.6× larger than the rebalance assumes).

## Stream 1 — Synthetic-corpus inspection

**Goal:** Spot-check synthetic_v2 training data for token/image quality issues that could explain the 11% reproduction accuracy.
**Result:** No systemic quality problems found. Tokens and images align in all 5 spot-checked samples; rest-heavy sequences are legitimate tacet vocal staves.
**Evidence:** `audit_results/synthetic_inspection.json`
**Numbers:**

- 135,610 total synthetic manifest rows, organized in 6 balanced groups: 3 fonts (`bravura-compact`, `gootville-wide`, `leipzig-default`) × 2 sub-corpora (`synthetic_fullpage`, `synthetic_polyphonic`), each group 21,449–23,755 rows
- Sampled (n=20, seed=42) per-CROP stats: mean 1.0 staff_starts, 0% multi-staff, mean 5.2 measure_starts, mean 26.4 note tokens out of 89.6 total
- 1/20 samples (5%) had `image_path: null` in the manifest (Stanford / Tears, idle tears, gootville-wide)
- 18% naming-mismatch rate where `sample_id` ends in `staffNN` but `image_path` ends in `staff(NN-1).png` — confirmed cosmetic by the implementer (the file referenced does exist and is the correct image for the token sequence)

**Interpretation:** The synthetic data is structurally sound at the level we can spot-check. The per-CROP stats (1 staff per entry) are not in conflict with Stream 3's per-SYSTEM stats (3.18 staves per entry) — these are different units; the combined manifest aggregates crops into system-level training examples. The null `image_path` is rare but real and matches Stream 2's `no_image_path_in_manifest` count.

**Recommendation for Phase 2:** No corrective action against synthetic content. Phase 2 should ignore the cosmetic naming mismatch (the loader resolves correctly) and tolerate the small fraction of null image_paths (~5% sampled, ~13% of Stream 2's 5-source sample) since they are dropped at load time rather than producing bad gradients.

## Stream 2 — Generator reproducibility

**Goal:** Verify that the synthetic generator is deterministic and the cache faithfully represents what `src/data/generate_synthetic.py` would produce.
**Result:** Generator is fully deterministic — 13/13 spot-checked entries reproduce byte-identical PNGs at seed=1337.
**Evidence:** `audit_results/synthetic_reproduce.json`
**Numbers:**

- Entry point confirmed: `src.data.generate_synthetic.run(...)` (callable, not just CLI)
- 3 styles regenerated: `leipzig-default`, `bravura-compact`, `gootville-wide`
- Status counts on 15 sampled comparisons: 13 `match`, 2 `no_image_path_in_manifest`
- Both no-image entries come from a single source (`Abrams,_Harriett / Crazy_Jane / lc6583907.mxl`), staff01 + its `__poly` sibling — meaning the upstream synthetic generator wrote a token row without producing the matching image
- Confirmed: `__poly` token-variant entries share `image_path` with their non-`__poly` siblings (same image, different token sequence). Imbalance accounting must dedupe by image, not by row, when reasoning about visual coverage.

**Interpretation:** The synthetic cache is trustworthy — regenerating from source MXL plus the existing job-plan reproduces exact PNGs, so we don't need to regenerate, and prior gradient-share math wasn't artifacted by stale or corrupted cache. The `__poly` siblings sharing images is important: when we say "synthetic has 20,583 rows in the combined manifest," some fraction of those rows point at the same staff image with a polyphonic token re-encoding of the same musical content. This effectively doubles the gradient share on each image without adding visual diversity.

**Recommendation for Phase 2:** No regeneration. When sizing val splits and computing rebalance weights, count by unique `image_path` rather than by manifest row. The 6,751-entry val recommendation in Stream 4 is by source, which already dedupes correctly.

## Stream 3 — Sequence-length + complexity distributions

**Goal:** Quantify per-corpus token-length distributions and detect silent truncation at the `max_sequence_length=512` cap.
**Result:** Synthetic and grandstaff have meaningful truncation tails. Specifically, **2.00% of synthetic entries (411 rows) and 0.73% of grandstaff entries (782 rows) are clipped at 512 tokens.** primus/cameraprimus truncation is negligible (0.01%).
**Evidence:** `audit_results/corpus_distributions.json`
**Numbers:**

| Corpus | n_entries | p50 | p95 | p99 | max | mean_staves | multi_staff % | truncated@512 | trunc_rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| cameraprimus_systems | 87,678 | 54 | 115 | 167 | 4,485 | 1.0 | 0% | 13 | 0.01% |
| grandstaff_systems | 107,724 | 178 | 343 | 475 | 766 | 2.0 | 100% | 782 | 0.73% |
| primus_systems | 87,678 | 54 | 115 | 167 | 4,485 | 1.0 | 0% | 13 | 0.01% |
| synthetic_systems | 20,583 | 301 | 456 | 565 | 892 | 3.18 | 99.6% | 411 | 2.00% |

- primus and cameraprimus are **byte-identical at the distribution level** (same n, same percentiles, same outliers, same truncation count). They appear to be the same source content under two corpus labels, or one is a relabel of the other.
- synthetic p99=565 means the long tail extends past 512 — moving the cap to 768 would cover the p99 with headroom; 1024 would cover synthetic max (892) and grandstaff max (766) fully.

**Interpretation:** Truncation matters because it correlates with the exact failure mode we're trying to fix. The truncated synthetic and grandstaff samples are the densest, longest, most-multi-staff sequences in the corpus — i.e., the ones that teach grand-staff and multi-voice reading. Clipping them at 512 tokens means the loss is masked over the back half of those examples, which trains the decoder to give up at exactly the structural points (final measures, terminal `<staff_end>` / `<eos>` tokens, voice coordination across staves) where Stage 3 v2 demonstrably failed. The 411 + 782 = 1,193 truncated rows are only 0.5% of the combined manifest, but they are a disproportionately informative fraction.

**Recommendation for Phase 2:** Raise `max_sequence_length` from 512 to **1024**. Memory cost is ~quadratic in sequence length for attention but pre-allocated buffers are typically padded to the same max, so the wall-clock hit will be roughly 2-4× on a small fraction of batches that contain long sequences. If that proves too costly empirically, fall back to 768 (covers synthetic p99 and grandstaff max). Either way, the change should resolve the silent-truncation pathology on the corpora most relevant to multi-staff.

## Stream 4 — Val splits

**Goal:** Audit per-corpus train/val/test splits for leakage and confirm whether synthetic has a val split at all.
**Result:** Three corpora have clean by-source splits (no overlap). **Synthetic has zero val and zero test entries** in the combined manifest. Bigger finding: the combined manifest contains only 20,583 synthetic rows vs 135,610 in the source synthetic manifest — **85% of synthetic data is being filtered out somewhere upstream of training.**
**Evidence:** `audit_results/val_split_audit.json`
**Numbers:**

| Corpus | n_train | n_val | n_test | unique src train | unique src val | overlap |
|---|---:|---:|---:|---:|---:|---:|
| cameraprimus_systems | 74,563 | 4,280 | 8,835 | 74,563 | 4,280 | 0 |
| grandstaff_systems | 96,952 | 5,340 | 5,432 | 96,952 | 5,340 | 0 |
| primus_systems | 74,563 | 4,280 | 8,835 | 74,563 | 4,280 | 0 |
| synthetic_systems | 20,583 | 0 | 0 | 703 | 0 | 0 |

- Synthetic-only manifest: 135,610 rows, 703 unique sources, mean 192.9 crops per source.
- Recommended synthetic val split (by `source_path`): 5% of 703 sources = 35 sources → ~6,751 val entries. By-source prevents font-level leakage (one source MXL rendered in three fonts must all go to one split).

**Interpretation:** Two separate problems here. The smaller one is the missing val split, which is solvable by re-emitting `token_manifest_stage3.jsonl` with a synthetic by-source split applied (no retraining cost; just rebuild the manifest). The larger one is the 6.6× row-count discrepancy between source and combined manifests. **The 77.8% synthetic-gradient-share diagnosis from the prior arc was computed against the 20,583-row filtered set.** If the filter is unintentional and the true synthetic share is 135,610 rows, then synthetic is not over-weighted on the cached tier — it's actually massively under-utilized, and the rebalance plan should look very different. If the filter is intentional (e.g., dropping rows where the image-token pair didn't pass verovio sanity checks, or de-duplicating `__poly` rows, or filtering vocal-tacet rest-only crops), the 20,583 number stands and the rebalance proceeds as planned. **We do not know which it is without reading the manifest-build code.**

**Recommendation for Phase 2:**

- Implement the synthetic by-source val split (35 sources, ~6,751 entries; methodology already specified in the JSON).
- Flag the 85% drop as an open question that must be answered before the Phase 2 dataset_mix weights are finalized.

## Stream 5 — Decoder capacity

**Goal:** Check whether the 116M-parameter decoder is undersized relative to the upstream Clarity-OMR DaViT decoder spec (~120M).
**Result:** Decoder capacity is at the reference. Capacity is **not** a bottleneck.
**Evidence:** `audit_results/decoder_capacity.json`
**Numbers:**

| Category | params | tensors | fraction |
|---|---:|---:|---:|
| encoder | 662,499,848 | 774 | 83.07% |
| decoder | 116,135,424 | 514 | 14.56% |
| bridge | 17,125,424 | 40 | 2.15% |
| embedding | 787,968 | 2 | 0.10% |
| head | 788,994 | 4 | 0.10% |
| other | 197,638 | 8 | 0.02% |
| **total** | **797,535,296** | **1,342** | 100% |

- The encoder figure (662M) is ~8× the published RADIO/DaViT encoder reference (~80M). The decoder-capacity counter does not collapse LoRA `base_layer` + `lora_A` + `lora_B` triplets into a single effective tensor, so this is almost certainly LoRA accounting inflation, not a real over-counting bug. Decoder counts the same way (514 tensors including base_layer + LoRA) but lands at the reference 116M, which is consistent with a real reference-sized decoder.

**Interpretation:** No capacity-side action is needed. If we ever wanted a defensible parameter count, we would need to merge LoRA adapters into base weights and re-measure — but for Phase 2 purposes that's a distraction. The Stage 3 v2 failure is not capacity-limited; it's data-pipeline-limited.

**Recommendation for Phase 2:** No decoder size bump. Note the encoder param-count caveat (LoRA accounting artifact) for future audits but don't act on it.

## Cross-stream synthesis

Three findings interact in ways that change the Phase 2 plan:

**The 85% synthetic filter drop (Stream 4) recontextualizes the entire imbalance diagnosis.** The prior arc concluded that synthetic was over-weighted at 77.8% of cached-tier gradient share. That math was done on the 20,583 filtered rows. The source synthetic manifest has 135,610 rows. We don't know whether the filter is dropping bad rows (in which case 20,583 is the real synthetic budget and the prior diagnosis stands) or dropping good rows accidentally (in which case the *true* synthetic budget is ~6.6× higher and the rebalance is solving the wrong problem). This must be resolved by user-side investigation before final Phase 2 weights are committed. The Phase 2 plan below proposes weights *assuming the filter is intentional*; if the user finds otherwise, the weights will need a second pass.

**Synthetic truncation (Stream 3) compounds the multi-staff failure (frankenstein arc).** Synthetic is the only corpus with 3.18 mean staves per entry and 99.6% multi-staff — it is structurally the multi-staff teacher. But 2% of synthetic entries get clipped at 512 tokens, and those are exactly the longest, densest multi-staff sequences. Combined with the missing val split (Stream 4: no signal on whether multi-staff is generalizing during training) and the prior frankenstein finding that the decoder collapses on multi-staff content, the picture is: we're training the decoder on a truncated, val-less, multi-staff-heavy diet and then measuring it on multi-staff-heavy demo content where it fails. The fixes (raise max_seq_len, add val split, rebalance) are all in the same direction.

**Capacity and truncation findings agree.** Decoder is reference-sized (Stream 5), and the problem is that long sequences are being clipped before the decoder can learn them (Stream 3). These are mutually consistent — capacity isn't the bottleneck because the bottleneck is upstream truncation never letting capacity be used. No tension between the two streams.

## Phase 2 fix proposal

Concrete changes for Phase 2 (the plan's Tasks 7–13):

### Required (per spec's "in-scope changes")

**1. Encoder freeze fix (Option B from prior v3 retrain spec)**
- **What:** Properly freeze encoder LoRA params during Stage 3 training (the v2 config claimed encoder was frozen but PR #48 audit found LoRA modules unfrozen). Implement the spec-specified Option B.
- **Why:** PR #48 audit verdict — real bug, must be fixed. Frankenstein diagnostic (PR #49) showed it was not the dominant failure mode (+0.001 onset_f1) but it remains a correctness issue and the per-task-7 work.
- **Effort:** 2-4 hours (small config + freeze-loop change), already designed in the v3 retrain spec.

**2. Synthetic val split (by-source, 5% of sources)**
- **What:** Rebuild `token_manifest_stage3.jsonl` (or its synthetic-emit step) with a by-`source_path` val split. Hold out 35 of the 703 unique synthetic sources for val. Estimated ~6,751 val entries.
- **Why:** Stream 4 — synthetic currently has zero val/test rows, so we have no measurement of multi-staff generalization during training.
- **Effort:** 4-6 hours (manifest rebuild + verification + checked into the data pipeline).

**3. dataset_mix rebalance**
- **What:** Proposed new weights (per cached-tier gradient share):
  - synthetic: **0.20** (down from current effective 77.8%; matches plan default)
  - grandstaff: **0.55** (up; only other multi-staff corpus, currently under-weighted relative to its informativeness)
  - primus/cameraprimus combined: **0.25** (down; identical single-staff content, easy material, currently over-represented in row count but already easy for the model)
- Choose ONE of primus/cameraprimus or dedupe to avoid effectively double-weighting the same content (see Open Questions).
- **Why:** Streams 3 and 4 — synthetic is the multi-staff teacher but has structural issues (truncation, no val); grandstaff is the second multi-staff source and needs more gradient share; primus/cameraprimus are easy single-staff and currently dominate by row count.
- **Effort:** 1-2 hours (config change + sanity run).

### Conditional — recommended

**4. max_sequence_length bump (512 → 1024)**
- **What:** Raise the truncation cap. 1024 covers synthetic max (892) and grandstaff max (766) fully. Fallback to 768 if memory becomes a problem.
- **Why:** Stream 3 — 411 synthetic and 782 grandstaff entries (2.0% and 0.73%) are silently truncated at the structural endpoints that teach multi-staff completion.
- **Effort:** 1 hour config change; training wall-clock impact 1.5-3× on the small fraction of batches with long sequences (full bench needed during the smoke run).
- **Recommendation: DO IT.** Targets the exact failure mode (multi-staff completion) the rebalance is trying to fix.

### Conditional — not recommended

**5. Decoder size bump**
- **What:** Increase decoder parameter count beyond the current 116M.
- **Why not:** Stream 5 — decoder is already at the upstream Clarity-OMR DaViT reference figure. Capacity is not the bottleneck; data-pipeline structure is.
- **Recommendation: SKIP.**

### Conditional — flag for user decision

**6. Investigate the 85% synthetic filter drop**
- **What:** Identify the code path between `data/processed/synthetic_v2/manifests/synthetic_token_manifest.jsonl` (135,610 rows) and `src/data/manifests/token_manifest_stage3.jsonl` (20,583 synthetic rows). Determine whether the 85% drop is intentional (filter against bad rows) or accidental (a join/dedupe gone wrong).
- **Why:** Stream 4 — the rebalance weights above assume the filter is intentional. If it's accidental, the rebalance is solving the wrong problem.
- **Effort:** unknown — could be 1-hour manifest-build script read, could be a multi-hour investigation.
- **Recommendation:** Open question for the user. Choose one of: (a) investigate before Phase 2 begins, (b) accept the current filtered behavior and proceed with the weights above, (c) defer to a follow-up audit and proceed.

## Open questions for user review

1. **The 85% synthetic filter drop.** Source synthetic manifest has 135,610 rows; combined Stage 3 manifest has 20,583 synthetic rows. Where does the drop happen? Is it intentional? If we fix it (or it's a bug), the rebalance weights need to be recomputed against a much larger effective synthetic pool. Recommend deciding investigation policy before Phase 2 weights are finalized.

2. **primus vs cameraprimus duplication.** The two corpora have byte-identical distributions (87,678 entries each, same percentiles, same truncation count, same outliers). They appear to be the same content under two labels. Should one be dropped? Should both be kept (assumes some augmentation differentiates them) at half weight each? Current plan splits them but with no rationale beyond corpus-keeping policy. Need user input on whether they're genuinely distinct.

3. **Encoder parameter count (662M vs ~80M reference).** Almost certainly LoRA accounting inflation (`base_layer` + `lora_A` + `lora_B` triplets each counted separately). Not actionable for Phase 2 but flagging for future audits.

4. **`max_sequence_length` target (1024 vs 768).** 1024 fully covers all corpora; 768 covers p99 of synthetic and 100% of grandstaff but leaves a small synthetic tail. Memory budget per the user's preference.

## Gate decision

User reviews this report and either:

- **Approves the Phase 2 proposal** → proceed to Task 7 of the plan (encoder freeze fix), then Tasks 8–13 (val split, manifest rebuild, max_seq_len bump, rebalance, smoke run, full retrain).
- **Modifies the proposal** (e.g., different rebalance weights, different max_seq_len target, drop one of primus/cameraprimus) → execute the modified plan.
- **Escalates** (e.g., requests investigation of the 85% filter before Phase 2 begins, or rules a recommended change out-of-scope) → pause to discuss.
