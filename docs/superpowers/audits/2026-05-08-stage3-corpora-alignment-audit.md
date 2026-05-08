# Stage 3 Corpora Token Alignment Audit (2026-05-08)

Companion to the synthetic_systems alignment fix plan
(`docs/superpowers/plans/2026-05-08-radio-stage3-token-alignment-fix.md`). Verifies
whether the contract bug found in `synthetic_v2` exists in any of the other
three Stage 3 corpora.

## Methodology

The contract bug under investigation: `generate_synthetic.py:2400` writes per-staff
manifest entries with `staff_index = enumerate(post-filter survivors)`. When
mid-page staff crops are dropped by the ink-quality filter, surviving indices
are renumbered, but the consumer (`yolo_aligned_systems.py:54`) computes expected
indices from physical-position cumsum. The two disagree silently on affected
pages.

Each corpus was checked for this class of bug **using the audit method appropriate
to its manifest shape** — they're not all structured the same way.

## synthetic_systems_v1

**Status: BROKEN.** Verified before this audit.

Canary: `Abbott___Jane_Bingham_____Just_for_Today__lc6583477.mxl__bravura-compact__p001`.
- 9 staves rendered (per `data\processed\synthetic_v2\labels\bravura-compact\<page>.txt`)
- 7 entries in per-staff manifest (`data\processed\synthetic_v2\manifests\synthetic_token_manifest.jsonl`)
- 2 systems present in `synthetic_systems_v1` (sys02 dropped via `dropped_token_miss`)
- sys00 clefs `[clef-G2, clef-G2, clef-F4]` ✓ (canonical)
- **sys01 clefs `[clef-G2, clef-F4, clef-G2]`** ✗ (canonical: `[clef-G2, clef-G2, clef-F4]`)
- sys01's image visually shows physical staves [3, 4, 5]; its tokens describe physical staves [3, 5, 6]

Root cause: post-filter `enumerate` renumbering in `generate_synthetic.py:2400-2437`,
inconsistent with the consumer's physical-position cumsum lookup.

**Action:** Phase 2 of the alignment-fix plan.

## primus_systems

**Status: CLEAN by construction.**

The expected audit (`scripts/audit_per_staff_alignment.py`) was not applicable —
the manifest doesn't carry `page_id` or `staff_index` fields, because primus
is a single-staff monophonic corpus where each entry IS its own system.
`scripts/retokenize_with_staff_markers.py` is a 1:1 entry transform that:
- copies each input entry,
- prepends `<staff_idx_0>` to the token sequence,
- forces `staves_in_system = 1`.

There is no filter, no drop, and no multi-staff aggregation. The bug class
under investigation cannot manifest.

Direct invariant check (87,678 entries):
- `total = 87,678`
- `not_single_staff = 0`
- `missing_marker_or_malformed_seq = 0`
- `unique_sample_ids = 87,678`
- `duplicate_sample_ids = 0`
- Fields: `dataset, image_path, sample_id, source_format, source_path, split, staves_in_system, token_count, token_sequence`

Every entry passes the structural shape `<bos> <staff_start> <staff_idx_0> ... <staff_end> <eos>`.

**Action:** None.

## cameraprimus_systems

**Status: CLEAN by construction.**

Same pipeline as primus_systems (the same retokenizer with `--source-dataset cameraprimus`).
Same invariants verified (87,678 entries):
- `total = 87,678`
- `not_single_staff = 0`
- `missing_marker_or_malformed_seq = 0`
- `unique_sample_ids = 87,678`
- `duplicate_sample_ids = 0`
- Same field set as primus_systems.

**Action:** None.

## grandstaff_systems

**Status: CLEAN.**

This corpus is built directly at the system level by Subproject 2's pipeline
(no per-staff intermediate manifest). Each manifest entry is one complete
grand-staff system with both staves' tokens already concatenated, using
`<staff_idx_N>` markers in source-physical order.

The relevant invariants — independent of the synthetic_v2 bug class — are:
- `staves_in_system` matches the count of `<staff_idx_N>` markers in `token_sequence`
- markers appear in order `0, 1, ..., K-1`
- each `<staff_idx_N>` marker immediately follows a `<staff_start>` token

Direct invariant check (107,724 entries):
- `total = 107,724`
- `staves_in_system_distribution = {2: 107,724}` (all grand staffs are 2-staff piano, as expected)
- `count_mismatch = 0`
- `bad_marker_order = 0`
- `misplaced_marker = 0`
- `structural_malformed = 0`
- `unique_sample_ids = 107,724`
- `duplicate_sample_ids = 0`

Because grandstaff_systems is built at the system level (no per-staff manifest
that the systems builder would later look up by index), the contract bug from
`generate_synthetic.py` cannot manifest in this pipeline.

**Action:** None.

## Synthesis

| Corpus | Pipeline | Audit Method | Status | Action |
|---|---|---|---|---|
| synthetic_systems_v1 | Plan A on top of synthetic_v2 per-staff manifest | direct canary check (Abbott p001 clef pattern) | BROKEN | Phase 2 of alignment-fix plan |
| primus_systems | retokenize_with_staff_markers.py from primus | structural invariants | CLEAN by construction | None |
| cameraprimus_systems | retokenize_with_staff_markers.py from cameraprimus | structural invariants | CLEAN by construction | None |
| grandstaff_systems | Subproject 2 system-level pipeline | system-token marker invariants | CLEAN | None |

**Phase 3 decision: SKIP.** Only `synthetic_systems_v1` requires fixes. Phase 2
of the plan is sufficient — no follow-up plan needed for the other corpora.

The contract bug class is specific to pipelines that:
1. Generate a per-staff intermediate manifest, AND
2. Apply a post-emission filter that renumbers surviving entries, AND
3. Are consumed by a builder that looks up entries by physical-position index.

Only synthetic_v2 → Plan A satisfies all three conditions.
