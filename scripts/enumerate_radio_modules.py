"""Enumerate all linear layers in RADIO and group by suspected DoRA-target categories.

Run with:
    venv\\Scripts\\python scripts\\enumerate_radio_modules.py > logs\\radio_modules.txt 2>&1

Output includes:
  - Raw counts per category (qkv, attn_proj, fc1, fc2, other)
  - Up to 5 example names per category
  - Full 'other' bucket listing (to catch any patterns the heuristics miss)
  - A suggested DoRA target_modules list in the format consumed by the pipeline
"""
import re
import sys

import torch


def main():
    print("Loading C-RADIOv4-H...")
    model = torch.hub.load(
        "NVlabs/RADIO",
        "radio_model",
        version="c-radio_v4-h",
        progress=True,
        skip_validation=True,
        trust_repo=True,  # required for non-interactive SSH
    )

    print("\nAll Linear layers in RADIO model (full hierarchy):")

    # -------------------------------------------------------------------
    # Category buckets -- ordered from most-specific to least-specific
    # so each layer lands in exactly one bucket.
    # -------------------------------------------------------------------
    categories = [
        # (bucket_key, compiled_regex, human description)
        ("qkv_combined",  re.compile(r"attn\.qkv$"),                   "combined QKV projection"),
        ("q_proj",        re.compile(r"(?:^|\.)(q|q_proj)$"),           "query projection"),
        ("k_proj",        re.compile(r"(?:^|\.)(k|k_proj)$"),           "key projection"),
        ("v_proj",        re.compile(r"(?:^|\.)(v|v_proj)$"),           "value projection"),
        ("attn_proj",     re.compile(r"attn\.proj$|(?:^|\.)out_proj$"), "attention output projection"),
        ("fc1",           re.compile(r"mlp\.fc1$|(?:^|\.)(gate_proj|up_proj|fc1)$"), "MLP first/gate projection"),
        ("fc2",           re.compile(r"mlp\.fc2$|(?:^|\.)(down_proj|fc2)$"),         "MLP second/down projection"),
        # Catch linear projections in patch embedding / head / adapters
        ("patch_embed",   re.compile(r"patch_embed|patch_generator"),   "patch embedding"),
        ("head",          re.compile(r"(?:^|\.)head(?:\.|$)"),           "classification/output head"),
        ("norm_proj",     re.compile(r"(?:^|\.)(norm|layer_norm)"),      "normalisation layer (Linear)"),
        ("other",         re.compile(r"."),                              "uncategorised"),
    ]

    counts  = {k: 0 for k, _, _ in categories}
    examples = {k: [] for k, _, _ in categories}
    all_names = {k: [] for k, _, _ in categories}

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        short = name.split(".")[-1]  # bare attribute name
        placed = False
        for bucket, pattern, _ in categories:
            # match against both full path and bare attribute name
            if pattern.search(name) or pattern.search(short):
                counts[bucket] += 1
                all_names[bucket].append(name)
                if len(examples[bucket]) < 5:
                    examples[bucket].append(name)
                placed = True
                break
        if not placed:
            # Fallback (should not happen — 'other' regex matches everything)
            counts["other"] += 1
            all_names["other"].append(name)
            if len(examples["other"]) < 5:
                examples["other"].append(name)

    # -------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------
    print("\nCounts by category:")
    total = 0
    for bucket, _, desc in categories:
        c = counts[bucket]
        total += c
        if c > 0:
            print(f"  {bucket:18s} {c:4d}  ({desc})")
            print(f"  {'':18s}       examples: {examples[bucket]}")
    print(f"  {'TOTAL':18s} {total:4d}")

    # -------------------------------------------------------------------
    # Full listing of 'other' bucket so we can extend patterns if needed
    # -------------------------------------------------------------------
    if all_names["other"]:
        print("\nFull 'other' bucket (uncategorised linears):")
        for n in all_names["other"]:
            print(f"  {n}")
    else:
        print("\n'other' bucket is empty -- all linears categorised.")

    # -------------------------------------------------------------------
    # Derive exact leaf names used in RADIO (for YAML target_modules)
    # -------------------------------------------------------------------
    print("\nDistinct leaf-attribute names per DoRA-relevant bucket:")
    dora_buckets = ["qkv_combined", "q_proj", "k_proj", "v_proj", "attn_proj", "fc1", "fc2"]
    leaf_names: dict[str, set[str]] = {b: set() for b in dora_buckets}
    for bucket in dora_buckets:
        for fullname in all_names[bucket]:
            leaf_names[bucket].add(fullname.split(".")[-1])
        if leaf_names[bucket]:
            print(f"  {bucket:18s}: {sorted(leaf_names[bucket])}")

    # -------------------------------------------------------------------
    # Suggested DoRA target list
    # -------------------------------------------------------------------
    print("\nSuggested DoRA target_modules (exact leaf names, no regex):")
    seen: set[str] = set()
    suggested: list[str] = []
    for bucket in dora_buckets:
        for name in sorted(leaf_names[bucket]):
            if name not in seen:
                seen.add(name)
                suggested.append(name)
    for s in suggested:
        print(f'  - "{s}"')

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
