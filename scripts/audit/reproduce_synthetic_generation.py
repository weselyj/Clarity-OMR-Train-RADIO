"""Stream 2: Reproduce a small sample of synthetic_systems generation and
compare against the cached training data.

Picks N source MusicXML files from the openscore_lieder corpus that are
present in the cache, regenerates them into a scratch directory with the
exact same style ids the cache was built with, then SHA-compares the
resulting staff-crop PNGs against the cached versions on disk.

Confirms generator determinism (cache_sha == regen_sha) and detects
silent rendering drift (e.g. Verovio upgrade, font substitution).

Generator entry point (discovered at audit time):
    src/data/generate_synthetic.py :: run(...)
        - signature: run(project_root, data_root, input_manifest,
                        output_dir, style_ids, max_scores,
                        max_pages_per_score, seed, render, write_png,
                        dpis=..., roundtrip_validate=..., ...)
        - `input_manifest` is a jsonl with `mscx_path` or `musicxml_path`
          keys. We build a tiny one-off manifest pointing at only the
          chosen N sources so regeneration is fast.

Usage (on seder):
    venv-cu132\\Scripts\\python -m scripts.audit.reproduce_synthetic_generation \\
        --source-dir data\\openscore_lieder\\scores \\
        --cache-dir data\\processed\\synthetic_v2 \\
        --out audit_results\\synthetic_reproduce.json \\
        --n 5
"""
from __future__ import annotations
import argparse
import json
import shutil
import sys
import hashlib
import tempfile
import time
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

# Style ids the cache was built with; confirmed from manifest counts
# (leipzig-default: 45208, bravura-compact: 47429, gootville-wide: 42973).
CACHE_STYLES = ("leipzig-default", "bravura-compact", "gootville-wide")


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()[:16]


def _pick_sources(cache_manifest: Path, n: int):
    """Return list of (source_path, [cache entries for that source]).

    Picks first N distinct sources (sorted) that have at least one cached
    entry whose image file still exists on disk.
    """
    by_source: dict[str, list[dict]] = {}
    with cache_manifest.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            sp = e.get("source_path")
            if not sp:
                continue
            by_source.setdefault(sp, []).append(e)
    sources_with_cache = sorted(by_source.keys())
    chosen: list[tuple[str, list[dict]]] = []
    for sp in sources_with_cache:
        entries = by_source[sp]
        if any((_REPO / e["image_path"]).exists() for e in entries if e.get("image_path")):
            chosen.append((sp, entries))
        if len(chosen) >= n:
            break
    return chosen


def _write_tiny_manifest(sources: list[str], out_path: Path) -> None:
    """Write a one-off manifest the generator's load_manifest_sources can
    consume. The `musicxml_path` key is recognised; paths are relative to
    project root."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for sp in sources:
            f.write(json.dumps({"musicxml_path": sp}) + "\n")


def _regenerate(sources: list[str], scratch_dir: Path) -> dict:
    """Invoke the generator on the given sources, output to scratch_dir.

    Returns the generator summary dict (or an error record on failure).
    """
    from src.data.generate_synthetic import run as gen_run

    tiny_manifest = scratch_dir / "tiny_manifest.jsonl"
    _write_tiny_manifest(sources, tiny_manifest)

    output_dir = scratch_dir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    try:
        summary = gen_run(
            project_root=_REPO,
            data_root=_REPO / "data",
            input_manifest=tiny_manifest,
            output_dir=output_dir,
            style_ids=CACHE_STYLES,
            max_scores=None,
            max_pages_per_score=None,
            seed=1337,
            render=True,
            write_png=False,
            dpis=(300,),
            roundtrip_validate=False,
            show_verovio_warnings=False,
            workers=1,
            allow_fallback_labels=False,
        )
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "elapsed_sec": round(time.time() - t0, 2),
            "output_dir": str(output_dir),
        }

    return {
        "ok": True,
        "elapsed_sec": round(time.time() - t0, 2),
        "output_dir": str(output_dir),
        "summary_keys": sorted(summary.keys()) if isinstance(summary, dict) else None,
        "rendered_pages": summary.get("rendered_pages") if isinstance(summary, dict) else None,
        "token_entries_written": summary.get("token_entries_written") if isinstance(summary, dict) else None,
    }


def _compare_crops(entry: dict, scratch_out: Path) -> dict:
    """Hash cached crop and the regenerated counterpart with the same
    basename. Returns a structured record per entry."""
    cache_image = _REPO / entry["image_path"]
    basename = Path(entry["image_path"]).name
    style = entry.get("style_id", "?")
    regen_image = scratch_out / "staff_crops" / style / basename

    rec: dict = {
        "sample_id": entry["sample_id"],
        "style_id": style,
        "cache_image_path": entry["image_path"],
        "regen_image_relpath": str(regen_image.relative_to(_REPO)) if regen_image.is_relative_to(_REPO) else str(regen_image),
        "token_count": entry.get("token_count", len(entry.get("token_sequence", []) or [])),
    }
    if not cache_image.exists():
        rec["status"] = "cache_image_missing"
        return rec
    rec["cache_sha256_16"] = _file_hash(cache_image)
    rec["cache_size_bytes"] = cache_image.stat().st_size

    if not regen_image.exists():
        rec["status"] = "regen_image_missing"
        return rec
    rec["regen_sha256_16"] = _file_hash(regen_image)
    rec["regen_size_bytes"] = regen_image.stat().st_size
    rec["status"] = "match" if rec["cache_sha256_16"] == rec["regen_sha256_16"] else "mismatch"
    return rec


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", type=Path, required=True,
                   help="Root of source MusicXML files (data/openscore_lieder/scores). "
                        "Used only as a sanity check that the corpus is present.")
    p.add_argument("--cache-dir", type=Path, required=True,
                   help="Root of cached generated data (data/processed/synthetic_v2)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--scratch-dir", type=Path, default=None,
                   help="Where to materialise regenerated outputs. Defaults to a tempdir. "
                        "Kept after the run for inspection if explicitly set.")
    p.add_argument("--keep-scratch", action="store_true",
                   help="Do not delete the scratch dir at the end.")
    args = p.parse_args()

    if not args.source_dir.exists():
        print(f"WARN: --source-dir {args.source_dir} does not exist; continuing because the "
              f"generator resolves sources from the manifest which uses repo-relative paths.")

    cache_manifest = args.cache_dir / "manifests" / "synthetic_token_manifest.jsonl"
    if not cache_manifest.exists():
        print(f"ERROR: cache manifest not found at {cache_manifest}", file=sys.stderr)
        return 2

    chosen = _pick_sources(cache_manifest, args.n)
    print(f"Picked {len(chosen)} sources with existing cache:")
    for sp, entries in chosen:
        print(f"  {sp}  ({len(entries)} cached entries)")

    if not chosen:
        print("ERROR: no sources with cached images found.", file=sys.stderr)
        return 3

    # Scratch dir for regeneration
    if args.scratch_dir is not None:
        scratch_root = args.scratch_dir
        scratch_root.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        scratch_root = Path(tempfile.mkdtemp(prefix="audit_synth_repro_"))
        cleanup = not args.keep_scratch

    print(f"\nScratch dir: {scratch_root}")
    sources = [sp for sp, _ in chosen]

    try:
        regen_info = _regenerate(sources, scratch_root)
        print(f"Regeneration: ok={regen_info.get('ok')}, "
              f"elapsed={regen_info.get('elapsed_sec')}s, "
              f"rendered_pages={regen_info.get('rendered_pages')}")
        if not regen_info["ok"]:
            print(f"Regeneration failed:\n{regen_info.get('traceback', '')}", file=sys.stderr)

        per_source = []
        per_status: dict[str, int] = {}
        scratch_out = scratch_root / "out"
        for sp, entries in chosen:
            # Hash up to 3 cached crops per source so the JSON stays bounded.
            sampled_entries = entries[:3]
            per_entry = []
            for e in sampled_entries:
                if not e.get("image_path"):
                    per_entry.append({"sample_id": e.get("sample_id"), "status": "no_image_path_in_manifest"})
                    per_status["no_image_path_in_manifest"] = per_status.get("no_image_path_in_manifest", 0) + 1
                    continue
                rec = _compare_crops(e, scratch_out)
                per_status[rec["status"]] = per_status.get(rec["status"], 0) + 1
                per_entry.append(rec)
            per_source.append({
                "source_path": sp,
                "n_cached_entries": len(entries),
                "n_compared": len(sampled_entries),
                "per_entry": per_entry,
            })

        out_doc = {
            "experiment": "synthetic_reproduction",
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "generator_entry_point": {
                "module": "src.data.generate_synthetic",
                "callable": "run",
                "input_kind": "manifest jsonl (musicxml_path keys)",
            },
            "cache_styles_used_for_regen": list(CACHE_STYLES),
            "regen_info": regen_info,
            "status_counts": per_status,
            "per_source": per_source,
        }

        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
        print(f"\nWrote {args.out}")
        print(f"\nStatus counts: {per_status}")
    finally:
        if cleanup:
            try:
                shutil.rmtree(scratch_root)
                print(f"Cleaned up scratch dir {scratch_root}")
            except Exception as exc:
                print(f"WARN: failed to clean up {scratch_root}: {exc}")


if __name__ == "__main__":
    sys.exit(main() or 0)
