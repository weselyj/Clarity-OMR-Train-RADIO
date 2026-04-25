"""Render each piece in the Lieder eval split to PDF for OMR input.

Requires MuseScore 4 (or 3) installed. Run from the repo root:
    venv\\Scripts\\python scripts\\render_lieder_eval.py
"""
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, ".")
from eval.lieder_split import get_eval_pieces, split_hash

# MuseScore CLI candidates in order of preference
MSCORE_CANDIDATES = [
    r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
    r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
    r"C:\Program Files (x86)\MuseScore 4\bin\MuseScore4.exe",
]


def find_mscore() -> str:
    for c in MSCORE_CANDIDATES:
        if Path(c).exists():
            return c
    raise RuntimeError(
        f"No MuseScore executable found. Tried: {MSCORE_CANDIDATES}. "
        "Install via 'winget install MuseScore.MuseScore' and retry."
    )


def main() -> None:
    mscore = find_mscore()
    print(f"Split hash: {split_hash()}")
    print(f"Using MuseScore: {mscore}")
    out_dir = Path("data/openscore_lieder/eval_pdfs")
    out_dir.mkdir(parents=True, exist_ok=True)

    rendered = skipped = failed = 0
    failed_pieces: list[str] = []
    pieces = get_eval_pieces()
    n_total = len(pieces)
    print(f"Rendering {n_total} pieces...")

    for i, piece in enumerate(pieces, 1):
        pdf_path = out_dir / f"{piece.stem}.pdf"
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            skipped += 1
            continue
        try:
            subprocess.run(
                [mscore, "-o", str(pdf_path.resolve()), str(piece.resolve())],
                check=True,
                capture_output=True,
                timeout=120,
            )
            rendered += 1
            if rendered % 10 == 0 or i == n_total:
                print(f"  [{i}/{n_total}] last: {piece.stem}")
        except subprocess.CalledProcessError as e:
            stderr_snippet = e.stderr.decode(errors="replace")[:300]
            print(f"  FAIL {piece.name}: {stderr_snippet}")
            failed_pieces.append(piece.name)
            failed += 1
        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT {piece.name}")
            failed_pieces.append(piece.name + " (timeout)")
            failed += 1

    print(f"\nrendered: {rendered}, skipped (already done): {skipped}, failed: {failed}")
    if failed_pieces:
        print("Failed pieces:")
        for p in failed_pieces:
            print(f"  {p}")
    if failed > 0 and n_total > 0 and failed / n_total > 0.05:
        print(f"WARNING: failure rate {failed/n_total:.1%} exceeds 5% threshold")


if __name__ == "__main__":
    main()
