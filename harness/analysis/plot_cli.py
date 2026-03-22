# harness/analysis/plot_cli.py
"""CLI for generating PNG plots from existing per-turn CSV artifacts.

Usage (single-condition):
    python -m harness.analysis.plot_cli \\
      --identity_csv out/sample.identity.csv \\
      --out_dir out/plots/

Usage (multi-condition overlay):
    python -m harness.analysis.plot_cli \\
      --identity_csv out/sample.identity.csv \\
      --null_csv out/sample.null.csv \\
      --shuffled_csv out/sample.shuffled.csv \\
      --identity_json out/sample.identity.json \\
      --out_dir out/plots/

Outputs written to --out_dir:
  pair.png          — multi-condition overlay (identity + whatever is provided)
  identity.png      — single-run chart for identity run
  null.png          — single-run chart for null run (if --null_csv given)
  shuffled.png      — single-run chart for shuffled run (if --shuffled_csv given)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Optional

from .plots import plot_pair, plot_xi_series


def _read_csv(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _read_tlock(json_path: Optional[str]) -> Optional[int]:
    if not json_path:
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tlock = data.get("Tlock")
    return int(tlock) if tlock is not None else None


def main():
    ap = argparse.ArgumentParser(
        description="Generate PNG plots from RC+ξ per-turn CSV artifacts."
    )
    ap.add_argument("--identity_csv", required=True, help="Per-turn CSV for identity run")
    ap.add_argument("--null_csv", default=None, help="Per-turn CSV for null run (optional)")
    ap.add_argument("--shuffled_csv", default=None, help="Per-turn CSV for shuffled run (optional)")
    ap.add_argument(
        "--identity_json",
        default=None,
        help="Summary JSON for identity run — used to read Tlock (optional)",
    )
    ap.add_argument("--out_dir", required=True, help="Directory to write PNG files")
    ap.add_argument("--title", default=None, help="Optional figure title override")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    identity_rows = _read_csv(args.identity_csv)
    null_rows = _read_csv(args.null_csv) if args.null_csv else None
    shuffled_rows = _read_csv(args.shuffled_csv) if args.shuffled_csv else None
    tlock = _read_tlock(args.identity_json)

    # Multi-condition overlay
    pair_path = str(out / "pair.png")
    plot_pair(
        identity_rows=identity_rows,
        null_rows=null_rows,
        shuffled_rows=shuffled_rows,
        out_path=pair_path,
        title=args.title,
        tlock_identity=tlock,
    )
    print(f"Written: {pair_path}")

    # Individual single-run charts
    for label, rows, json_path in [
        ("identity", identity_rows, args.identity_json),
        ("null", null_rows, None),
        ("shuffled", shuffled_rows, None),
    ]:
        if rows is None:
            continue
        run_tlock = _read_tlock(json_path) if json_path else None
        single_path = str(out / f"{label}.png")
        plot_xi_series(rows=rows, out_path=single_path, tlock=run_tlock)
        print(f"Written: {single_path}")


if __name__ == "__main__":
    main()
