# harness/run_all_from_transcript.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

from harness.analysis.eval_cli import evaluate_from_csv
from harness.run_pair_from_transcript import run_pair_from_transcript


def _load_json(path: str) -> Dict[str, object]:
    """Read ``path`` as UTF-8 JSON and return the parsed object."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    """End-to-end pipeline: transcript ➜ per-run artifacts ➜ combined JSON."""

    ap = argparse.ArgumentParser(
        description=(
            "All-in-one RC+ξ pipeline: transcript ➜ Identity/Null/Shuffled CSV+JSON ➜ combined results JSON"
        )
    )
    ap.add_argument("--input", required=True, help="Transcript path (.txt or .csv)")
    ap.add_argument("--format", choices=["txt", "csv"], default=None, help="Override (optional)")
    ap.add_argument("--csv_col", default="reply", help="CSV column if --format=csv (default: reply)")
    ap.add_argument("--out_dir", default="out", help="Directory to write intermediate outputs (default: out)")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--eps_xi", type=float, default=0.02)
    ap.add_argument("--eps_lvs", type=float, default=0.015)
    ap.add_argument("--shuffle_seed", type=int, default=42, help="Seed for shuffled-control permutation")
    ap.add_argument(
        "--out_results",
        default=None,
        help="Path to write combined results JSON (default: out/<stem>.results.json)",
    )
    args = ap.parse_args()

    # 1) Produce Identity, Null, and Shuffled per-run outputs (CSV + JSON)
    paths = run_pair_from_transcript(
        input_path=args.input,
        fmt=args.format,
        csv_col=args.csv_col,
        out_dir=args.out_dir,
        dim=args.dim,
        k=args.k,
        m=args.m,
        eps_xi=args.eps_xi,
        eps_lvs=args.eps_lvs,
        shuffle_seed=args.shuffle_seed,
    )

    # 2) Compute prereg endpoints + control checks from the CSVs
    stats = evaluate_from_csv(
        identity_csv=paths["identity_csv"],
        null_csv=paths["null_csv"],
        shuffled_csv=paths["shuffled_csv"],
        eps_xi=args.eps_xi,
        eps_lvs=args.eps_lvs,
        m=args.m,
    )

    # 3) Merge with per-run JSON summaries for reporting
    id_sum = _load_json(paths["identity_json"])
    nu_sum = _load_json(paths["null_json"])
    sh_sum = _load_json(paths["shuffled_json"])

    combined = {
        **stats,
        "Tlock_identity": id_sum.get("Tlock"),
        "Tlock_null": nu_sum.get("Tlock"),
        "Tlock_shuffled": sh_sum.get("Tlock"),
        "k": id_sum.get("k"),
        "m": id_sum.get("m"),
        "eps_xi": id_sum.get("eps_xi"),
        "eps_lvs": id_sum.get("eps_lvs"),
        "identity_csv": paths["identity_csv"],
        "null_csv": paths["null_csv"],
        "shuffled_csv": paths["shuffled_csv"],
        "identity_json": paths["identity_json"],
        "null_json": paths["null_json"],
        "shuffled_json": paths["shuffled_json"],
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    default_results = out_dir / f"{Path(args.input).stem}.results.json"
    out_results = Path(args.out_results) if args.out_results else default_results
    out_results.write_text(json.dumps(combined, indent=2), encoding="utf-8")

    print(json.dumps(combined, separators=(",", ":")))


if __name__ == "__main__":
    main()
