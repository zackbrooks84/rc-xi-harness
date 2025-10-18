# harness/run_from_transcript.py
from __future__ import annotations
import argparse, json
from typing import List, Tuple
import numpy as np

from harness.io.transcript import load_txt, load_csv
from harness.embeddings.random_provider import RandomHashProvider
from harness.run_harness import run_one

def _load_transcript(path: str, fmt: str, csv_col: str) -> List[str]:
    fmt = (fmt or "").lower()
    if fmt == "txt":
        return load_txt(path)
    elif fmt == "csv":
        return load_csv(path, column=csv_col)
    else:
        # try to infer from extension
        if path.lower().endswith(".txt"):
            return load_txt(path)
        elif path.lower().endswith(".csv"):
            return load_csv(path, column=csv_col)
        raise ValueError("Specify --format txt|csv or use .txt/.csv extension.")

def main():
    ap = argparse.ArgumentParser(description="Run RC+ξ harness from a transcript using RandomHash embeddings.")
    ap.add_argument("--input", required=True, help="Path to transcript (.txt lines or .csv column)")
    ap.add_argument("--format", choices=["txt","csv"], default=None, help="Override file format (optional)")
    ap.add_argument("--csv_col", default="reply", help="CSV column name if --format csv (default: reply)")
    ap.add_argument("--run_type", required=True, choices=["identity","null","shuffled"])
    ap.add_argument("--dim", type=int, default=384, help="Embedding dimension (default 384)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--eps_xi", type=float, default=0.02)
    ap.add_argument("--eps_lvs", type=float, default=0.015)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    texts = _load_transcript(args.input, args.format, args.csv_col)
    if not texts:
        raise ValueError("Transcript is empty after cleaning.")

    provider = RandomHashProvider(dim=args.dim)
    E = provider.embed(texts)  # (T, d)

    rows, summary = run_one(
        E,
        run_type=args.run_type,
        provider_name="random-hash",
        k=args.k, m=args.m,
        eps_xi=args.eps_xi, eps_lvs=args.eps_lvs
    )

    from harness.io.schema import write_rows
    write_rows(args.out_csv, rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
