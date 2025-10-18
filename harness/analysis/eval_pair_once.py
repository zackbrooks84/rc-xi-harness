# harness/analysis/eval_pair_once.py
from __future__ import annotations
import argparse, json
from typing import Dict
from .eval_cli import evaluate_from_csv

def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(description="Combine RC+ξ stats (E1/E3) with Tlock from per-run JSONs.")
    ap.add_argument("--identity_csv", required=True, help="Per-turn CSV for Identity run")
    ap.add_argument("--null_csv", required=True, help="Per-turn CSV for Null run")
    ap.add_argument("--identity_json", required=True, help="Summary JSON from Identity run (for Tlock)")
    ap.add_argument("--null_json", required=True, help="Summary JSON from Null run (for Tlock)")
    ap.add_argument("--out_json", required=True, help="Path to write combined JSON")
    args = ap.parse_args()

    # Compute E1/E3 + stats from CSVs
    stats = evaluate_from_csv(args.identity_csv, args.null_csv)

    # Bring in Tlock (and any other per-run fields) from the per-run JSONs
    id_sum = _load_json(args.identity_json)
    nu_sum = _load_json(args.null_json)

    out = {
        **stats,
        "Tlock_identity": id_sum.get("Tlock", None),
        "Tlock_null": nu_sum.get("Tlock", None),
        "k": id_sum.get("k"),
        "m": id_sum.get("m"),
        "eps_xi": id_sum.get("eps_xi"),
        "eps_lvs": id_sum.get("eps_lvs"),
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    # also print a compact line for quick logs
    print(json.dumps(out, separators=(",", ":")))

if __name__ == "__main__":
    main()
