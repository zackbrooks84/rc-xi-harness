# harness/run_harness.py
from __future__ import annotations
import argparse, json
from typing import Tuple, List
import numpy as np

from harness.metrics import (
    xi_series, k_window_lvs, anchor_vector,
    anchor_persistence, ewma, lock_detect
)
from harness.io.schema import write_rows

def run_one(E: np.ndarray,
            run_type: str,
            provider_name: str,
            k: int,
            m: int,
            eps_xi: float,
            eps_lvs: float) -> Tuple[List[dict], dict]:
    """
    Compute per-turn metrics + summary endpoints (E1 median ξ last-10, Tlock).
    E shape: (T, d)
    """
    # core series
    xi = xi_series(E)              # shape (T-1,)
    lvs = k_window_lvs(E, k=k)     # shape (T,)
    a = anchor_vector(E, n_seed=3) # shape (d,)
    Pt = anchor_persistence(E, a)  # shape (T,)
    xi_s = ewma(xi, alpha=0.5)     # smoothed ξ

    # E1: median ξ over final 10 turns (or all if <10)
    if xi.size >= 10:
        e1 = float(np.median(xi[-10:]))
    else:
        e1 = float(np.median(xi)) if xi.size else float("nan")

    # E2: lock time (first t that meets criteria)
    Tlock = None
    for t in range(xi.size):
        # check using all points up to t, and LVS at t+1 (since lvs is length T)
        if (t + 1) >= m:
            if lock_detect(xi[:t+1], lvs[t+1], eps_xi=eps_xi, eps_lvs=eps_lvs, m=m):
                Tlock = int(t + 1)  # lock time in turn index space
                break

    # per-turn rows
    rows: List[dict] = []
    T = E.shape[0]
    for t in range(T):
        rows.append({
            "t": t,
            "xi": float(xi[t-1]) if t > 0 and (t-1) < xi.size else "",
            "lvs": float(lvs[t]),
            "Pt": float(Pt[t]),
            "ewma_xi": float(xi_s[t-1]) if t > 0 and (t-1) < xi_s.size else "",
            "run_type": run_type,
            "provider": provider_name
        })

    summary = {
        "E1_median_xi_last10": e1,
        "Tlock": Tlock,
        "k": k, "m": m, "eps_xi": eps_xi, "eps_lvs": eps_lvs,
        "provider": provider_name, "run_type": run_type
    }
    return rows, summary

def main():
    ap = argparse.ArgumentParser(description="RC+ξ public harness (per-run)")
    ap.add_argument("--embed_npy", required=True, help="Path to (T,d) NumPy embeddings file")
    ap.add_argument("--run_type", required=True, choices=["identity", "null", "shuffled"])
    ap.add_argument("--provider", default="dummy")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--eps_xi", type=float, default=0.02)
    ap.add_argument("--eps_lvs", type=float, default=0.015)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    E = np.load(args.embed_npy)  # expected shape (T, d)
    if E.ndim != 2:
        raise ValueError("Embeddings array must be 2D (T, d).")

    rows, summary = run_one(
        E, args.run_type, args.provider,
        k=args.k, m=args.m,
        eps_xi=args.eps_xi, eps_lvs=args.eps_lvs
    )

    write_rows(args.out_csv, rows)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
