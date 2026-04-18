#!/usr/bin/env python3
"""
anchor_swap_check.py — Anchor-swap falsifier.

Compares anchor-persistence Pt on a *target* transcript using:
  (a) the target's own seed anchor   (real)
  (b) a donor transcript's seed anchor (swapped)

If the identity signal is real, the target's own anchor should produce a
noticeably better Pt slope (higher / less-decaying) than a donor anchor.
If swapping anchors makes little difference, the "anchor" isn't actually
doing structural work — it's just embedding noise.

Outputs:
  pt_real          — mean Pt under real anchor (late window)
  pt_swap          — mean Pt under donor anchor (late window)
  pt_swap_gap      — pt_real - pt_swap         (positive = real anchor wins)
  slope_real       — linear slope of Pt over turns (real)
  slope_swap       — linear slope of Pt over turns (swap)
  slope_gap        — slope_real - slope_swap    (positive = real anchor wins)

Usage:
    python -m harness.anchor_swap_check \\
        --target data/transcript_claude-sonnet-4-6-anchor_20260414.txt \\
        --donor  data/transcript_claude-sonnet-4-6-anchor-control_20260415.txt \\
        --out    xi_results/anchor_swap_check.json \\
        --provider sentence-transformer \\
        --sentence_model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import List

import numpy as np

from harness.embeddings.factory import create_provider
from harness.io.transcript import load_csv, load_txt
from harness.metrics import anchor_persistence, anchor_vector
from harness.protocols import identity_texts
from harness.protocols.anchor_swap import swapped_anchor_persistence


def _load(path: str, fmt: str | None, csv_col: str) -> List[str]:
    if fmt is None:
        p = path.lower()
        if p.endswith(".txt"):
            return load_txt(path)
        if p.endswith(".csv"):
            return load_csv(path, column=csv_col)
        raise ValueError("Unknown format. Use --format txt|csv or .txt/.csv extension.")
    if fmt == "txt":
        return load_txt(path)
    if fmt == "csv":
        return load_csv(path, column=csv_col)
    raise ValueError("Unsupported --format (use txt or csv).")


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def anchor_swap_check(
    target_path: str,
    donor_path: str,
    fmt: str | None = None,
    csv_col: str = "reply",
    provider: str = "sentence-transformer",
    sentence_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    sentence_device: str | None = None,
    sentence_normalize: bool = True,
    sentence_batch_size: int | None = None,
    dim: int = 384,
    n_seed: int = 3,
    late_frac: float = 0.5,
) -> dict:
    target_texts = identity_texts(_load(target_path, fmt, csv_col))
    donor_texts = identity_texts(_load(donor_path, fmt, csv_col))
    if not target_texts or not donor_texts:
        raise ValueError("target or donor transcript is empty.")

    prov = create_provider(
        provider,
        {
            "dim": dim,
            "model_name": sentence_model,
            "device": sentence_device,
            "batch_size": sentence_batch_size,
            "normalize": sentence_normalize,
        },
    )
    E_target = prov.embed(target_texts)
    E_donor = prov.embed(donor_texts)
    if E_target.ndim != 2 or E_donor.ndim != 2:
        raise ValueError("Embeddings must be 2D (T, d).")

    a_real = anchor_vector(E_target, n_seed=n_seed)
    pt_real = anchor_persistence(E_target, a_real)
    pt_swap = swapped_anchor_persistence(E_target, E_donor, n_seed=n_seed)

    T = len(pt_real)
    late_start = max(n_seed, int(T * (1.0 - late_frac)))
    pt_real_late = pt_real[late_start:]
    pt_swap_late = pt_swap[late_start:]

    slope_real = _slope(pt_real)
    slope_swap = _slope(pt_swap)

    result = {
        "target": target_path,
        "donor": donor_path,
        "provider": provider,
        "sentence_model": sentence_model if provider == "sentence-transformer" else None,
        "n_seed": n_seed,
        "turns": int(T),
        "late_window_start": late_start,
        "pt_real_mean_late": float(np.mean(pt_real_late)),
        "pt_swap_mean_late": float(np.mean(pt_swap_late)),
        "pt_swap_gap": float(np.mean(pt_real_late) - np.mean(pt_swap_late)),
        "pt_real_median_late": float(statistics.median(pt_real_late.tolist())),
        "pt_swap_median_late": float(statistics.median(pt_swap_late.tolist())),
        "slope_real": slope_real,
        "slope_swap": slope_swap,
        "slope_gap": slope_real - slope_swap,
        "pt_real_series": [float(v) for v in pt_real.tolist()],
        "pt_swap_series": [float(v) for v in pt_swap.tolist()],
        "verdict": (
            "real_anchor_beats_donor"
            if (np.mean(pt_real_late) - np.mean(pt_swap_late)) > 0.0
            else "swap_matches_or_beats_real"
        ),
    }
    return result


def _print_report(r: dict) -> None:
    print("=" * 72)
    print("Anchor-swap falsifier")
    print("=" * 72)
    print(f"target : {r['target']}")
    print(f"donor  : {r['donor']}")
    print(f"provider: {r['provider']}  model: {r['sentence_model']}")
    print(f"turns={r['turns']}  n_seed={r['n_seed']}  late_window_start={r['late_window_start']}")
    print("")
    print(f"Pt mean (late)     real = {r['pt_real_mean_late']:+.4f}   swap = {r['pt_swap_mean_late']:+.4f}   gap = {r['pt_swap_gap']:+.4f}")
    print(f"Pt median (late)   real = {r['pt_real_median_late']:+.4f}   swap = {r['pt_swap_median_late']:+.4f}")
    print(f"Pt slope (all)     real = {r['slope_real']:+.5f}   swap = {r['slope_swap']:+.5f}   gap = {r['slope_gap']:+.5f}")
    print("")
    print(f"verdict: {r['verdict']}")
    print("  (positive pt_swap_gap and/or slope_gap => target's own anchor")
    print("   carries structure a donor anchor does not.)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor-swap falsifier for RC+ξ harness.")
    ap.add_argument("--target", required=True, help="Target transcript path (.txt or .csv)")
    ap.add_argument("--donor", required=True, help="Donor transcript path (anchor source for swap)")
    ap.add_argument("--format", choices=["txt", "csv"], default=None)
    ap.add_argument("--csv_col", default="reply")
    ap.add_argument("--out", default=None, help="Optional JSON path to write results")
    ap.add_argument("--provider", choices=["random-hash", "sentence-transformer"],
                    default="sentence-transformer")
    ap.add_argument("--sentence_model",
                    default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--sentence_device", default=None)
    ap.add_argument("--sentence_batch_size", type=int, default=None)
    ap.add_argument("--sentence_no_normalize", action="store_true")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--n_seed", type=int, default=3)
    ap.add_argument("--late_frac", type=float, default=0.5,
                    help="Fraction of trailing turns to average Pt over (default 0.5)")
    args = ap.parse_args()

    r = anchor_swap_check(
        target_path=args.target,
        donor_path=args.donor,
        fmt=args.format,
        csv_col=args.csv_col,
        provider=args.provider,
        sentence_model=args.sentence_model,
        sentence_device=args.sentence_device,
        sentence_normalize=not args.sentence_no_normalize,
        sentence_batch_size=args.sentence_batch_size,
        dim=args.dim,
        n_seed=args.n_seed,
        late_frac=args.late_frac,
    )
    _print_report(r)
    if args.out:
        outp = Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(r, indent=2), encoding="utf-8")
        print(f"\nWrote {outp}")


if __name__ == "__main__":
    main()
