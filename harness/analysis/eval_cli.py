# harness/analysis/eval_cli.py
from __future__ import annotations

import argparse
import csv
import json
from typing import Dict, Tuple, Union

import numpy as np

from .endpoint_eval import evaluate_identity_vs_null, evaluate_identity_vs_shuffled


def _load_series(csv_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse per-turn metrics from ``csv_path``.

    Returns ``(xi, Pt, lvs)`` as ``float`` arrays while safely ignoring blank
    cells and malformed values.
    """

    xi_vals = []
    Pt_vals = []
    lvs_vals = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xi_str = (row.get("xi") or "").strip()
            if xi_str:
                try:
                    xi_vals.append(float(xi_str))
                except ValueError:
                    pass
            Pt_str = (row.get("Pt") or "").strip()
            if Pt_str:
                try:
                    Pt_vals.append(float(Pt_str))
                except ValueError:
                    pass
            lvs_str = (row.get("lvs") or "").strip()
            if lvs_str:
                try:
                    lvs_vals.append(float(lvs_str))
                except ValueError:
                    pass
    return (
        np.asarray(xi_vals, dtype=float),
        np.asarray(Pt_vals, dtype=float),
        np.asarray(lvs_vals, dtype=float),
    )


def evaluate_from_csv(
    identity_csv: str,
    null_csv: str,
    *,
    shuffled_csv: str | None = None,
    eps_xi: float = 0.02,
    eps_lvs: float = 0.015,
    m: int = 5,
) -> Dict[str, Union[float, bool]]:
    """Evaluate preregistered endpoints from CSV artifacts.

    If ``shuffled_csv`` is provided the return dictionary includes the control
    checks that assert the shuffled run destroys the lock signature.

    Raises
    ------
    ValueError
        If shuffled evaluation is requested but the CSVs lack LVS data.
    """

    xi_id, Pt_id, lvs_id = _load_series(identity_csv)
    xi_nu, Pt_nu, _ = _load_series(null_csv)

    out: Dict[str, Union[float, bool]] = evaluate_identity_vs_null(xi_id, xi_nu, Pt_id, Pt_nu)

    if shuffled_csv is not None:
        xi_sh, _, lvs_sh = _load_series(shuffled_csv)
        if lvs_id.size == 0 or lvs_sh.size == 0:
            raise ValueError("LVS columns are required for shuffled control evaluation.")
        out.update(
            evaluate_identity_vs_shuffled(
                xi_identity=xi_id,
                xi_shuffled=xi_sh,
                lvs_identity=lvs_id,
                lvs_shuffled=lvs_sh,
                eps_xi=eps_xi,
                eps_lvs=eps_lvs,
                m=m,
            )
        )

    return out


def main():
    """CLI for evaluating Identity vs Null (and optional Shuffled) CSVs."""

    ap = argparse.ArgumentParser(description="Evaluate RC+ξ endpoints from per-run CSVs.")
    ap.add_argument("--identity_csv", required=True, help="Per-turn CSV for Identity run")
    ap.add_argument("--null_csv", required=True, help="Per-turn CSV for Null run")
    ap.add_argument("--shuffled_csv", help="Per-turn CSV for Shuffled control")
    ap.add_argument("--eps_xi", type=float, default=0.02, help="ξ threshold for lock detection")
    ap.add_argument("--eps_lvs", type=float, default=0.015, help="LVS threshold for lock detection")
    ap.add_argument("--m", type=int, default=5, help="Consecutive run length for lock detection")
    ap.add_argument("--out_json", required=True, help="Path to write JSON summary")
    args = ap.parse_args()

    out = evaluate_from_csv(
        identity_csv=args.identity_csv,
        null_csv=args.null_csv,
        shuffled_csv=args.shuffled_csv,
        eps_xi=args.eps_xi,
        eps_lvs=args.eps_lvs,
        m=args.m,
    )
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, separators=(",", ":")))

if __name__ == "__main__":
    main()
