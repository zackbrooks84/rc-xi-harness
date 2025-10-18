# tests/harness/test_eval_cli_smoke.py
from __future__ import annotations

import numpy as np
from pathlib import Path

from harness.analysis.eval_cli import evaluate_from_csv
from harness.io.schema import write_rows


def _make_csv(
    path: Path,
    xi_series,
    Pt_series,
    lvs_series=None,
    run_type: str = "identity",
    provider: str = "dummy",
) -> None:
    rows = []
    T = len(Pt_series)
    if lvs_series is None:
        lvs_series = [0.0] * T
    assert len(lvs_series) == T
    # xi has length T-1 conceptually (blank at t=0 in our per-turn CSV)
    for t in range(T):
        rows.append({
            "t": t,
            "xi": "" if t == 0 else float(xi_series[t-1]),
            "lvs": float(lvs_series[t]),
            "Pt": float(Pt_series[t]),
            "ewma_xi": "",
            "run_type": run_type,
            "provider": provider
        })
    write_rows(str(path), rows)

def test_evaluate_from_csv_identity_vs_null(tmp_path):
    # Build synthetic series: Identity gets smaller ξ late; Pt rises.
    T = 40
    xi_identity = list(np.concatenate([np.full(20, 0.12), np.full(T-1-20, 0.02)]))
    xi_null     = list(np.full(T-1, 0.10))
    Pt_identity = list(np.linspace(0.2, 0.8, T))
    Pt_null     = list(np.linspace(0.6, 0.55, T))
    lvs_identity = list(np.linspace(0.03, 0.01, T))
    lvs_null = list(np.linspace(0.03, 0.025, T))

    id_csv = tmp_path / "id.csv"
    nu_csv = tmp_path / "nu.csv"
    _make_csv(id_csv, xi_identity, Pt_identity, lvs_identity, run_type="identity")
    _make_csv(nu_csv, xi_null, Pt_null, lvs_null, run_type="null")

    # Shuffled control keeps ξ high and LVS noisy
    xi_shuffled = list(np.full(T-1, 0.12))
    lvs_shuffled = list(np.linspace(0.04, 0.035, T))
    Pt_shuffled = list(np.linspace(0.2, 0.25, T))
    sh_csv = tmp_path / "sh.csv"
    _make_csv(sh_csv, xi_shuffled, Pt_shuffled, lvs_shuffled, run_type="shuffled")

    out = evaluate_from_csv(
        str(id_csv),
        str(nu_csv),
        shuffled_csv=str(sh_csv),
        eps_xi=0.05,
        eps_lvs=0.02,
        m=5,
    )

    assert out["E1_pass"] is True
    assert out["E3_pass"] is True
    assert 0.0 <= out["mann_whitney_p"] <= 1.0
    assert out["lock_identity"] is True
    assert out["lock_shuffled"] is False
    assert out["shuffle_breaks_lock"] is True
