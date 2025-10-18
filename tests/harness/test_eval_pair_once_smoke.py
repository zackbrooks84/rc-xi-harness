# tests/harness/test_eval_pair_once_smoke.py
from __future__ import annotations
import json
import numpy as np
from pathlib import Path
from harness.analysis.eval_pair_once import main as eval_pair_main
from harness.analysis.eval_cli import evaluate_from_csv
from harness.io.schema import write_rows

def _make_csv(path: Path, xi, Pt, run_type):
    rows = []
    T = len(Pt)
    for t in range(T):
        rows.append({
            "t": t,
            "xi": "" if t == 0 else float(xi[t-1]),
            "lvs": 0.0,
            "Pt": float(Pt[t]),
            "ewma_xi": "",
            "run_type": run_type,
            "provider": "dummy",
        })
    write_rows(str(path), rows)

def test_eval_pair_once(tmp_path, monkeypatch):
    # Build simple series where Identity wins on E1 and Pt trend
    T = 40
    xi_id  = list(np.concatenate([np.full(20, 0.12), np.full(T-1-20, 0.02)]))
    xi_nu  = list(np.full(T-1, 0.10))
    Pt_id  = list(np.linspace(0.2, 0.8, T))
    Pt_nu  = list(np.linspace(0.6, 0.55, T))

    id_csv = tmp_path / "id.csv"
    nu_csv = tmp_path / "nu.csv"
    _make_csv(id_csv, xi_id, Pt_id, "identity")
    _make_csv(nu_csv, xi_nu, Pt_nu, "null")

    # Minimal per-run JSONs with Tlock
    id_json = tmp_path / "id.json"
    nu_json = tmp_path / "nu.json"
    id_json.write_text(json.dumps({"Tlock": 25, "k": 5, "m": 5, "eps_xi": 0.02, "eps_lvs": 0.015}), encoding="utf-8")
    nu_json.write_text(json.dumps({"Tlock": None, "k": 5, "m": 5, "eps_xi": 0.02, "eps_lvs": 0.015}), encoding="utf-8")

    out_json = tmp_path / "out.json"

    # Run the evaluator via its main() by simulating argv
    import sys
    argv_backup = sys.argv
    sys.argv = [
        "eval_pair_once",
        "--identity_csv", str(id_csv),
        "--null_csv", str(nu_csv),
        "--identity_json", str(id_json),
        "--null_json", str(nu_json),
        "--out_json", str(out_json),
    ]
    try:
        eval_pair_main()
    finally:
        sys.argv = argv_backup

    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert "E1_pass" in out and "E3_pass" in out
    assert "Tlock_identity" in out and "Tlock_null" in out
