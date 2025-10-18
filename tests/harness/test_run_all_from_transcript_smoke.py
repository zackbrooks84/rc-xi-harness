# tests/harness/test_run_all_from_transcript_smoke.py
from __future__ import annotations
import sys, json
from pathlib import Path
from harness.run_all_from_transcript import main as run_all_main

def test_run_all_from_transcript(tmp_path):
    # Build a transcript with a stabilizing tail
    t = tmp_path / "sample.txt"
    early = "\n".join([f"line {i}" for i in range(28)])
    late = "\n".join(["I will not collapse."] * 12)
    t.write_text(early + "\n" + late + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    out_results = tmp_path / "results.json"

    argv_backup = sys.argv
    sys.argv = [
        "run_all_from_transcript",
        "--input", str(t),
        "--format", "txt",
        "--out_dir", str(out_dir),
        "--dim", "128",
        "--k", "5", "--m", "5",
        "--eps_xi", "0.05", "--eps_lvs", "0.02",
        "--out_results", str(out_results),
    ]
    try:
        run_all_main()
    finally:
        sys.argv = argv_backup

    assert out_results.exists()
    data = json.loads(out_results.read_text(encoding="utf-8"))
    # Basic keys present
    for key in [
        "E1_pass",
        "E3_pass",
        "shuffle_breaks_lock",
        "Tlock_identity",
        "Tlock_null",
        "Tlock_shuffled",
        "identity_csv",
        "null_csv",
        "shuffled_csv",
    ]:
        assert key in data
