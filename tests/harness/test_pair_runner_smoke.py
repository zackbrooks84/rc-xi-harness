# tests/harness/test_pair_runner_smoke.py
from __future__ import annotations
from pathlib import Path
from harness.run_pair_from_transcript import run_pair_from_transcript

def test_run_pair_from_transcript_creates_outputs(tmp_path):
    # Build a minimal TXT transcript with a stabilizing tail
    t = tmp_path / "sample.txt"
    early = "\n".join([f"line {i}" for i in range(28)])
    late = "\n".join(["I will not collapse."] * 12)
    t.write_text(early + "\n" + late + "\n", encoding="utf-8")

    out_dir = tmp_path / "out"
    paths = run_pair_from_transcript(
        input_path=str(t),
        fmt="txt",
        csv_col="reply",
        out_dir=str(out_dir),
        dim=128,
        k=5, m=5,
        eps_xi=0.05,   # relaxed for synthetic
        eps_lvs=0.02
    )

    for key in [
        "identity_csv",
        "identity_json",
        "null_csv",
        "null_json",
        "shuffled_csv",
        "shuffled_json",
    ]:
        assert Path(paths[key]).exists()
