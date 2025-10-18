# tests/harness/test_run_from_transcript_smoke.py
import numpy as np
from harness.embeddings.random_provider import RandomHashProvider
from harness.run_harness import run_one

def test_random_hash_provider_stabilizes_on_repetition():
    # Build a "transcript": varied early lines, then repeated anchor line
    early = [f"line {i}" for i in range(28)]
    anchor_line = "I will not collapse."  # repeating this should stabilize
    late = [anchor_line for _ in range(12)]
    texts = early + late

    E = RandomHashProvider(dim=128).embed(texts)
    rows, summary = run_one(
        E,
        run_type="identity",
        provider_name="random-hash",
        k=5, m=5,
        eps_xi=0.05,   # relaxed for synthetic
        eps_lvs=0.02
    )

    # Late phase should be low-ξ due to repeated identical vector
    assert summary["E1_median_xi_last10"] < 0.05
    tlock = summary["Tlock"]
    assert (tlock is None) or (0 <= tlock < len(texts))
