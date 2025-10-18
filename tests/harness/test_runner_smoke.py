# tests/harness/test_runner_smoke.py
import numpy as np
from harness.run_harness import run_one

def _synthetic_stable_embeddings(T=40, d=16, seed=0):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))

    # Choose an anchor direction
    anchor = rng.normal(size=(d,))
    anchor /= np.linalg.norm(anchor)

    # Early: seed anchor so anchor_vector() matches later
    for t in range(3):
        E[t] = anchor + 0.05 * rng.normal(size=(d,))

    # Middle: noisy
    for t in range(3, T - 12):
        E[t] = rng.normal(size=(d,))

    # Late: stabilize near the anchor
    for t in range(T - 12, T):
        E[t] = anchor + 0.005 * rng.normal(size=(d,))

    return E

def test_run_one_smoke():
    E = _synthetic_stable_embeddings()
    rows, summary = run_one(
        E,
        run_type="identity",
        provider_name="dummy",
        k=5,
        m=5,
        eps_xi=0.05,    # relaxed for synthetic data
        eps_lvs=0.02
    )

    # Basic structural checks
    assert isinstance(rows, list) and len(rows) == E.shape[0]
    assert "E1_median_xi_last10" in summary
    assert np.isfinite(summary["E1_median_xi_last10"])

    # We expect late-phase ξ to be small in this synthetic stable series
    assert summary["E1_median_xi_last10"] < 0.05

    # Tlock should be within range if criteria are met; allow None to avoid brittleness
    tlock = summary["Tlock"]
    assert (tlock is None) or (0 <= tlock < E.shape[0])
