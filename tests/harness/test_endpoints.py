# tests/harness/test_endpoints.py
import numpy as np
from harness.metrics import (
    xi_series, k_window_lvs, anchor_vector,
    anchor_persistence, ewma, lock_detect
)

def test_lock_detect_smoke():
    rng = np.random.default_rng(0)
    T, d = 40, 16

    # Choose a true anchor and build a series around it
    anchor = rng.normal(size=(d,))
    anchor /= np.linalg.norm(anchor)

    E = rng.normal(size=(T, d))

    # Make the first 3 turns near the anchor so anchor_vector() ~= anchor
    for t in range(3):
        E[t] = anchor + 0.05 * rng.normal(size=(d,))

    # Middle region: arbitrary/noisy
    for t in range(3, T - 12):
        E[t] = rng.normal(size=(d,))

    # Late region: very close to anchor (stable phase)
    for t in range(T - 12, T):
        E[t] = anchor + 0.005 * rng.normal(size=(d,))

    # Metrics
    xi = xi_series(E)             # (T-1,)
    lvs = k_window_lvs(E, k=5)    # (T,)
    a = anchor_vector(E, n_seed=3)
    Pt = anchor_persistence(E, a)
    xi_s = ewma(xi, alpha=0.5)

    # Expectations for a stabilizing series
    assert np.median(xi[-10:]) < 0.05
    assert lock_detect(xi, lvs[-1], eps_xi=0.05, eps_lvs=0.02, m=5)
    assert Pt[-1] > Pt[5]                   # anchor persistence increases
    assert np.isfinite(xi_s).all()          # EWMA well-behaved
