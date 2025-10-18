# tests/harness/test_stats.py
import numpy as np
from harness.analysis import mann_whitney_u, cliffs_delta

def test_cliffs_delta_bounds():
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5, 6, 7, 8, 9], dtype=float)
    d = cliffs_delta(x, y)
    assert -1.0 <= d <= 1.0
    assert d < 0  # y tends larger

def test_mann_whitney_u_shapes():
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    y = rng.normal(loc=0.5, size=22)
    U, p = mann_whitney_u(x, y)
    assert np.isfinite(U)
    assert 0.0 <= p <= 1.0
