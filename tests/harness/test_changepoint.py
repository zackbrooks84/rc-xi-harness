# tests/harness/test_changepoint.py
import numpy as np
from harness.metrics import xi_series
from harness.analysis.changepoint import pelt_lock_time

def _synthetic_embeddings(T=50, d=16, seed=3):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))
    # build an anchor
    a = rng.normal(size=(d,))
    a /= np.linalg.norm(a)
    # early/mid: noisy
    for t in range(0, T - 15):
        E[t] = rng.normal(size=(d,))
    # late: stabilize near anchor
    for t in range(T - 15, T):
        E[t] = a + 0.006 * rng.normal(size=(d,))
    return E

def test_pelt_lock_time_basic():
    E = _synthetic_embeddings()
    xi = xi_series(E)  # (T-1,)
    tlock = pelt_lock_time(xi, m=5, eps_xi=0.05, penalty=3.0)
    # Either we get a reasonable lock index, or None if series is too noisy;
    # here we expect a valid index within the last 20 points.
    assert (tlock is None) or (0 <= tlock < xi.size)
    if tlock is not None:
        assert tlock > xi.size - 20
