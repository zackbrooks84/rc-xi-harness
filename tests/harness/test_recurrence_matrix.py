# tests/harness/test_recurrence_matrix.py
import numpy as np
from harness.analysis import recurrence_matrix

def _stable_late_embeddings(T=50, d=16, seed=7):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))
    # define an anchor direction
    a = rng.normal(size=(d,))
    a /= np.linalg.norm(a)
    # early/mid: noisy
    for t in range(0, T - 12):
        E[t] = rng.normal(size=(d,))
    # late: near anchor (stable)
    for t in range(T - 12, T):
        E[t] = a + 0.006 * rng.normal(size=(d,))
    return E

def _block_mean(M: np.ndarray, r0: int, r1: int, c0: int, c1: int) -> float:
    sub = M[r0:r1, c0:c1]
    return float(np.mean(sub))

def test_recurrence_has_late_low_distance_block():
    E = _stable_late_embeddings()
    D = recurrence_matrix(E)
    T = E.shape[0]
    assert D.shape == (T, T)

    # Compare a late (stable) 10x10 block vs a mid-run 10x10 block
    late_start = T - 10
    late_mean = _block_mean(D, late_start, T, late_start, T)
    mid_start = (T // 2) - 5
    mid_mean = _block_mean(D, mid_start, mid_start + 10, mid_start, mid_start + 10)

    # Expect the late block to show tighter recurrence (smaller mean distance)
    assert late_mean < mid_mean
    # And both are finite and reasonable
    assert np.isfinite(late_mean) and np.isfinite(mid_mean)
