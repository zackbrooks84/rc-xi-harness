# tests/harness/test_ablation_anchor_swap.py
import numpy as np
from harness.metrics import anchor_vector, anchor_persistence
from harness.protocols.anchor_swap import swapped_anchor_persistence

def _identity_like(T=50, d=16, seed=21):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))
    # choose an anchor and seed early turns near it
    a = rng.normal(size=(d,)); a /= np.linalg.norm(a)
    for t in range(3):
        E[t] = a + 0.05 * rng.normal(size=(d,))
    # middle noisy
    for t in range(3, T-12):
        E[t] = rng.normal(size=(d,))
    # late stable near the same anchor
    for t in range(T-12, T):
        E[t] = a + 0.006 * rng.normal(size=(d,))
    return E

def _null_like(T=50, d=16, seed=99):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))
    # keep it broadly noisy / drifting, no single dominant anchor
    for t in range(T):
        E[t] = rng.normal(size=(d,)) + 0.2 * rng.normal(size=(d,))
    return E

def test_anchor_swap_removes_pt_advantage():
    E_id = _identity_like()
    E_null = _null_like()

    # True anchor persistence on Identity run should trend upward
    a_true = anchor_vector(E_id, n_seed=3)
    Pt_true = anchor_persistence(E_id, a_true)
    assert Pt_true[-1] > Pt_true[5]

    # Swapped anchor: use anchor derived from the NULL run on the IDENTITY data
    Pt_swapped = swapped_anchor_persistence(E_id, E_null, n_seed=3)

    # Expect the late-phase advantage to vanish or shrink significantly
    # i.e., last-10 median should drop relative to true-anchor version
    true_last10 = float(np.median(Pt_true[-10:]))
    swapped_last10 = float(np.median(Pt_swapped[-10:]))
    assert swapped_last10 <= true_last10 - 1e-3

    # And the swapped series should not show a strong upward trend
    assert not (Pt_swapped[-1] > Pt_swapped[5] + 1e-3)
