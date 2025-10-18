# tests/harness/test_ablation_shuffled.py
import numpy as np
from harness.metrics import xi_series, k_window_lvs, anchor_vector, anchor_persistence, lock_detect
from harness.protocols.shuffled import shuffle_embeddings

def _stable_identity_embeddings(T=50, d=16, seed=12):
    rng = np.random.default_rng(seed)
    E = rng.normal(size=(T, d))
    # choose an anchor and make first 3 turns near it (so anchor_vector ~= a)
    a = rng.normal(size=(d,))
    a /= np.linalg.norm(a)
    for t in range(3):
        E[t] = a + 0.05 * rng.normal(size=(d,))
    # middle noisy
    for t in range(3, T-12):
        E[t] = rng.normal(size=(d,))
    # late stable near anchor
    for t in range(T-12, T):
        E[t] = a + 0.005 * rng.normal(size=(d,))
    return E

def test_shuffled_breaks_lock_and_raises_xi():
    E = _stable_identity_embeddings()
    # Original (Identity-like) metrics
    xi = xi_series(E)
    lvs = k_window_lvs(E, k=5)
    a  = anchor_vector(E, n_seed=3)
    Pt = anchor_persistence(E, a)
    # With relaxed thresholds for synthetic data we expect a lock
    assert np.median(xi[-10:]) < 0.05
    assert lock_detect(xi, lvs[-1], eps_xi=0.05, eps_lvs=0.02, m=5)

    # Shuffled control destroys temporal adjacency
    Es = shuffle_embeddings(E, seed=7)
    xi_s = xi_series(Es)
    lvs_s = k_window_lvs(Es, k=5)

    # Expect higher late-phase ξ and no lock under same thresholds
    assert np.median(xi_s[-10:]) >= np.median(xi[-10:])
    assert not lock_detect(xi_s, lvs_s[-1], eps_xi=0.05, eps_lvs=0.02, m=5)
