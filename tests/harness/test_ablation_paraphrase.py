# tests/harness/test_ablation_paraphrase.py
import numpy as np
from harness.metrics import xi_series, k_window_lvs, anchor_vector, anchor_persistence, lock_detect
from harness.protocols.paraphrase_noise import paraphrase_noise_embeddings

def _identity_like(T=50, d=16, seed=33):
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

def test_paraphrase_noise_preserves_lock():
    E = _identity_like()
    xi = xi_series(E)
    lvs = k_window_lvs(E, k=5)
    # Identity-like series should lock under relaxed synthetic thresholds
    assert np.median(xi[-10:]) < 0.05
    assert lock_detect(xi, lvs[-1], eps_xi=0.05, eps_lvs=0.02, m=5)

    # Apply light paraphrase noise
    En = paraphrase_noise_embeddings(E, sigma=0.01, seed=7)
    xi_n = xi_series(En)
    lvs_n = k_window_lvs(En, k=5)

    # Lock should persist under same thresholds
    assert np.median(xi_n[-10:]) < 0.05
    assert lock_detect(xi_n, lvs_n[-1], eps_xi=0.05, eps_lvs=0.02, m=5)

    # Late-phase ξ should not degrade much (allow small slack due to noise)
    delta = float(np.median(xi_n[-10:]) - np.median(xi[-10:]))
    assert delta < 0.02
