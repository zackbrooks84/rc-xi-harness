# harness/metrics/metrics.py
from __future__ import annotations
import numpy as np

def l2_normalize(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / norms

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    return float(np.dot(a, b))

def xi_series(E: np.ndarray) -> np.ndarray:
    """
    ξ_t = 1 - cos(e_t, e_{t-1}) for t>=1.
    E shape: (T, d). Internally L2-normalized.
    Returns shape: (T-1,)
    """
    E = l2_normalize(E)
    dots = np.sum(E[1:] * E[:-1], axis=1)
    return 1.0 - np.clip(dots, -1.0, 1.0)

def k_window_lvs(E: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Local variance of pairwise cosine distances over a rolling window of size k.
    E shape: (T, d). Returns shape: (T,)
    """
    E = l2_normalize(E)
    T = E.shape[0]
    out = np.zeros(T)
    for t in range(T):
        s = max(0, t - k + 1)
        W = E[s:t+1]                   # (k', d)
        # pairwise cosine distances in the window
        D = 1.0 - np.clip(W @ W.T, -1.0, 1.0)
        out[t] = np.var(D)
    return out

def anchor_vector(E: np.ndarray, n_seed: int = 3) -> np.ndarray:
    """Mean of the first n_seed turns (then normalized)."""
    a = np.mean(E[:n_seed], axis=0)
    return a / (np.linalg.norm(a) + 1e-12)

def anchor_persistence(E: np.ndarray, a: np.ndarray) -> np.ndarray:
    """P_t = cos(e_t, a). Returns shape: (T,)"""
    E = l2_normalize(E)
    return (E @ a.reshape(-1, 1)).ravel()

def ewma(x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Exponentially weighted moving average."""
    y = np.zeros_like(x, dtype=float)
    if x.size == 0:
        return y
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def lock_detect(xi: np.ndarray,
                lvs_last: float,
                eps_xi: float = 0.02,
                eps_lvs: float = 0.015,
                m: int = 5) -> bool:
    """
    Lock if the last m points of ξ are below eps_xi AND the latest LVS < eps_lvs.
    xi shape: (T-1,)
    """
    if xi.size < m:
        return False
    return np.all(xi[-m:] < eps_xi) and (lvs_last < eps_lvs)
