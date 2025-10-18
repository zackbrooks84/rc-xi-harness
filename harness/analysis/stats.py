# harness/analysis/stats.py
from __future__ import annotations
import math
import numpy as np
from typing import Tuple

def mann_whitney_u(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Return (U, p_approx) using a normal approximation with tie correction.
    No SciPy dependency. Two-sided p-value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")

    z = np.concatenate([x, y])
    ranks = _rankdata(z)  # average ranks for ties
    R1 = np.sum(ranks[:n1])
    U1 = R1 - n1 * (n1 + 1) / 2.0
    U2 = n1 * n2 - U1
    U = min(U1, U2)

    # tie correction
    ties = _tie_counts(z)
    tie_term = np.sum(t * (t*t - 1) for t in ties)
    var = n1 * n2 * (n1 + n2 + 1) / 12.0
    if tie_term > 0:
        var -= n1 * n2 * tie_term / (12.0 * (n1 + n2) * (n1 + n2 - 1))

    if var <= 0:
        return U, 1.0

    mean = n1 * n2 / 2.0
    zscore = (U - mean) / math.sqrt(var + 1e-12)
    # two-sided p via normal approx
    p = 2.0 * (1.0 - _phi(abs(zscore)))
    return float(U), float(max(min(p, 1.0), 0.0))

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """
    Cliff's delta (d) in [-1, 1]. Positive => x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    # Efficient pairwise comparison using broadcasting signs
    diff = x[:, None] - y[None, :]
    gt = np.sum(diff > 0)
    lt = np.sum(diff < 0)
    d = (gt - lt) / (n1 * n2)
    return float(d)

# ----- helpers -----
def _phi(z: float) -> float:
    """CDF of standard normal (Abramowitz & Stegun)."""
    # error function approximation
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def _rankdata(a: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties (1-based ranks). Returns float array.
    """
    idx = np.argsort(a, kind="mergesort")
    ranks = np.empty_like(a, dtype=float)
    ranks[idx] = np.arange(1, len(a) + 1, dtype=float)

    # average ranks for ties
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[idx[j]] == a[idx[i]]:
            j += 1
        if j - i > 1:
            r = (ranks[idx[i]] + ranks[idx[j - 1]]) / 2.0
            ranks[idx[i:j]] = r
        i = j
    return ranks

def _tie_counts(a: np.ndarray) -> np.ndarray:
    """Return counts of ties (values occurring >1)."""
    vals, counts = np.unique(a, return_counts=True)
    return counts[counts > 1]
