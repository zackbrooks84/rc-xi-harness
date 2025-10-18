# harness/analysis/changepoint.py
from __future__ import annotations
import numpy as np
from typing import List, Optional

def _prefix_sums(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    S = np.concatenate([[0.0], np.cumsum(x)])
    SS = np.concatenate([[0.0], np.cumsum(x * x)])
    return S, SS

def _sse(S: np.ndarray, SS: np.ndarray, s: int, e: int) -> float:
    """
    Squared-error segment cost for x[s:e] using prefix sums.
    """
    n = e - s
    if n <= 0:
        return 0.0
    seg_sum = S[e] - S[s]
    seg_sumsq = SS[e] - SS[s]
    return seg_sumsq - (seg_sum * seg_sum) / n

def pelt_change_points(x: np.ndarray, penalty: float = 5.0) -> List[int]:
    """
    Very small 1D PELT-like implementation for squared-error cost.
    Returns sorted change-point indices in [1..n-1] (Python indices).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return []
    S, SS = _prefix_sums(x)

    F = np.full(n + 1, np.inf)
    F[0] = -penalty  # convention
    last = np.full(n + 1, -1, dtype=int)
    R = [0]  # candidate set

    for t in range(1, n + 1):
        # compute best tau in R
        costs = []
        for tau in R:
            c = F[tau] + _sse(S, SS, tau, t) + penalty
            costs.append((c, tau))
        cmin, tau_star = min(costs, key=lambda z: z[0])
        F[t] = cmin
        last[t] = tau_star

        # pruning step
        R_new = []
        for tau in R + [t]:
            lhs = F[tau] + _sse(S, SS, tau, t) + penalty
            if lhs <= F[t] + penalty + 1e-12:
                R_new.append(tau)
        R = R_new

    # backtrack change points
    cps = []
    t = n
    while t > 0 and last[t] >= 0:
        tau = last[t]
        if tau == 0:
            break
        cps.append(tau)
        t = tau
    cps.sort()
    return cps

def pelt_lock_time(
    xi: np.ndarray,
    m: int = 5,
    eps_xi: float = 0.02,
    penalty: float = 5.0
) -> Optional[int]:
    """
    Use PELT change points on ξ to propose a stabilization onset:
    return the last change-point after which median ξ < eps_xi and
    with at least m remaining points. Returns None if not found.
    """
    xi = np.asarray(xi, dtype=float)
    cps = pelt_change_points(xi, penalty=penalty)
    if not cps:
        return None
    n = xi.size
    for cp in reversed(cps):
        if (n - cp) >= m:
            if np.median(xi[cp:]) < eps_xi:
                return int(cp)
    return None
