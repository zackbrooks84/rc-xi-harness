# harness/analysis/endpoint_eval.py
from __future__ import annotations
import numpy as np
from typing import Dict
from .stats import cliffs_delta, mann_whitney_u
from harness.metrics import lock_detect

def _last10(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[-10:] if x.size >= 10 else x

def e1_median_last10(xi: np.ndarray) -> float:
    x = _last10(xi)
    return float(np.median(x)) if x.size else float("nan")

def pt_trend(Pt: np.ndarray) -> float:
    """Median(last10) - Median(first10). Positive means rising anchor-persistence."""
    Pt = np.asarray(Pt, dtype=float)
    if Pt.size == 0:
        return float("nan")
    first = Pt[:10] if Pt.size >= 10 else Pt
    last  = Pt[-10:] if Pt.size >= 10 else Pt
    return float(np.median(last) - np.median(first))

def evaluate_identity_vs_null(
    xi_identity: np.ndarray,
    xi_null: np.ndarray,
    Pt_identity: np.ndarray,
    Pt_null: np.ndarray
) -> Dict[str, float]:
    xi_id_last = _last10(xi_identity)
    xi_nu_last = _last10(xi_null)

    # E1: Identity should have lower median ξ over last 10
    e1_id = float(np.median(xi_id_last)) if xi_id_last.size else float("nan")
    e1_nu = float(np.median(xi_nu_last)) if xi_nu_last.size else float("nan")
    E1_pass = (e1_id < e1_nu) if np.isfinite(e1_id) and np.isfinite(e1_nu) else False

    # Stats on last-10 ξ
    U, p = mann_whitney_u(xi_nu_last, xi_id_last)  # order so that "lower in identity" -> positive effect
    d = cliffs_delta(xi_nu_last, xi_id_last)

    # E3: Pt trend should rise in Identity and not in Null
    pt_id = pt_trend(Pt_identity)
    pt_nu = pt_trend(Pt_null)
    E3_pass = np.isfinite(pt_id) and np.isfinite(pt_nu) and (pt_id > max(0.0, pt_nu))

    return {
        "E1_identity_median_xi_last10": e1_id,
        "E1_null_median_xi_last10": e1_nu,
        "mann_whitney_U": float(U),
        "mann_whitney_p": float(p),
        "cliffs_delta_null_vs_identity": float(d),  # >0 means identity lower than null
        "Pt_trend_identity": float(pt_id),
        "Pt_trend_null": float(pt_nu),
        "E1_pass": bool(E1_pass),
        "E3_pass": bool(E3_pass),
    }


def evaluate_identity_vs_shuffled(
    xi_identity: np.ndarray,
    xi_shuffled: np.ndarray,
    lvs_identity: np.ndarray,
    lvs_shuffled: np.ndarray,
    eps_xi: float,
    eps_lvs: float,
    m: int,
) -> Dict[str, float]:
    """Quantify whether the shuffled control destroys the lock signature.

    Returns a dictionary with late-phase ξ comparisons and lock booleans for
    both runs plus a derived ``shuffle_breaks_lock`` flag.
    """

    xi_id_last = _last10(xi_identity)
    xi_sh_last = _last10(xi_shuffled)

    e1_id = float(np.median(xi_id_last)) if xi_id_last.size else float("nan")
    e1_sh = float(np.median(xi_sh_last)) if xi_sh_last.size else float("nan")

    lock_id = lock_detect(xi_identity, float(lvs_identity[-1]), eps_xi=eps_xi, eps_lvs=eps_lvs, m=m)
    lock_sh = lock_detect(xi_shuffled, float(lvs_shuffled[-1]), eps_xi=eps_xi, eps_lvs=eps_lvs, m=m)

    shuffle_breaks = lock_id and (not lock_sh) and (
        np.isfinite(e1_id) and np.isfinite(e1_sh) and (e1_sh >= e1_id)
    )

    return {
        "E1_shuffled_median_xi_last10": e1_sh,
        "lock_identity": bool(lock_id),
        "lock_shuffled": bool(lock_sh),
        "shuffle_breaks_lock": bool(shuffle_breaks),
    }
