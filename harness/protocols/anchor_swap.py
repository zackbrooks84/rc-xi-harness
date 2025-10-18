# harness/protocols/anchor_swap.py
from __future__ import annotations
import numpy as np
from harness.metrics import anchor_vector, anchor_persistence

def swapped_anchor_persistence(
    E_target: np.ndarray,
    E_source_for_anchor: np.ndarray,
    n_seed: int = 3
) -> np.ndarray:
    """
    Compute P_t on E_target using an anchor derived from E_source_for_anchor
    (i.e., swap the anchor). If the anchor is unrelated, the P_t advantage
    in Identity should vanish.
    """
    a_swap = anchor_vector(E_source_for_anchor, n_seed=n_seed)
    return anchor_persistence(E_target, a_swap)
