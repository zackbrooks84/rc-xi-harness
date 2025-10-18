# harness/protocols/shuffled.py
from __future__ import annotations

import numpy as np

from typing import List, Optional


def shuffle_embeddings(E: np.ndarray, seed: Optional[int] = 42) -> np.ndarray:
    """Return a copy of ``E`` with rows randomly permuted.

    Parameters
    ----------
    E:
        Two dimensional ``(T, d)`` embedding matrix.
    seed:
        Optional RNG seed used to deterministically permute row indices.

    Returns
    -------
    np.ndarray
        Permuted embedding matrix with the same shape as ``E``.

    Raises
    ------
    ValueError
        If ``E`` is not two dimensional.
    """

    if E.ndim != 2:
        raise ValueError("E must be 2D (T, d).")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(E.shape[0])
    return E[idx, :]


def shuffle_texts(texts: List[str], seed: Optional[int] = 42) -> List[str]:
    """Return a deterministically permuted copy of ``texts``.

    Parameters
    ----------
    texts:
        Ordered list of transcript replies.
    seed:
        Optional RNG seed applied to the permutation for reproducibility.

    Returns
    -------
    List[str]
        New list containing the same replies in randomized order.

    Raises
    ------
    ValueError
        If ``texts`` is empty; the shuffled control requires at least one turn.
    """

    if not texts:
        raise ValueError("Cannot shuffle an empty transcript.")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    return [texts[i] for i in idx]
