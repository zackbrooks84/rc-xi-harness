# harness/protocols/paraphrase_noise.py
from __future__ import annotations
import numpy as np
from typing import Optional

def paraphrase_noise_embeddings(
    E: np.ndarray,
    sigma: float = 0.01,
    seed: Optional[int] = 123
) -> np.ndarray:
    """
    Add small Gaussian noise to embeddings to simulate paraphrase-level changes.
    Shape preserved; semantics approximately preserved.
    """
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, d).")
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=E.shape) * sigma
    return E + noise
