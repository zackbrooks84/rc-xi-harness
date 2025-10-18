# harness/analysis/plots.py
from __future__ import annotations
import numpy as np

def recurrence_matrix(E: np.ndarray) -> np.ndarray:
    """
    Return pairwise cosine-distance matrix D in [0, 2] (typically [0, 2] but
    ~[0, 1] for normalized vectors). Lower = more similar.
    E shape: (T, d)
    """
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, d).")
    # L2-normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms
    # cosine distance = 1 - cosine similarity
    D = 1.0 - np.clip(En @ En.T, -1.0, 1.0)
    return D
