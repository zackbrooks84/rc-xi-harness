# harness/embeddings/random_provider.py
from __future__ import annotations
import numpy as np
import hashlib
from typing import List

class RandomHashProvider:
    """
    Deterministic, keyless embedding provider.
    Each text -> stable pseudo-random unit vector via SHA-256 seeding.
    Useful for wiring tests and CI without calling external APIs.
    """
    def __init__(self, dim: int = 384):
        self.dim = int(dim)

    def embed(self, texts: List[str]) -> np.ndarray:
        E = np.zeros((len(texts), self.dim), dtype=float)
        for i, t in enumerate(texts):
            h = hashlib.sha256((t or "").encode("utf-8")).digest()
            seed = int.from_bytes(h[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            v = rng.normal(size=(self.dim,))
            n = np.linalg.norm(v) + 1e-12
            E[i] = v / n
        return E
