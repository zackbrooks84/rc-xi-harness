# harness/embeddings/providers.py
from __future__ import annotations
from typing import List, Protocol
import numpy as np

class EmbeddingProvider(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Return an array of shape (T, d) for a list of T texts.
        Implementations should L2-normalize or leave raw; the metrics module
        will normalize again to be safe.
        """
        ...

class NumpyFileProvider:
    """
    Simple provider for wiring the harness before API keys.
    It ignores the given texts and loads a precomputed .npy matrix.
    """
    def __init__(self, path: str):
        self.path = path

    def embed(self, texts: List[str]) -> np.ndarray:
        E = np.load(self.path)  # expected shape (T, d)
        if E.ndim != 2:
            raise ValueError("Embeddings array must be 2D (T, d).")
        return E

# ─────────────────────────────────────────────────────────────────────────────
# PLANNED — NOT YET IMPLEMENTED
#
# The providers below are stubs. All validated results in the published study
# use SentenceTransformerProvider exclusively. Cross-provider replication
# (OpenAI, Cohere) is a planned next step.
#
# Reference: DOI: 10.5281/zenodo.17203755
# ─────────────────────────────────────────────────────────────────────────────

class OpenAIProvider:
    """PLANNED — NOT YET IMPLEMENTED.

    Use SentenceTransformerProvider for all current harness runs.
    See DOI: 10.5281/zenodo.17203755 for validated configuration.
    """
    def __init__(self, model: str = "text-embedding-3-large"):
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "OpenAIProvider is not yet implemented. "
            "Use SentenceTransformerProvider instead."
        )

class CohereProvider:
    """PLANNED — NOT YET IMPLEMENTED.

    Use SentenceTransformerProvider for all current harness runs.
    See DOI: 10.5281/zenodo.17203755 for validated configuration.
    """
    def __init__(self, model: str = "embed-english-v3.0"):
        self.model = model

    def embed(self, texts: List[str]) -> np.ndarray:
        raise NotImplementedError(
            "CohereProvider is not yet implemented. "
            "Use SentenceTransformerProvider instead."
        )
