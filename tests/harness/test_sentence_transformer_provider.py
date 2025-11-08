"""Tests for the SentenceTransformerProvider shim."""

from __future__ import annotations

import math
import numpy as np

from harness.embeddings.sentence_transformer_provider import SentenceTransformerProvider


class _StubEncoder:
    """Minimal stub emulating a SentenceTransformer encoder."""

    def __init__(self, dim: int = 4) -> None:
        self._dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences,
        *,
        convert_to_numpy=True,
        normalize_embeddings=False,
        batch_size=None,
    ):
        del convert_to_numpy, normalize_embeddings, batch_size
        arr = np.zeros((len(sentences), self._dim), dtype=float)
        for row, text in enumerate(sentences):
            base = float(len(text or ""))
            arr[row] = base + np.arange(self._dim, dtype=float)
        return arr


def test_sentence_transformer_provider_shape_and_dtype() -> None:
    """Provider emits a dense float matrix with consistent shape."""

    provider = SentenceTransformerProvider(model=_StubEncoder(dim=3), normalize=False)
    texts = ["alpha", "beta", ""]
    embeddings = provider.embed(texts)

    assert embeddings.shape == (3, 3)
    row0 = [float(x) for x in embeddings[0]]
    assert row0 == [5.0, 6.0, 7.0]


def test_sentence_transformer_provider_normalization_and_stability() -> None:
    """Normalization produces unit vectors and calls are deterministic."""

    provider = SentenceTransformerProvider(model=_StubEncoder(dim=2), normalize=True)
    texts = ["a", "abcd"]
    embeddings = provider.embed(texts)

    rows = [[float(x) for x in row] for row in embeddings]
    norms = [math.sqrt(sum(val * val for val in row)) for row in rows]
    assert all(math.isclose(norm, 1.0) for norm in norms)

    second = provider.embed(texts)
    rows_second = [[float(x) for x in row] for row in second]
    for first_row, second_row in zip(rows, rows_second):
        assert all(math.isclose(a, b) for a, b in zip(first_row, second_row))
