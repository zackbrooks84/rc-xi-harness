"""OpenAI embedding provider for the RC+xi harness."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional

import numpy as np


@dataclass(frozen=True)
class OpenAIEmbeddingProvider:
    """Embedding provider backed by the OpenAI Embeddings API.

    Parameters
    ----------
    model_name:
        OpenAI embedding model. Options:
        - text-embedding-3-small  (1536 dims, fast, cheap)
        - text-embedding-3-large  (3072 dims, best quality)
        - text-embedding-ada-002  (1536 dims, legacy)
    api_key:
        OpenAI API key. Defaults to OPENAI_API_KEY env var.
    normalize:
        L2-normalize embeddings before returning. Default True.
    batch_size:
        Max texts per API call. Default 64 (well under the 2048 limit).
    """

    model_name: str = "text-embedding-3-large"
    api_key: Optional[str] = None
    normalize: bool = True
    batch_size: int = 64

    def __post_init__(self) -> None:
        try:
            from openai import OpenAI  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Install openai to use OpenAIEmbeddingProvider: pip install openai"
            ) from exc

        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "No OpenAI API key found. Set OPENAI_API_KEY or pass api_key=."
            )
        from openai import OpenAI
        object.__setattr__(self, "_client", OpenAI(api_key=key))

        # Resolve dim by making a single test call
        test = self._client.embeddings.create(
            model=self.model_name, input=["test"], encoding_format="float"
        )
        dim = len(test.data[0].embedding)
        object.__setattr__(self, "dim", dim)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        seq: List[str] = [t if t else " " for t in texts]
        if not seq:
            return np.zeros((0, self.dim), dtype=float)

        all_embeddings: List[List[float]] = []
        for i in range(0, len(seq), self.batch_size):
            batch = seq[i : i + self.batch_size]
            response = self._client.embeddings.create(
                model=self.model_name,
                input=batch,
                encoding_format="float",
            )
            response.data.sort(key=lambda d: d.index)
            all_embeddings.extend(d.embedding for d in response.data)

        E = np.array(all_embeddings, dtype=float)
        if self.normalize:
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            norms = np.where(norms > 0, norms, 1.0)
            E = E / norms
        return E
