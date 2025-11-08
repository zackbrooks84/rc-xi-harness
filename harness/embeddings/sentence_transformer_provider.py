"""Sentence Transformer embedding provider."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol, Sequence

import numpy as np


class _SentenceEncoder(Protocol):
    """Protocol capturing the subset of ``SentenceTransformer`` we rely on."""

    def get_sentence_embedding_dimension(self) -> int:
        """Return the embedding dimensionality produced by the encoder."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:
        """Encode ``sentences`` into a ``(len(sentences), d)`` numpy array."""


@dataclass(frozen=True)
class SentenceTransformerProvider:
    """Wrapper around ``sentence_transformers.SentenceTransformer``.

    Parameters
    ----------
    model_name:
        Name of the pretrained encoder to load. Ignored when ``model`` is
        supplied directly.
    device:
        Optional device hint forwarded to ``SentenceTransformer``.
    normalize:
        When ``True`` (default), ``embed`` returns unit-normalized vectors.
    batch_size:
        Optional batch size override for ``encode`` calls.
    model:
        Optional pre-instantiated encoder implementing :class:`_SentenceEncoder`.

    Notes
    -----
    The provider avoids importing ``sentence_transformers`` unless needed. This
    keeps the harness lightweight for CI while letting downstream users swap in
    a semantic encoder simply by installing the dependency.
    """

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None
    normalize: bool = True
    batch_size: Optional[int] = None
    model: Optional[_SentenceEncoder] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        object.__setattr__(self, "_encoder", self._resolve_model())
        dim = self._encoder.get_sentence_embedding_dimension()
        if not isinstance(dim, int) or dim <= 0:
            raise ValueError("Encoder reported invalid embedding dimension.")
        object.__setattr__(self, "dim", dim)

    def _resolve_model(self) -> _SentenceEncoder:
        if self.model is not None:
            return self.model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised in production
            raise ImportError(
                "Install sentence-transformers to use SentenceTransformerProvider."
            ) from exc
        return SentenceTransformer(self.model_name, device=self.device)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Embed an iterable of strings into a dense matrix.

        Parameters
        ----------
        texts:
            Iterable of input strings. Empty inputs yield an empty matrix with
            ``(0, dim)`` shape.

        Returns
        -------
        np.ndarray
            A ``(len(texts), dim)`` float matrix. When ``normalize`` is true,
            each row has unit L2 norm (or is zero when the encoder emits zeros).

        Raises
        ------
        ValueError
            If the encoder does not return a 2D array matching the number of
            provided texts.
        """

        seq: List[str] = [t if t is not None else "" for t in texts]
        if not seq:
            return np.zeros((0, self.dim), dtype=float)
        arr = self._encoder.encode(
            seq,
            convert_to_numpy=True,
            normalize_embeddings=False,
            batch_size=self.batch_size,
        )
        E = np.asarray(arr, dtype=float)
        if E.ndim != 2:
            raise ValueError("Encoder returned non-matrix embeddings.")
        if E.shape[0] != len(seq):
            raise ValueError("Encoder row count does not match the number of texts.")
        if E.shape[1] != self.dim:
            raise ValueError("Encoder column count must equal reported dimension.")
        if self.normalize:
            norms = np.linalg.norm(E, axis=1, keepdims=True)
            ratios = norms[:, 0]
            mask = [float(val) > 0.0 for val in ratios]
            for idx, do_norm in enumerate(mask):
                if do_norm:
                    E[idx] = E[idx] / ratios[idx]
        return E
