"""Factory helpers for embedding providers."""

from __future__ import annotations

from typing import Any, Mapping

from harness.embeddings.random_provider import RandomHashProvider

try:  # pragma: no cover - optional dependency import path
    from harness.embeddings.sentence_transformer_provider import (
        SentenceTransformerProvider,
    )
except ImportError:  # pragma: no cover - only when file missing from old installs
    SentenceTransformerProvider = None  # type: ignore


def create_provider(name: str, params: Mapping[str, Any]) -> Any:
    """Instantiate an embedding provider by name.

    Parameters
    ----------
    name:
        Provider slug, e.g. ``"random-hash"`` or ``"sentence-transformer"``.
    params:
        Keyword arguments forwarded to the provider constructor.

    Returns
    -------
    Any
        Provider instance exposing an ``embed(List[str]) -> np.ndarray`` method.

    Raises
    ------
    ValueError
        If ``name`` is unsupported or required dependencies are missing.
    """

    key = name.strip().lower()
    if key == "random-hash":
        dim = int(params.get("dim", 384))
        return RandomHashProvider(dim=dim)
    if key == "sentence-transformer":
        if SentenceTransformerProvider is None:  # pragma: no cover
            raise ValueError(
                "SentenceTransformerProvider unavailable; ensure file is present."
            )
        return SentenceTransformerProvider(
            model_name=params.get("model_name", SentenceTransformerProvider.model_name),
            device=params.get("device"),
            normalize=params.get("normalize", True),
            batch_size=params.get("batch_size"),
            model=params.get("model"),
        )
    raise ValueError(f"Unknown embedding provider: {name}")
