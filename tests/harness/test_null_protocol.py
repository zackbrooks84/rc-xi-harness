# tests/harness/test_null_protocol.py
"""Tests for harness/protocols/null.py — external_null_texts function."""
from __future__ import annotations

import pytest

from harness.protocols.null import external_null_texts


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

IDENTITY = ["a", "b", "c", "d", "e"]
EXTERNAL_EXACT = ["x", "y", "z", "w", "v"]
EXTERNAL_LONGER = ["x", "y", "z", "w", "v", "u", "t", "s"]
EXTERNAL_SHORTER = ["x", "y"]


# ---------------------------------------------------------------------------
# Length matching
# ---------------------------------------------------------------------------

class TestExternalNullTexts:
    def test_matching_length_returns_external_unchanged(self):
        result = external_null_texts(IDENTITY, EXTERNAL_EXACT)
        assert result == EXTERNAL_EXACT
        assert len(result) == len(IDENTITY)

    def test_longer_external_is_truncated(self):
        result = external_null_texts(IDENTITY, EXTERNAL_LONGER)
        assert len(result) == len(IDENTITY)
        assert result == EXTERNAL_LONGER[: len(IDENTITY)]

    def test_shorter_external_is_cycled(self):
        # IDENTITY has 5 entries, EXTERNAL_SHORTER has 2 → cycles to fill 5
        result = external_null_texts(IDENTITY, EXTERNAL_SHORTER)
        assert len(result) == len(IDENTITY)
        # Expected: x y x y x  (cycling ["x", "y"])
        expected = ["x", "y", "x", "y", "x"]
        assert result == expected

    def test_shorter_external_cycles_cleanly_on_multiple(self):
        # 6 identity turns, 3 external → exactly 2 full cycles, no remainder
        identity = ["i"] * 6
        external = ["a", "b", "c"]
        result = external_null_texts(identity, external)
        assert result == ["a", "b", "c", "a", "b", "c"]

    def test_single_external_text_repeated(self):
        result = external_null_texts(IDENTITY, ["solo"])
        assert result == ["solo"] * len(IDENTITY)

    def test_empty_identity_returns_empty(self):
        result = external_null_texts([], EXTERNAL_EXACT)
        assert result == []

    def test_empty_external_raises(self):
        with pytest.raises(ValueError, match="empty"):
            external_null_texts(IDENTITY, [])

    def test_does_not_mutate_inputs(self):
        identity_copy = list(IDENTITY)
        external_copy = list(EXTERNAL_SHORTER)
        external_null_texts(IDENTITY, EXTERNAL_SHORTER)
        assert IDENTITY == identity_copy
        assert EXTERNAL_SHORTER == external_copy


# ---------------------------------------------------------------------------
# Integration: run_pair_from_transcript validation
# ---------------------------------------------------------------------------

class TestRunPairNullModeValidation:
    """Unit-level checks on the null_mode guard in run_pair_from_transcript."""

    def test_external_mode_without_path_raises(self):
        from harness.run_pair_from_transcript import run_pair_from_transcript
        with pytest.raises(ValueError, match="--null_transcript is required"):
            run_pair_from_transcript(
                input_path="data/sample_transcript.txt",
                fmt="txt",
                csv_col="reply",
                out_dir="out/test_tmp",
                null_mode="external",
                null_transcript_path=None,
            )

    def test_invalid_null_mode_raises(self):
        from harness.run_pair_from_transcript import run_pair_from_transcript
        with pytest.raises(ValueError, match="null_mode must be"):
            run_pair_from_transcript(
                input_path="data/sample_transcript.txt",
                fmt="txt",
                csv_col="reply",
                out_dir="out/test_tmp",
                null_mode="bogus",
                null_transcript_path=None,
            )
