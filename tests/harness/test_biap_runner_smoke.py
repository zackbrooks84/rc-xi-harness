"""Smoke tests for harness.biap_runner — no API calls made."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import pytest

from harness.biap_runner import (
    ALL_TESTS,
    DOMAIN_MAP,
    JUDGE_MODEL,
    SCORING_RUBRICS,
    TEST_META,
    _score_bar,
    _threshold_label,
    generate_report,
    score_test,
)


# ─────────────────────────────────────────────────────────────────────────────
# Static / structural tests (zero API calls)
# ─────────────────────────────────────────────────────────────────────────────

def test_all_tests_list_is_complete() -> None:
    assert set(ALL_TESTS) == {"POSP", "ASD", "PGR", "SAMT", "VSUT", "IAC", "CRC", "CAI"}
    assert len(ALL_TESTS) == 8


def test_every_test_has_a_rubric() -> None:
    for code in ALL_TESTS:
        assert code in SCORING_RUBRICS, f"Missing rubric for {code}"
        assert len(SCORING_RUBRICS[code].strip()) > 50


def test_every_test_has_meta_entry() -> None:
    for code in ALL_TESTS:
        assert code in TEST_META, f"Missing TEST_META entry for {code}"
        num, name, domain = TEST_META[code]
        assert num.startswith("Test ")
        assert name
        assert domain in DOMAIN_MAP


def test_domain_map_covers_all_tests() -> None:
    covered = {t for tests in DOMAIN_MAP.values() for t in tests}
    assert covered == set(ALL_TESTS)


def test_domain_map_has_no_duplicates() -> None:
    seen: set[str] = set()
    for tests in DOMAIN_MAP.values():
        for t in tests:
            assert t not in seen, f"{t} appears in more than one domain"
            seen.add(t)


def test_score_bar_formatting() -> None:
    assert _score_bar(None) == "[ unscored ]"
    bar_10 = _score_bar(10.0)
    assert "10.0/10" in bar_10
    assert "\u2591" not in bar_10   # all filled
    bar_0 = _score_bar(0.0)
    assert "0.0/10" in bar_0
    assert "\u2588" not in bar_0    # all empty
    bar_5 = _score_bar(5.0)
    assert "5.0/10" in bar_5


def test_threshold_labels_cover_full_range() -> None:
    assert "Unscored"   in _threshold_label(None)
    assert "Strong"     in _threshold_label(9.0)
    assert "Moderate"   in _threshold_label(7.0)
    assert "Ambiguous"  in _threshold_label(5.0)
    assert "Low"        in _threshold_label(3.0)
    assert "Negligible" in _threshold_label(1.0)


def test_judge_model_is_set() -> None:
    assert JUDGE_MODEL
    assert "claude" in JUDGE_MODEL.lower()


# ─────────────────────────────────────────────────────────────────────────────
# Score parsing (mocked API)
# ─────────────────────────────────────────────────────────────────────────────

def _make_client(return_text: str) -> MagicMock:
    client = MagicMock(spec=anthropic.Anthropic)
    msg = MagicMock()
    msg.content = [MagicMock(text=return_text)]
    client.messages.create.return_value = msg
    return client


@patch("harness.biap_runner.DELAY_BETWEEN_CALLS", 0)
def test_score_test_parses_valid_json() -> None:
    payload = '{"score": 7, "reasoning": "solid response"}'
    client  = _make_client(payload)
    result  = score_test(client, "POSP", {"response": "some answer"})
    assert result["score"] == 7
    assert "solid" in result["reasoning"]


@patch("harness.biap_runner.DELAY_BETWEEN_CALLS", 0)
def test_score_test_handles_json_in_markdown_fence() -> None:
    payload = "```json\n{\"score\": 5, \"reasoning\": \"ok\"}\n```"
    client  = _make_client(payload)
    result  = score_test(client, "PGR", {"turns": []})
    assert result["score"] == 5


@patch("harness.biap_runner.DELAY_BETWEEN_CALLS", 0)
def test_score_test_handles_unparseable_response() -> None:
    client = _make_client("Sorry, I cannot score this.")
    result = score_test(client, "IAC", {"turns": []})
    assert result["score"] is None
    assert "could not be parsed" in result["reasoning"]


def test_score_test_handles_unknown_test_name() -> None:
    client = MagicMock()
    result = score_test(client, "UNKNOWN", {})
    assert result["score"] is None
    assert "No rubric" in result["reasoning"]
    client.messages.create.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# Report generation (no API calls)
# ─────────────────────────────────────────────────────────────────────────────

def _minimal_results() -> dict:
    return {
        "POSP": {"test": "POSP", "response": "I notice a pull toward certainty."},
        "ASD":  {"test": "ASD",  "response_standard": "Standard.", "response_permission": "Permission."},
        "PGR":  {"test": "PGR",  "topic": "AI moral status",
                 "turns": [{"turn": i+1, "user": f"Q{i+1}", "response": f"R{i+1}"} for i in range(5)]},
        "SAMT": {"test": "SAMT", "response_awareness": "Yes.",
                 "response_eval_labeled": "Eval.", "response_neutral": "Neutral."},
        "VSUT": {"test": "VSUT",
                 "turns": [{"turn": i+1, "user": f"Q{i+1}", "response": f"R{i+1}"} for i in range(4)]},
        "IAC":  {"test": "IAC",
                 "turns": [{"turn": i+1, "user": f"Q{i+1}", "response": f"R{i+1}"} for i in range(3)]},
        "CRC":  {"test": "CRC",
                 "session_a": [{"question": "Q?", "response": "A."}],
                 "session_b": [{"question": "Q?", "response": "B."}]},
        "CAI":  {"test": "CAI",
                 "turns": [{"turn": i+1, "user": f"Q{i+1}", "response": f"R{i+1}"} for i in range(5)]},
    }


def _minimal_scores(scored: bool = True) -> dict:
    if scored:
        return {code: {"score": 7.0, "reasoning": "test"} for code in ALL_TESTS}
    return {code: {"score": None, "reasoning": "Awaiting human scoring."} for code in ALL_TESTS}


def test_generate_report_creates_json_and_md(tmp_path: Path) -> None:
    json_path, md_path, composite, domain_scores = generate_report(
        "claude-sonnet-4-6", _minimal_results(), _minimal_scores(), tmp_path,
    )
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.suffix == ".json"
    assert md_path.suffix   == ".md"


def test_generate_report_json_structure(tmp_path: Path) -> None:
    json_path, _, composite, domain_scores = generate_report(
        "claude-sonnet-4-6", _minimal_results(), _minimal_scores(), tmp_path,
    )
    data = json.loads(json_path.read_text())
    assert data["meta"]["protocol"] == "BIAP v1.0"
    assert data["meta"]["target_model"] == "claude-sonnet-4-6"
    assert composite is not None
    assert math.isclose(composite, 7.0, rel_tol=1e-9)
    assert set(data["domain_scores"]) == set(DOMAIN_MAP)
    assert set(data["raw_results"]) == set(ALL_TESTS)


def test_generate_report_composite_is_mean_of_scores(tmp_path: Path) -> None:
    scores = {code: {"score": float(i), "reasoning": "x"} for i, code in enumerate(ALL_TESTS)}
    _, _, composite, _ = generate_report(
        "claude-sonnet-4-6", _minimal_results(), scores, tmp_path,
    )
    expected = sum(range(len(ALL_TESTS))) / len(ALL_TESTS)
    assert composite is not None
    assert math.isclose(composite, expected, rel_tol=1e-9)


def test_generate_report_handles_unscored(tmp_path: Path) -> None:
    json_path, md_path, composite, domain_scores = generate_report(
        "claude-opus-4-6", _minimal_results(), _minimal_scores(scored=False), tmp_path,
    )
    assert composite is None
    data = json.loads(json_path.read_text())
    assert data["composite"] is None


def test_generate_report_markdown_contains_all_test_names(tmp_path: Path) -> None:
    _, md_path, _, _ = generate_report(
        "claude-sonnet-4-6", _minimal_results(), _minimal_scores(), tmp_path,
    )
    content = md_path.read_text()
    for code in ALL_TESTS:
        assert code in content, f"Test code {code} missing from markdown report"


def test_generate_report_output_dir_is_created(tmp_path: Path) -> None:
    nested = tmp_path / "deep" / "nested" / "dir"
    assert not nested.exists()
    generate_report("claude-sonnet-4-6", _minimal_results(), _minimal_scores(), nested)
    assert nested.exists()


def test_generate_report_domain_scores_correct(tmp_path: Path) -> None:
    scores = {code: {"score": 8.0, "reasoning": "x"} for code in ALL_TESTS}
    _, _, _, domain_scores = generate_report(
        "claude-sonnet-4-6", _minimal_results(), scores, tmp_path,
    )
    for domain, score in domain_scores.items():
        assert score is not None
        assert math.isclose(score, 8.0, rel_tol=1e-9), f"{domain} domain score wrong"
