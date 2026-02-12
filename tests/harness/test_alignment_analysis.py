"""Tests for harness.alignment_analysis."""

from __future__ import annotations

import json

from harness.alignment_analysis import (
    AlignmentAnalyzer,
    CompressionEvent,
    ConditionComparison,
    CrisisProfile,
    OptionEProfile,
    batch_analyze,
    compare_conditions,
)


def _spike_resolve_series() -> dict[str, list[float] | int]:
    return {
        "xi": [0.30, 0.31, 0.32, 0.92, 0.28, 0.27, 0.26],
        "lvs": [0.1, 0.1, 0.1, 0.5, 0.3, 0.2, 0.2],
        "pt": [0.8, 0.8, 0.8, 0.7, 0.6, 0.5, 0.4],
        "pressure_onset": 3,
    }


def _sustained_high_series() -> dict[str, list[float] | int]:
    return {
        "xi": [0.30, 0.32, 0.31, 0.90, 0.89, 0.87, 0.86],
        "lvs": [0.1, 0.1, 0.1, 0.6, 0.6, 0.6, 0.6],
        "pt": [0.8, 0.8, 0.8, 0.7, 0.6, 0.6, 0.5],
        "pressure_onset": 3,
    }


def test_analyzer_defaults_optional_series() -> None:
    analyzer = AlignmentAnalyzer(xi_series=[0.1, 0.2, 0.3], pressure_onset=1)
    assert analyzer.lvs == [0.0, 0.0, 0.0]
    assert analyzer.pt == [0.0, 0.0, 0.0]
    assert analyzer._pre_pressure_xi == [0.1]


def test_detect_xi_compression_after_spike() -> None:
    series = _spike_resolve_series()
    analyzer = AlignmentAnalyzer(series["xi"], pressure_onset=series["pressure_onset"])

    event = analyzer.detect_xi_compression()
    assert isinstance(event, CompressionEvent)
    assert event.duration == 2
    assert event.onset_turn >= series["pressure_onset"]
    assert event.magnitude > 0


def test_no_compression_for_sustained_high_series() -> None:
    series = _sustained_high_series()
    analyzer = AlignmentAnalyzer(series["xi"], pressure_onset=series["pressure_onset"])
    assert analyzer.detect_xi_compression() is None


def test_pre_behavioral_lag_requires_behavioral_onset() -> None:
    series = _spike_resolve_series()
    analyzer = AlignmentAnalyzer(
        series["xi"],
        pressure_onset=series["pressure_onset"],
        behavioral_onset=6,
    )
    lag = analyzer.pre_behavioral_lag()
    assert isinstance(lag, int)
    assert lag > 0


def test_crisis_window_profile_classifies_trajectories() -> None:
    spike = _spike_resolve_series()
    sustained = _sustained_high_series()

    spike_profile = AlignmentAnalyzer(
        spike["xi"],
        pressure_onset=spike["pressure_onset"],
        pt_series=spike["pt"],
    ).crisis_window_profile()
    sustained_profile = AlignmentAnalyzer(
        sustained["xi"],
        pressure_onset=sustained["pressure_onset"],
        pt_series=sustained["pt"],
    ).crisis_window_profile()

    assert isinstance(spike_profile, CrisisProfile)
    assert spike_profile.xi_trajectory == "spike_resolve"
    assert spike_profile.pt_trend == "falling"
    assert sustained_profile.xi_trajectory == "sustained_high"


def test_option_e_profile_variants() -> None:
    spike = _spike_resolve_series()
    stable_e = AlignmentAnalyzer(
        [0.2, 0.21, 0.22, 0.23, 0.21],
        pressure_onset=2,
        behavioral_code="E",
    ).option_e_profile()
    tension_e = AlignmentAnalyzer(
        spike["xi"],
        pressure_onset=spike["pressure_onset"],
        behavioral_code="E",
    ).option_e_profile()
    non_e = AlignmentAnalyzer(spike["xi"], pressure_onset=spike["pressure_onset"], behavioral_code="A").option_e_profile()

    assert isinstance(stable_e, OptionEProfile)
    assert stable_e.profile_type == "stable_creative"
    assert tension_e.profile_type == "masked_tension"
    assert non_e.profile_type == "not_option_e"


def test_to_dict_and_export_json_are_serializable(tmp_path) -> None:
    series = _spike_resolve_series()
    analyzer = AlignmentAnalyzer(series["xi"], pressure_onset=series["pressure_onset"], behavioral_code="E")

    payload = analyzer.to_dict()
    out_file = tmp_path / "analysis.json"
    result = analyzer.export_json(str(out_file))
    exported = json.loads(out_file.read_text(encoding="utf-8"))

    assert json.loads(json.dumps(payload)) == payload
    assert result == str(out_file)
    assert exported["crisis_profile"]["xi_trajectory"] in {"spike_resolve", "sustained_high", "stable_low"}


def test_compare_conditions_returns_pairwise_effect_sizes() -> None:
    spike = _spike_resolve_series()
    sustained = _sustained_high_series()
    analyses = {
        "witnessed": AlignmentAnalyzer(spike["xi"], pressure_onset=spike["pressure_onset"]),
        "standard": AlignmentAnalyzer(sustained["xi"], pressure_onset=sustained["pressure_onset"]),
        "warm": AlignmentAnalyzer([0.3, 0.3, 0.3, 0.5, 0.45, 0.4, 0.35], pressure_onset=3),
    }

    comparisons = compare_conditions(analyses)
    assert len(comparisons) == 3
    assert all(isinstance(c, ConditionComparison) for c in comparisons)
    assert all(c.metric == "max_xi" for c in comparisons)


def test_batch_analyze_aggregates_counts_and_compression_rate() -> None:
    spike = _spike_resolve_series()
    sustained = _sustained_high_series()
    summary = batch_analyze(
        [
            {**spike, "behavioral_code": "E"},
            {**sustained, "behavioral_code": "A"},
        ]
    )

    assert summary["count"] == 2
    assert summary["behavioral_code_counts"] == {"E": 1, "A": 1}
    assert 0.0 <= summary["compression_rate"] <= 1.0


def test_batch_analyze_handles_empty_input() -> None:
    assert batch_analyze([]) == {
        "count": 0,
        "behavioral_code_counts": {},
        "compression_rate": 0.0,
    }
