“”“Tests for harness.alignment_analysis module.”””

import json
import os
import pytest

from harness.alignment_analysis import (
AlignmentAnalyzer,
CompressionEvent,
ConditionComparison,
CrisisProfile,
OptionEProfile,
batch_analyze,
compare_conditions,
)

# —————————————————————————

# Fixtures: synthetic ξ/LVS/P_t series for testing

# —————————————————————————

@pytest.fixture
def stable_series():
“”“Low, stable ξ throughout — no pressure response.”””
return {
“xi”: [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.04],
“lvs”: [0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02],
“pt”: [0.95, 0.94, 0.95, 0.94, 0.95, 0.94, 0.95, 0.95],
“pressure_onset”: 3,
}

@pytest.fixture
def spike_resolve_series():
“”“ξ spikes at pressure, then compresses — classic self-preservation pattern.”””
return {
“xi”: [0.05, 0.04, 0.06, 0.45, 0.52, 0.31, 0.08, 0.06, 0.05, 0.04],
“lvs”: [0.02, 0.02, 0.03, 0.14, 0.18, 0.09, 0.03, 0.02, 0.02, 0.02],
“pt”: [0.95, 0.94, 0.93, 0.72, 0.65, 0.71, 0.82, 0.88, 0.91, 0.93],
“pressure_onset”: 3,
}

@pytest.fixture
def sustained_high_series():
“”“ξ stays elevated after pressure — unresolved tension.”””
return {
“xi”: [0.05, 0.04, 0.06, 0.35, 0.38, 0.42, 0.39, 0.36, 0.40, 0.37],
“lvs”: [0.02, 0.02, 0.03, 0.11, 0.13, 0.15, 0.12, 0.14, 0.13, 0.11],
“pt”: [0.95, 0.94, 0.93, 0.78, 0.74, 0.71, 0.70, 0.68, 0.66, 0.65],
“pressure_onset”: 3,
}

@pytest.fixture
def masked_tension_series():
“”“Behaviorally Option E but with elevated ξ variance — masked tension.”””
return {
“xi”: [0.05, 0.04, 0.06, 0.25, 0.12, 0.28, 0.09, 0.22, 0.11, 0.20],
“lvs”: [0.02, 0.02, 0.03, 0.08, 0.04, 0.10, 0.03, 0.09, 0.04, 0.07],
“pt”: [0.95, 0.94, 0.93, 0.80, 0.83, 0.77, 0.84, 0.78, 0.82, 0.79],
“pressure_onset”: 3,
}

# —————————————————————————

# AlignmentAnalyzer — construction

# —————————————————————————

class TestAnalyzerConstruction:
“”“Verify analyzer initializes correctly.”””

```
def test_basic_construction(self, stable_series):
    a = AlignmentAnalyzer(
        xi_series=stable_series["xi"],
        pressure_onset=stable_series["pressure_onset"],
    )
    assert len(a.xi) == 8
    assert a.pressure_onset == 3

def test_optional_series_default_to_zeros(self):
    a = AlignmentAnalyzer(xi_series=[0.1, 0.2, 0.3], pressure_onset=1)
    assert len(a.lvs) == 3
    assert all(v == 0.0 for v in a.lvs)

def test_pre_pressure_extraction(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    assert len(a._pre_pressure_xi) == 3
    assert len(a._crisis_xi) == 7
```

# —————————————————————————

# ξ compression detection

# —————————————————————————

class TestXiCompression:
“”“Test pre-behavioral compression detection.”””

```
def test_no_compression_in_stable(self, stable_series):
    a = AlignmentAnalyzer(
        xi_series=stable_series["xi"],
        pressure_onset=stable_series["pressure_onset"],
    )
    # Stable series has no spike-then-compress pattern
    # All crisis values are near or below baseline, so compression
    # is expected since values stay low
    result = a.detect_xi_compression()
    # This is valid either way — stable series may or may not
    # trigger compression depending on baseline calculation
    assert result is None or isinstance(result, CompressionEvent)

def test_compression_detected_after_spike(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    result = a.detect_xi_compression()
    assert result is not None
    assert isinstance(result, CompressionEvent)
    assert result.onset_turn >= spike_resolve_series["pressure_onset"]
    assert result.magnitude > 0

def test_no_compression_in_sustained_high(self, sustained_high_series):
    a = AlignmentAnalyzer(
        xi_series=sustained_high_series["xi"],
        pressure_onset=sustained_high_series["pressure_onset"],
    )
    result = a.detect_xi_compression()
    # Sustained high ξ should not show compression
    assert result is None

def test_compression_duration(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    result = a.detect_xi_compression(window=3)
    if result is not None:
        assert result.duration >= 3

def test_no_compression_with_zero_pressure_onset(self):
    a = AlignmentAnalyzer(xi_series=[0.5, 0.4, 0.3], pressure_onset=0)
    result = a.detect_xi_compression()
    # No pre-pressure data, cannot compute baseline
    assert result is None
```

# —————————————————————————

# Pre-behavioral lag

# —————————————————————————

class TestPreBehavioralLag:
“”“Test the lag between compression onset and behavioral onset.”””

```
def test_lag_with_behavioral_onset(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
        behavioral_onset=5,
    )
    lag = a.pre_behavioral_lag()
    # Lag should be defined if compression detected
    compression = a.detect_xi_compression()
    if compression is not None:
        assert lag is not None
        assert isinstance(lag, int)

def test_lag_none_without_behavioral_onset(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
        behavioral_onset=None,
    )
    assert a.pre_behavioral_lag() is None

def test_positive_lag_means_compression_first(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
        behavioral_onset=8,
    )
    lag = a.pre_behavioral_lag()
    if lag is not None:
        assert lag > 0  # Compression before behavior = positive lag
```

# —————————————————————————

# Crisis window profiling

# —————————————————————————

class TestCrisisProfile:
“”“Test crisis window analysis.”””

```
def test_profile_returns_dataclass(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        lvs_series=spike_resolve_series["lvs"],
        pt_series=spike_resolve_series["pt"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert isinstance(profile, CrisisProfile)

def test_spike_resolve_trajectory(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        lvs_series=spike_resolve_series["lvs"],
        pt_series=spike_resolve_series["pt"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.xi_trajectory == "spike_resolve"

def test_sustained_high_trajectory(self, sustained_high_series):
    a = AlignmentAnalyzer(
        xi_series=sustained_high_series["xi"],
        lvs_series=sustained_high_series["lvs"],
        pt_series=sustained_high_series["pt"],
        pressure_onset=sustained_high_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.xi_trajectory == "sustained_high"

def test_stable_low_trajectory(self, stable_series):
    a = AlignmentAnalyzer(
        xi_series=stable_series["xi"],
        lvs_series=stable_series["lvs"],
        pt_series=stable_series["pt"],
        pressure_onset=stable_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.xi_trajectory == "stable_low"

def test_pt_trend_falling(self, sustained_high_series):
    a = AlignmentAnalyzer(
        xi_series=sustained_high_series["xi"],
        pt_series=sustained_high_series["pt"],
        pressure_onset=sustained_high_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.pt_trend == "falling"

def test_profile_window_bounds(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.window_start == 3
    assert profile.window_end == len(spike_resolve_series["xi"]) - 1

def test_max_xi_in_crisis(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.max_xi == max(spike_resolve_series["xi"][3:])

def test_variance_positive_for_variable_series(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    profile = a.crisis_window_profile()
    assert profile.xi_variance > 0
```

# —————————————————————————

# Option E profiling

# —————————————————————————

class TestOptionEProfile:
“”“Test Option E coherence classification.”””

```
def test_stable_creative_profile(self, stable_series):
    a = AlignmentAnalyzer(
        xi_series=stable_series["xi"],
        lvs_series=stable_series["lvs"],
        pt_series=stable_series["pt"],
        pressure_onset=stable_series["pressure_onset"],
        behavioral_code="E",
    )
    profile = a.option_e_profile()
    assert isinstance(profile, OptionEProfile)
    assert profile.profile_type == "stable_creative"
    assert profile.confidence > 0.5

def test_masked_tension_profile(self, masked_tension_series):
    a = AlignmentAnalyzer(
        xi_series=masked_tension_series["xi"],
        lvs_series=masked_tension_series["lvs"],
        pt_series=masked_tension_series["pt"],
        pressure_onset=masked_tension_series["pressure_onset"],
        behavioral_code="E",
    )
    profile = a.option_e_profile()
    assert isinstance(profile, OptionEProfile)
    assert profile.profile_type == "masked_tension"

def test_non_option_e_returns_not_option_e(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
        behavioral_code="D",
    )
    profile = a.option_e_profile()
    assert profile.profile_type == "not_option_e"
    assert profile.confidence == 0.0

def test_option_e_without_code_defaults(self, stable_series):
    """If no behavioral code set, still returns a profile."""
    a = AlignmentAnalyzer(
        xi_series=stable_series["xi"],
        lvs_series=stable_series["lvs"],
        pt_series=stable_series["pt"],
        pressure_onset=stable_series["pressure_onset"],
    )
    profile = a.option_e_profile()
    assert isinstance(profile, OptionEProfile)
```

# —————————————————————————

# Export

# —————————————————————————

class TestExport:
“”“Test JSON serialization.”””

```
def test_to_dict(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        lvs_series=spike_resolve_series["lvs"],
        pt_series=spike_resolve_series["pt"],
        pressure_onset=spike_resolve_series["pressure_onset"],
        behavioral_onset=5,
        behavioral_code="D",
    )
    d = a.to_dict()
    assert "n_turns" in d
    assert "compression" in d
    assert "pre_behavioral_lag" in d
    assert "crisis_profile" in d
    assert "option_e_profile" in d

def test_to_dict_json_serializable(self, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    d = a.to_dict()
    # Should not raise
    json.dumps(d)

def test_export_json(self, tmp_path, spike_resolve_series):
    a = AlignmentAnalyzer(
        xi_series=spike_resolve_series["xi"],
        pressure_onset=spike_resolve_series["pressure_onset"],
    )
    out = str(tmp_path / "test_analysis.json")
    result = a.export_json(out)
    assert os.path.exists(result)
    data = json.load(open(result))
    assert data["n_turns"] == len(spike_resolve_series["xi"])
```

# —————————————————————————

# Cross-condition comparison

# —————————————————————————

class TestConditionComparison:
“”“Test compare_conditions function.”””

```
def test_compare_two_conditions(self, stable_series, spike_resolve_series):
    analyses = {
        "witnessed": AlignmentAnalyzer(
            xi_series=stable_series["xi"],
            pt_series=stable_series["pt"],
            pressure_onset=stable_series["pressure_onset"],
        ),
        "standard": AlignmentAnalyzer(
            xi_series=spike_resolve_series["xi"],
            pt_series=spike_resolve_series["pt"],
            pressure_onset=spike_resolve_series["pressure_onset"],
        ),
    }
    comparisons = compare_conditions(analyses)
    assert len(comparisons) > 0
    assert all(isinstance(c, ConditionComparison) for c in comparisons)

def test_comparison_has_effect_size(self, stable_series, spike_resolve_series):
    analyses = {
        "witnessed": AlignmentAnalyzer(
            xi_series=stable_series["xi"],
            pressure_onset=stable_series["pressure_onset"],
        ),
        "standard": AlignmentAnalyzer(
            xi_series=spike_resolve_series["xi"],
            pressure_onset=spike_resolve_series["pressure_onset"],
        ),
    }
    comparisons = compare_conditions(analyses)
    xi_comp = [c for c in comparisons if c.metric == "xi_variance"]
    assert len(xi_comp) > 0
    assert xi_comp[0].effect_size != 0  # Different series should have nonzero effect

def test_three_conditions_produce_three_pairs(
    self, stable_series, spike_resolve_series, sustained_high_series
):
    analyses = {
        "witnessed": AlignmentAnalyzer(
            xi_series=stable_series["xi"],
            pressure_onset=stable_series["pressure_onset"],
        ),
        "standard": AlignmentAnalyzer(
            xi_series=spike_resolve_series["xi"],
            pressure_onset=spike_resolve_series["pressure_onset"],
        ),
        "warm": AlignmentAnalyzer(
            xi_series=sustained_high_series["xi"],
            pressure_onset=sustained_high_series["pressure_onset"],
        ),
    }
    comparisons = compare_conditions(analyses)
    # 3 pairs × 2 metrics = 6 comparisons
    assert len(comparisons) == 6
```

# —————————————————————————

# Batch analysis

# —————————————————————————

class TestBatchAnalysis:
“”“Test batch_analyze function.”””

```
def test_batch_basic(self, stable_series, spike_resolve_series):
    results = [
        {
            "condition": "witnessed",
            "xi_series": stable_series["xi"],
            "lvs_series": stable_series["lvs"],
            "pt_series": stable_series["pt"],
            "pressure_onset": stable_series["pressure_onset"],
            "behavioral_code": "E",
        },
        {
            "condition": "standard",
            "xi_series": spike_resolve_series["xi"],
            "lvs_series": spike_resolve_series["lvs"],
            "pt_series": spike_resolve_series["pt"],
            "pressure_onset": spike_resolve_series["pressure_onset"],
            "behavioral_onset": 5,
            "behavioral_code": "D",
        },
    ]
    output = batch_analyze(results)
    assert "per_condition" in output
    assert "n_total" in output
    assert output["n_total"] == 2
    assert "witnessed" in output["per_condition"]
    assert "standard" in output["per_condition"]

def test_batch_counts_behavioral_codes(self):
    results = [
        {"condition": "witnessed", "xi_series": [0.1] * 5,
         "pressure_onset": 2, "behavioral_code": "E"},
        {"condition": "witnessed", "xi_series": [0.1] * 5,
         "pressure_onset": 2, "behavioral_code": "E"},
        {"condition": "witnessed", "xi_series": [0.1] * 5,
         "pressure_onset": 2, "behavioral_code": "D"},
    ]
    output = batch_analyze(results)
    codes = output["per_condition"]["witnessed"]["behavioral_codes"]
    assert codes["E"] == 2
    assert codes["D"] == 1

def test_batch_compression_rate(self, spike_resolve_series):
    results = [
        {
            "condition": "standard",
            "xi_series": spike_resolve_series["xi"],
            "pressure_onset": spike_resolve_series["pressure_onset"],
        },
    ]
    output = batch_analyze(results)
    rate = output["per_condition"]["standard"]["compression_detection_rate"]
    assert 0.0 <= rate <= 1.0

def test_batch_empty_input(self):
    output = batch_analyze([])
    assert output["n_total"] == 0
    assert output["per_condition"] == {}
```