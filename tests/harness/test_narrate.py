# tests/harness/test_narrate.py
"""Tests for harness/analysis/narrate.py."""
from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from harness.analysis.narrate import (
    _label_xi,
    _pt_slope,
    _xi_trend,
    compute_findings,
    narrate,
    render_report,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _write_csv(path, n: int = 10, xi_base: float = 0.4, run_type: str = "identity"):
    """Write a minimal per-turn CSV to path."""
    rows = []
    for t in range(n):
        xi = "" if t == 0 else str(xi_base + 0.01 * t)
        ewma = "" if t == 0 else str(xi_base + 0.005 * t)
        rows.append({
            "t": t,
            "xi": xi,
            "ewma_xi": ewma,
            "lvs": str(0.1 - 0.005 * t),
            "Pt": str(0.8 - 0.02 * t),
            "run_type": run_type,
            "provider": "test",
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["t", "xi", "ewma_xi", "lvs", "Pt", "run_type", "provider"])
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(path, run_type: str = "identity", tlock=None, e1: float = 0.4):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "E1_median_xi_last10": e1,
            "Tlock": tlock,
            "k": 5, "m": 5,
            "eps_xi": 0.02, "eps_lvs": 0.015,
            "provider": "test",
            "run_type": run_type,
        }, f)


def _write_eval(path, e1_pass: bool = False, e3_pass: bool = False):
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "E1_identity_median_xi_last10": 0.4,
            "E1_null_median_xi_last10": 0.5,
            "mann_whitney_U": 30.0,
            "mann_whitney_p": 0.3,
            "cliffs_delta_null_vs_identity": 0.2,
            "Pt_trend_identity": 0.05,
            "Pt_trend_null": -0.02,
            "E1_pass": e1_pass,
            "E3_pass": e3_pass,
        }, f)


def _make_series(n: int = 10, xi_val: float = 0.4) -> dict:
    xi = np.array([float("nan")] + [xi_val + 0.01 * i for i in range(n - 1)])
    return {
        "t": np.arange(n, dtype=float),
        "xi": xi,
        "lvs": np.linspace(0.1, 0.05, n),
        "Pt": np.linspace(0.8, 0.6, n),
        "ewma_xi": xi * 0.9,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests — helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelXi:
    def test_very_high(self):
        assert _label_xi(0.7) == "very high"

    def test_high(self):
        assert _label_xi(0.5) == "high"

    def test_moderate(self):
        assert _label_xi(0.3) == "moderate"

    def test_low(self):
        assert _label_xi(0.1) == "low"

    def test_very_low(self):
        assert _label_xi(0.01) == "very low"

    def test_nan_returns_unknown(self):
        assert _label_xi(float("nan")) == "unknown"


class TestXiTrend:
    def test_rising(self):
        xi = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        assert _xi_trend(xi) == "rising"

    def test_falling(self):
        xi = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
        assert _xi_trend(xi) == "falling"

    def test_stable(self):
        xi = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
        assert _xi_trend(xi) == "stable"

    def test_volatile(self):
        xi = np.array([0.1, 0.9, 0.1, 0.9, 0.1, 0.9])
        assert _xi_trend(xi) == "volatile"

    def test_insufficient_data(self):
        xi = np.array([0.3, 0.4])
        assert _xi_trend(xi) == "insufficient data"

    def test_handles_nan(self):
        xi = np.array([float("nan"), 0.3, 0.4, 0.5, 0.6])
        result = _xi_trend(xi)
        assert result in ("rising", "falling", "stable", "volatile", "insufficient data")


class TestPtSlope:
    def test_rising_slope_positive(self):
        pt = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        assert _pt_slope(pt) > 0

    def test_falling_slope_negative(self):
        pt = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        assert _pt_slope(pt) < 0

    def test_flat_slope_near_zero(self):
        pt = np.array([0.7, 0.7, 0.7, 0.7])
        assert abs(_pt_slope(pt)) < 1e-6

    def test_single_point_returns_zero(self):
        pt = np.array([0.5])
        assert _pt_slope(pt) == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# compute_findings
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFindings:
    def test_identity_only(self):
        series = _make_series()
        f = compute_findings(identity_series=series)
        assert "identity" in f
        assert "verdict" in f
        assert "null" not in f
        assert "comparison" not in f
        assert "shuffled" not in f

    def test_identity_plus_null(self):
        id_s = _make_series(xi_val=0.3)
        nu_s = _make_series(xi_val=0.5)
        f = compute_findings(identity_series=id_s, null_series=nu_s)
        assert "null" in f
        assert "comparison" in f

    def test_all_three_conditions(self):
        id_s = _make_series(xi_val=0.3)
        nu_s = _make_series(xi_val=0.5)
        sh_s = _make_series(xi_val=0.6)
        f = compute_findings(identity_series=id_s, null_series=nu_s, shuffled_series=sh_s)
        assert "shuffled" in f

    def test_verdict_is_string(self):
        f = compute_findings(identity_series=_make_series())
        assert isinstance(f["verdict"], str)

    def test_tlock_from_summary(self):
        series = _make_series()
        summary = {"Tlock": 7, "E1_median_xi_last10": 0.3}
        f = compute_findings(identity_series=series, identity_summary=summary)
        assert f["identity"]["Tlock"] == 7
        assert f["verdict"] == "strong_stabilization"

    def test_high_xi_no_lock_verdict(self):
        series = _make_series(xi_val=0.7)
        f = compute_findings(identity_series=series)
        assert f["verdict"] == "high_tension_no_lock"

    def test_no_nan_in_output(self):
        """All numeric values in findings must be Python float/int/None — no NaN."""
        series = _make_series()
        nu = _make_series(xi_val=0.5)
        f = compute_findings(identity_series=series, null_series=nu)
        raw = json.dumps(f)  # would raise if numpy types slip through
        assert "NaN" not in raw

    def test_comparison_has_expected_keys(self):
        id_s = _make_series(xi_val=0.3)
        nu_s = _make_series(xi_val=0.5)
        f = compute_findings(identity_series=id_s, null_series=nu_s)
        comp = f["comparison"]
        for key in ("xi_delta_id_minus_null", "mann_whitney_p", "cliffs_delta_null_vs_identity",
                    "E1_pass", "E3_pass"):
            assert key in comp, f"Missing key: {key}"


# ─────────────────────────────────────────────────────────────────────────────
# render_report
# ─────────────────────────────────────────────────────────────────────────────

class TestRenderReport:
    def _findings(self, **kwargs):
        series = _make_series(**kwargs)
        return compute_findings(identity_series=series)

    def test_returns_string(self):
        assert isinstance(render_report(self._findings()), str)

    def test_contains_verdict_header(self):
        report = render_report(self._findings())
        assert "## Overall Verdict" in report

    def test_contains_identity_section(self):
        report = render_report(self._findings())
        assert "## Identity Run" in report

    def test_contains_interpretation_section(self):
        report = render_report(self._findings())
        assert "## Interpretation" in report

    def test_null_section_present_when_null_included(self):
        id_s = _make_series(xi_val=0.3)
        nu_s = _make_series(xi_val=0.5)
        f = compute_findings(identity_series=id_s, null_series=nu_s)
        report = render_report(f)
        assert "## Null Run" in report
        assert "## Identity vs Null Comparison" in report

    def test_shuffled_section_present_when_shuffled_included(self):
        id_s = _make_series()
        nu_s = _make_series(xi_val=0.5)
        sh_s = _make_series(xi_val=0.6)
        f = compute_findings(identity_series=id_s, null_series=nu_s, shuffled_series=sh_s)
        report = render_report(f)
        assert "## Shuffled Control" in report

    def test_strong_stabilization_label_in_report(self):
        series = _make_series()
        summary = {"Tlock": 5, "E1_median_xi_last10": 0.01}
        f = compute_findings(identity_series=series, identity_summary=summary)
        report = render_report(f)
        assert "Strong Identity Stabilization" in report


# ─────────────────────────────────────────────────────────────────────────────
# narrate() end-to-end (file I/O)
# ─────────────────────────────────────────────────────────────────────────────

class TestNarrateEndToEnd:
    def test_identity_only(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        _write_csv(id_csv, run_type="identity")
        report = narrate(identity_csv=str(id_csv))
        assert isinstance(report, str)
        assert len(report) > 100

    def test_with_null_and_summaries(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        nu_csv = tmp_path / "nu.csv"
        id_json = tmp_path / "id.json"
        nu_json = tmp_path / "nu.json"
        _write_csv(id_csv, xi_base=0.3, run_type="identity")
        _write_csv(nu_csv, xi_base=0.5, run_type="null")
        _write_summary(id_json, run_type="identity", e1=0.3)
        _write_summary(nu_json, run_type="null", e1=0.5)
        report = narrate(
            identity_csv=str(id_csv),
            null_csv=str(nu_csv),
            identity_json=str(id_json),
            null_json=str(nu_json),
        )
        assert "## Null Run" in report
        assert "## Identity vs Null Comparison" in report

    def test_with_eval_json(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        nu_csv = tmp_path / "nu.csv"
        ev_json = tmp_path / "eval.json"
        _write_csv(id_csv, xi_base=0.3)
        _write_csv(nu_csv, xi_base=0.5)
        _write_eval(ev_json, e1_pass=True, e3_pass=False)
        report = narrate(
            identity_csv=str(id_csv),
            null_csv=str(nu_csv),
            eval_json=str(ev_json),
        )
        assert "E1" in report

    def test_with_shuffled(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        nu_csv = tmp_path / "nu.csv"
        sh_csv = tmp_path / "sh.csv"
        _write_csv(id_csv, xi_base=0.3)
        _write_csv(nu_csv, xi_base=0.5)
        _write_csv(sh_csv, xi_base=0.6)
        report = narrate(
            identity_csv=str(id_csv),
            null_csv=str(nu_csv),
            shuffled_csv=str(sh_csv),
        )
        assert "## Shuffled Control" in report

    def test_report_starts_with_title(self, tmp_path):
        """narrate() returns a string starting with the expected markdown heading."""
        id_csv = tmp_path / "id.csv"
        _write_csv(id_csv)
        report = narrate(identity_csv=str(id_csv))
        assert report.startswith("# RC+ξ Analysis Report")

    def test_use_claude_false_does_not_call_api(self, tmp_path, monkeypatch):
        """Confirm no API call when use_claude=False (even if key is set)."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-key-should-not-be-called")
        id_csv = tmp_path / "id.csv"
        _write_csv(id_csv)
        # If it tried to call the API it would fail with an auth error — but it shouldn't.
        report = narrate(identity_csv=str(id_csv), use_claude=False)
        assert "Claude Narrative" not in report

    def test_use_claude_without_key_adds_skip_note(self, tmp_path, monkeypatch):
        """With use_claude=True but no key, report gets a skip note instead of crashing."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        id_csv = tmp_path / "id.csv"
        _write_csv(id_csv)
        report = narrate(identity_csv=str(id_csv), use_claude=True)
        assert "ANTHROPIC_API_KEY" in report or "Claude narrative" in report.lower()

    def test_tlock_in_summary_surfaces_in_report(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        id_json = tmp_path / "id.json"
        _write_csv(id_csv)
        _write_summary(id_json, tlock=6, e1=0.01)
        report = narrate(identity_csv=str(id_csv), identity_json=str(id_json))
        assert "turn 6" in report or "Lock at turn 6" in report

    def test_report_is_valid_markdown(self, tmp_path):
        id_csv = tmp_path / "id.csv"
        _write_csv(id_csv)
        report = narrate(identity_csv=str(id_csv))
        assert report.count("##") >= 2
        assert "|" in report  # table rows
