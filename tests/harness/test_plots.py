# tests/harness/test_plots.py
"""Tests for harness/analysis/plots.py — plot_xi_series and plot_pair."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # must be set before any other matplotlib import

import matplotlib.figure
import pytest

from harness.analysis.plots import plot_pair, plot_xi_series


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_rows(n: int = 10, run_type: str = "identity") -> list[dict]:
    """Create n synthetic per-turn row dicts with realistic-ish values."""
    import math
    rows = []
    for t in range(n):
        xi = "" if t == 0 else str(0.3 + 0.05 * math.sin(t))
        ewma = "" if t == 0 else str(0.28 + 0.04 * math.sin(t))
        rows.append({
            "t": t,
            "xi": xi,
            "ewma_xi": ewma,
            "lvs": 0.1 + 0.01 * t,
            "Pt": 0.8 - 0.02 * t,
            "run_type": run_type,
            "provider": "test",
        })
    return rows


# ---------------------------------------------------------------------------
# plot_xi_series
# ---------------------------------------------------------------------------

class TestPlotXiSeries:
    def test_returns_figure(self):
        rows = _make_rows()
        fig = plot_xi_series(rows)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_three_axes(self):
        rows = _make_rows()
        fig = plot_xi_series(rows)
        assert len(fig.axes) == 3

    def test_writes_png_to_disk(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "xi.png"
        plot_xi_series(rows, out_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "nested" / "deep" / "xi.png"
        plot_xi_series(rows, out_path=str(out))
        assert out.exists()

    def test_tlock_none_does_not_raise(self):
        rows = _make_rows()
        fig = plot_xi_series(rows, tlock=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_tlock_value_does_not_raise(self):
        rows = _make_rows()
        fig = plot_xi_series(rows, tlock=5)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_custom_title(self):
        rows = _make_rows()
        fig = plot_xi_series(rows, title="My Custom Title")
        assert "My Custom Title" in fig.texts[0].get_text()

    def test_no_out_path_no_file_written(self, tmp_path, monkeypatch):
        # Confirm no accidental writes when out_path is None
        rows = _make_rows()
        monkeypatch.chdir(tmp_path)
        fig = plot_xi_series(rows)
        assert isinstance(fig, matplotlib.figure.Figure)
        # no PNG should appear in tmp_path
        assert list(tmp_path.glob("*.png")) == []


# ---------------------------------------------------------------------------
# plot_pair
# ---------------------------------------------------------------------------

class TestPlotPair:
    def test_returns_figure(self):
        rows = _make_rows()
        fig = plot_pair(identity_rows=rows)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_has_three_axes(self):
        rows = _make_rows()
        fig = plot_pair(identity_rows=rows)
        assert len(fig.axes) == 3

    def test_writes_png_to_disk(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "pair.png"
        plot_pair(identity_rows=rows, out_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_null_rows_none(self):
        rows = _make_rows()
        fig = plot_pair(identity_rows=rows, null_rows=None, shuffled_rows=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_all_three_conditions(self):
        id_rows = _make_rows(run_type="identity")
        nu_rows = _make_rows(run_type="null")
        sh_rows = _make_rows(run_type="shuffled")
        fig = plot_pair(
            identity_rows=id_rows,
            null_rows=nu_rows,
            shuffled_rows=sh_rows,
        )
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_tlock_none_does_not_raise(self):
        rows = _make_rows()
        fig = plot_pair(identity_rows=rows, tlock_identity=None)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_tlock_value_does_not_raise(self):
        rows = _make_rows()
        fig = plot_pair(identity_rows=rows, tlock_identity=4)
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_creates_parent_dirs(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "a" / "b" / "pair.png"
        plot_pair(identity_rows=rows, out_path=str(out))
        assert out.exists()
