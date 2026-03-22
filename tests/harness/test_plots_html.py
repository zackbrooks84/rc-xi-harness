# tests/harness/test_plots_html.py
"""Tests for the Plotly HTML output functions in harness/analysis/plots.py."""
from __future__ import annotations

import math

import pytest

from harness.analysis.plots import plot_pair_html, plot_xi_series_html


# ---------------------------------------------------------------------------
# Synthetic data helpers (mirrors test_plots.py)
# ---------------------------------------------------------------------------

def _make_rows(n: int = 10, run_type: str = "identity") -> list[dict]:
    """Create n synthetic per-turn row dicts with realistic-ish values."""
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
# plot_xi_series_html
# ---------------------------------------------------------------------------

class TestPlotXiSeriesHtml:
    def test_returns_nonempty_string(self):
        rows = _make_rows()
        html = plot_xi_series_html(rows)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_writes_file_to_disk(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "xi.html"
        html = plot_xi_series_html(rows, out_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
        assert html == out.read_text(encoding="utf-8")

    def test_creates_parent_dirs(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "nested" / "deep" / "xi.html"
        plot_xi_series_html(rows, out_path=str(out))
        assert out.exists()

    def test_tlock_none_does_not_raise(self):
        rows = _make_rows()
        html = plot_xi_series_html(rows, tlock=None)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_tlock_value_does_not_raise(self):
        rows = _make_rows()
        html = plot_xi_series_html(rows, tlock=5)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_custom_title_appears_in_output(self):
        rows = _make_rows()
        html = plot_xi_series_html(rows, title="My Custom Title")
        assert "My Custom Title" in html

    def test_no_out_path_no_file_written(self, tmp_path, monkeypatch):
        rows = _make_rows()
        monkeypatch.chdir(tmp_path)
        html = plot_xi_series_html(rows)
        assert isinstance(html, str)
        assert list(tmp_path.glob("*.html")) == []

    def test_uses_cdn_plotlyjs(self):
        rows = _make_rows()
        html = plot_xi_series_html(rows)
        assert "cdn.plot.ly" in html or "plotly" in html.lower()


# ---------------------------------------------------------------------------
# plot_pair_html
# ---------------------------------------------------------------------------

class TestPlotPairHtml:
    def test_returns_nonempty_string(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_writes_file_to_disk(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "pair.html"
        html = plot_pair_html(identity_rows=rows, out_path=str(out))
        assert out.exists()
        assert out.stat().st_size > 0
        assert html == out.read_text(encoding="utf-8")

    def test_creates_parent_dirs(self, tmp_path):
        rows = _make_rows()
        out = tmp_path / "a" / "b" / "pair.html"
        plot_pair_html(identity_rows=rows, out_path=str(out))
        assert out.exists()

    def test_null_rows_none(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows, null_rows=None, shuffled_rows=None)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_with_all_three_conditions(self):
        id_rows = _make_rows(run_type="identity")
        nu_rows = _make_rows(run_type="null")
        sh_rows = _make_rows(run_type="shuffled")
        html = plot_pair_html(
            identity_rows=id_rows,
            null_rows=nu_rows,
            shuffled_rows=sh_rows,
        )
        assert isinstance(html, str)
        assert len(html) > 0

    def test_tlock_none_does_not_raise(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows, tlock_identity=None)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_tlock_value_does_not_raise(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows, tlock_identity=4)
        assert isinstance(html, str)
        assert len(html) > 0

    def test_custom_title_appears_in_output(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows, title="Pair Custom Title")
        assert "Pair Custom Title" in html

    def test_no_out_path_no_file_written(self, tmp_path, monkeypatch):
        rows = _make_rows()
        monkeypatch.chdir(tmp_path)
        html = plot_pair_html(identity_rows=rows)
        assert isinstance(html, str)
        assert list(tmp_path.glob("*.html")) == []

    def test_uses_cdn_plotlyjs(self):
        rows = _make_rows()
        html = plot_pair_html(identity_rows=rows)
        assert "cdn.plot.ly" in html or "plotly" in html.lower()
