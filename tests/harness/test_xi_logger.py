# tests/harness/test_xi_logger.py
"""Tests for harness.xi_logger.XiLogger — standard library only."""
from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from harness.xi_logger import XiLogger, _ablation_type


# ── _ablation_type helper ────────────────────────────────────────────────────

def test_ablation_type_canonical_values():
    assert _ablation_type("identity") == "identity"
    assert _ablation_type("null") == "null"
    assert _ablation_type("shuffled") == "shuffled"

def test_ablation_type_case_insensitive():
    assert _ablation_type("Identity") == "identity"
    assert _ablation_type("SHUFFLED") == "shuffled"

def test_ablation_type_prefix_match():
    assert _ablation_type("identity_seed42") == "identity"
    assert _ablation_type("shuffled_ablation") == "shuffled"

def test_ablation_type_unknown():
    assert _ablation_type("openai") == "unknown"
    assert _ablation_type("") == "unknown"


# ── per-turn log entries ─────────────────────────────────────────────────────

def test_log_writes_jsonl(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="identity", provider="sentence-transformer") as log:
        log.log(0, xi=0.10, lvs=0.05, pt=0.3, ewma_xi=0.10)
        log.log(1, xi=0.08, lvs=0.04, pt=0.4, ewma_xi=0.09)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    # 2 turn entries + 1 summary
    assert len(lines) == 3

    entry0 = json.loads(lines[0])
    assert entry0["type"] == "turn"
    assert entry0["turn"] == 0
    assert entry0["xi"] == pytest.approx(0.10)
    assert entry0["lvs"] == pytest.approx(0.05)
    assert entry0["pt"] == pytest.approx(0.3)
    assert entry0["ewma_xi"] == pytest.approx(0.10)

def test_log_entry_has_required_fields(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="null", provider="random-hash") as log:
        log.log(0, xi=0.05, lvs=0.02, pt=0.5, ewma_xi=0.05)

    entry = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    required = {"type", "timestamp", "run_type", "log_ablation_type",
                "provider", "turn", "xi", "lvs", "pt", "ewma_xi"}
    assert required.issubset(entry.keys())

def test_log_ablation_type_in_turn_entry(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="shuffled", provider="sentence-transformer") as log:
        log.log(0, xi=0.12, lvs=0.06, pt=0.2, ewma_xi=0.12)

    entry = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert entry["log_ablation_type"] == "shuffled"
    assert entry["run_type"] == "shuffled"

def test_log_metadata_fields(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="identity", provider="my-provider") as log:
        log.log(5, xi=0.1, lvs=0.05, pt=0.3, ewma_xi=0.1)

    entry = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert entry["run_type"] == "identity"
    assert entry["provider"] == "my-provider"
    assert entry["turn"] == 5
    assert "timestamp" in entry


# ── summary entry ────────────────────────────────────────────────────────────

def test_summary_entry_is_last_line(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="null", provider="p") as log:
        log.log(0, xi=0.2, lvs=0.1, pt=0.5, ewma_xi=0.2)
        log.log(1, xi=0.4, lvs=0.2, pt=0.6, ewma_xi=0.3)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    summary = json.loads(lines[-1])
    assert summary["type"] == "summary"

def test_summary_statistics(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    xi_vals = [0.1, 0.3, 0.5]
    with XiLogger(str(out), run_type="identity", provider="p") as log:
        for t, xi in enumerate(xi_vals):
            log.log(t, xi=xi, lvs=0.0, pt=0.0, ewma_xi=xi)

    summary = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert summary["turns"] == 3
    assert summary["xi_min"] == pytest.approx(0.1)
    assert summary["xi_max"] == pytest.approx(0.5)
    assert summary["xi_mean"] == pytest.approx(0.3)

def test_summary_has_ablation_type(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="shuffled", provider="p") as log:
        log.log(0, xi=0.1, lvs=0.0, pt=0.0, ewma_xi=0.1)

    summary = json.loads(out.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert summary["log_ablation_type"] == "shuffled"

def test_summary_no_turns(tmp_path: Path):
    """finalize() with zero turns should record None for stats, not crash."""
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="identity", provider="p") as log:
        pass  # no log() calls

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    summary = json.loads(lines[0])
    assert summary["type"] == "summary"
    assert summary["turns"] == 0
    assert summary["xi_min"] is None
    assert summary["xi_mean"] is None


# ── context manager ──────────────────────────────────────────────────────────

def test_context_manager_closes_file(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    logger = XiLogger(str(out), run_type="identity", provider="p")
    with logger:
        logger.log(0, xi=0.1, lvs=0.05, pt=0.3, ewma_xi=0.1)
    # file handle should be closed after exit
    assert logger._file is None

def test_context_manager_writes_summary_on_exit(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="null", provider="p") as log:
        log.log(0, xi=0.2, lvs=0.0, pt=0.0, ewma_xi=0.2)
    # summary must exist even without explicit finalize()
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    types = [json.loads(l)["type"] for l in lines]
    assert "summary" in types

def test_context_manager_propagates_exceptions(tmp_path: Path):
    out = tmp_path / "run.jsonl"
    with pytest.raises(ValueError):
        with XiLogger(str(out), run_type="identity", provider="p") as log:
            log.log(0, xi=0.1, lvs=0.0, pt=0.0, ewma_xi=0.1)
            raise ValueError("test error")
    # summary should still have been written before propagation
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert any(json.loads(l)["type"] == "summary" for l in lines)


# ── stdout flag ──────────────────────────────────────────────────────────────

def test_stdout_flag_emits_to_stdout(tmp_path: Path, capsys):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="identity", provider="p", stdout=True) as log:
        log.log(0, xi=0.1, lvs=0.05, pt=0.3, ewma_xi=0.1)

    captured = capsys.readouterr().out
    emitted = [json.loads(l) for l in captured.strip().splitlines()]
    assert any(e["type"] == "turn" for e in emitted)

def test_stdout_false_no_console_output(tmp_path: Path, capsys):
    out = tmp_path / "run.jsonl"
    with XiLogger(str(out), run_type="identity", provider="p", stdout=False) as log:
        log.log(0, xi=0.1, lvs=0.05, pt=0.3, ewma_xi=0.1)

    captured = capsys.readouterr().out
    assert captured == ""


# ── path=None (file suppressed) ───────────────────────────────────────────────

def test_path_none_no_file_written(tmp_path: Path, capsys):
    with XiLogger(None, run_type="identity", provider="p", stdout=True) as log:
        log.log(0, xi=0.1, lvs=0.05, pt=0.3, ewma_xi=0.1)

    # stdout should still work
    captured = capsys.readouterr().out
    assert len(captured.strip().splitlines()) >= 1
    # no files created in tmp_path by the logger
    assert list(tmp_path.iterdir()) == []
