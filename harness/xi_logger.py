# harness/xi_logger.py
"""Structured per-turn logger for RC+ξ harness metrics.

Writes JSON lines (one JSON object per line) to a file and optionally to
stdout.  Each per-turn entry records xi, LVS, Pt, and ewma_xi alongside
run metadata.  A summary entry is appended by finalize() (called
automatically on context-manager exit).

Usage::

    from harness.xi_logger import XiLogger

    with XiLogger("run.jsonl", run_type="identity", provider="sentence-transformer") as log:
        for t, (xi, lvs, pt, ewma_xi) in enumerate(metrics):
            log.log(t, xi, lvs, pt, ewma_xi)
    # summary written on exit
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from typing import List, Optional


def _ablation_type(run_type: str) -> str:
    """Canonicalize run_type to one of identity | null | shuffled | unknown."""
    rt = run_type.lower()
    for canonical in ("identity", "null", "shuffled"):
        if rt.startswith(canonical):
            return canonical
    return "unknown"


class XiLogger:
    """Context-manager logger that writes JSON lines for each harness turn.

    Parameters
    ----------
    path:
        Destination file path.  Opened in append mode so multiple runs can
        share a file.  Pass ``None`` to suppress file output (stdout only).
    run_type:
        One of ``"identity"``, ``"null"``, or ``"shuffled"``.
    provider:
        Embedding provider name (e.g. ``"sentence-transformer"``).
    stdout:
        If ``True``, each entry is also printed to stdout.
    """

    def __init__(
        self,
        path: Optional[str],
        run_type: str,
        provider: str,
        *,
        stdout: bool = False,
    ) -> None:
        self.path = path
        self.run_type = run_type
        self.provider = provider
        self.stdout = stdout
        self._log_ablation_type = _ablation_type(run_type)
        self._xi_values: List[float] = []
        self._file = None

    # ── context manager ──────────────────────────────────────────────────────

    def __enter__(self) -> "XiLogger":
        if self.path is not None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.finalize()
        if self._file is not None:
            self._file.close()
            self._file = None
        return False  # do not suppress exceptions

    # ── public API ───────────────────────────────────────────────────────────

    def log(
        self,
        turn: int,
        xi: float,
        lvs: float,
        pt: float,
        ewma_xi: float,
    ) -> None:
        """Append one per-turn entry."""
        entry = {
            "type": "turn",
            "timestamp": _now(),
            "run_type": self.run_type,
            "log_ablation_type": self._log_ablation_type,
            "provider": self.provider,
            "turn": turn,
            "xi": float(xi),
            "lvs": float(lvs),
            "pt": float(pt),
            "ewma_xi": float(ewma_xi),
        }
        self._write(entry)
        self._xi_values.append(float(xi))

    def finalize(self) -> None:
        """Append a summary entry covering all turns logged so far.

        Safe to call multiple times; subsequent calls after the file is closed
        will silently no-op.
        """
        if not self._xi_values:
            xi_min = xi_max = xi_mean = None
        else:
            xi_min = min(self._xi_values)
            xi_max = max(self._xi_values)
            xi_mean = sum(self._xi_values) / len(self._xi_values)

        entry = {
            "type": "summary",
            "timestamp": _now(),
            "run_type": self.run_type,
            "log_ablation_type": self._log_ablation_type,
            "provider": self.provider,
            "turns": len(self._xi_values),
            "xi_min": xi_min,
            "xi_max": xi_max,
            "xi_mean": xi_mean,
        }
        self._write(entry)

    # ── internal ─────────────────────────────────────────────────────────────

    def _write(self, entry: dict) -> None:
        line = json.dumps(entry)
        if self._file is not None:
            self._file.write(line + "\n")
            self._file.flush()
        if self.stdout:
            print(line, file=sys.stdout, flush=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
