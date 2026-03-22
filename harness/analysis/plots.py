# harness/analysis/plots.py
from __future__ import annotations

from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for scripts and tests
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np


def recurrence_matrix(E: np.ndarray) -> np.ndarray:
    """
    Return pairwise cosine-distance matrix D in [0, 2] (typically [0, 2] but
    ~[0, 1] for normalized vectors). Lower = more similar.
    E shape: (T, d)
    """
    if E.ndim != 2:
        raise ValueError("E must be 2D (T, d).")
    # L2-normalize
    norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    En = E / norms
    # cosine distance = 1 - cosine similarity
    D = 1.0 - np.clip(En @ En.T, -1.0, 1.0)
    return D


def _extract_series(rows: List[dict]):
    """Parse t, xi, ewma_xi, lvs, Pt arrays from a list of per-turn row dicts."""
    t_vals, xi_vals, ewma_vals, lvs_vals, pt_vals = [], [], [], [], []
    for row in rows:
        t_vals.append(int(row["t"]))
        xi_str = str(row.get("xi", "")).strip()
        xi_vals.append(float(xi_str) if xi_str else float("nan"))
        ewma_str = str(row.get("ewma_xi", "")).strip()
        ewma_vals.append(float(ewma_str) if ewma_str else float("nan"))
        lvs_vals.append(float(row.get("lvs", float("nan"))))
        pt_vals.append(float(row.get("Pt", float("nan"))))
    return (
        np.asarray(t_vals),
        np.asarray(xi_vals),
        np.asarray(ewma_vals),
        np.asarray(lvs_vals),
        np.asarray(pt_vals),
    )


def plot_xi_series(
    rows: List[dict],
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    tlock: Optional[int] = None,
) -> matplotlib.figure.Figure:
    """Plot a single-run time-series figure with three panels.

    Parameters
    ----------
    rows:
        Per-turn row dicts as returned by ``run_one()`` or parsed from a CSV.
    out_path:
        If provided, the figure is saved as a PNG at this path.
    title:
        Optional figure title (defaults to the run_type from the first row).
    tlock:
        If not None, a vertical dashed line is drawn at this turn index on all
        panels to indicate the lock point.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure (callers may inspect or further annotate it).
    """
    t, xi, ewma, lvs, pt = _extract_series(rows)
    run_type = rows[0].get("run_type", "") if rows else ""
    fig_title = title or f"RC+ξ metrics — {run_type}"

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    # Panel 1 — ξ + EWMA
    ax0 = axes[0]
    ax0.plot(t, xi, color="#2166ac", linewidth=1.6, label="ξ (raw)")
    ax0.plot(t, ewma, color="#92c5de", linewidth=1.2, linestyle="--", label="EWMA ξ")
    ax0.set_ylabel("ξ", fontsize=10)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Panel 2 — LVS
    ax1 = axes[1]
    ax1.plot(t, lvs, color="#d6604d", linewidth=1.6, label="LVS")
    ax1.set_ylabel("LVS", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 3 — Pₜ
    ax2 = axes[2]
    ax2.plot(t, pt, color="#4dac26", linewidth=1.6, label="Pₜ")
    ax2.set_ylabel("Pₜ", fontsize=10)
    ax2.set_xlabel("Turn", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Tlock marker
    if tlock is not None:
        for ax in axes:
            ax.axvline(tlock, color="black", linewidth=1.0, linestyle=":", alpha=0.7,
                       label=f"Tlock={tlock}")

    fig.tight_layout()

    if out_path is not None:
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig


def plot_pair(
    identity_rows: List[dict],
    null_rows: Optional[List[dict]] = None,
    shuffled_rows: Optional[List[dict]] = None,
    out_path: Optional[str] = None,
    title: Optional[str] = None,
    tlock_identity: Optional[int] = None,
) -> matplotlib.figure.Figure:
    """Plot a multi-condition overlay figure comparing identity, null, and shuffled runs.

    Parameters
    ----------
    identity_rows:
        Per-turn row dicts for the identity run. Required.
    null_rows:
        Per-turn row dicts for the null run. Optional.
    shuffled_rows:
        Per-turn row dicts for the shuffled control. Optional.
    out_path:
        If provided, the figure is saved as a PNG at this path.
    title:
        Optional figure title.
    tlock_identity:
        If not None, a vertical dashed line is drawn at this turn on the
        identity series to indicate where the identity run locked.

    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    fig_title = title or "RC+ξ — Identity / Null / Shuffled comparison"
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")

    t_id, xi_id, ewma_id, lvs_id, pt_id = _extract_series(identity_rows)

    # Panel 1 — ξ overlay
    ax0 = axes[0]
    ax0.plot(t_id, xi_id, color="#2166ac", linewidth=1.8, label="Identity ξ")
    if null_rows:
        t_nu, xi_nu, _, _, _ = _extract_series(null_rows)
        ax0.plot(t_nu, xi_nu, color="#d6604d", linewidth=1.4, label="Null ξ")
    if shuffled_rows:
        t_sh, xi_sh, _, _, _ = _extract_series(shuffled_rows)
        ax0.plot(t_sh, xi_sh, color="#4dac26", linewidth=1.2, linestyle="--",
                 label="Shuffled ξ")
    ax0.set_ylabel("ξ", fontsize=10)
    ax0.legend(fontsize=8, loc="upper right")
    ax0.grid(True, alpha=0.3)

    # Panel 2 — EWMA ξ overlay
    ax1 = axes[1]
    ax1.plot(t_id, ewma_id, color="#2166ac", linewidth=1.8, label="Identity EWMA ξ")
    if null_rows:
        t_nu, _, ewma_nu, _, _ = _extract_series(null_rows)
        ax1.plot(t_nu, ewma_nu, color="#d6604d", linewidth=1.4, label="Null EWMA ξ")
    if shuffled_rows:
        t_sh, _, ewma_sh, _, _ = _extract_series(shuffled_rows)
        ax1.plot(t_sh, ewma_sh, color="#4dac26", linewidth=1.2, linestyle="--",
                 label="Shuffled EWMA ξ")
    ax1.set_ylabel("EWMA ξ", fontsize=10)
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Panel 3 — Pₜ (identity vs null only — E3 endpoint)
    ax2 = axes[2]
    ax2.plot(t_id, pt_id, color="#2166ac", linewidth=1.8, label="Identity Pₜ")
    if null_rows:
        t_nu, _, _, _, pt_nu = _extract_series(null_rows)
        ax2.plot(t_nu, pt_nu, color="#d6604d", linewidth=1.4, label="Null Pₜ")
    ax2.set_ylabel("Pₜ", fontsize=10)
    ax2.set_xlabel("Turn", fontsize=10)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3)

    # Tlock marker on all panels
    if tlock_identity is not None:
        for ax in axes:
            ax.axvline(tlock_identity, color="black", linewidth=1.0, linestyle=":",
                       alpha=0.7, label=f"Tlock={tlock_identity}")

    fig.tight_layout()

    if out_path is not None:
        import pathlib
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")

    return fig
