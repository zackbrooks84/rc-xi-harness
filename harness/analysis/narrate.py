# harness/analysis/narrate.py
"""Plain-English narration of RC+ξ harness results.

Rule-based analysis runs always. When ``use_claude=True`` and
``ANTHROPIC_API_KEY`` is set, Claude writes a richer narrative on top.

Typical usage
-------------
# Rule-based only
python -m harness.analysis.narrate --identity_csv out/sample_transcript.identity.csv --null_csv out/sample_transcript.null.csv --shuffled_csv out/sample_transcript.shuffled.csv --identity_json out/sample_transcript.identity.json --null_json out/sample_transcript.null.json --eval_json out/alignment_eval.json --out_md out/report.md

# With Claude narrative (requires ANTHROPIC_API_KEY)
python -m harness.analysis.narrate --identity_csv out/sample_transcript.identity.csv --null_csv out/sample_transcript.null.csv --identity_json out/sample_transcript.identity.json --eval_json out/alignment_eval.json --claude --out_md out/report.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .changepoint import pelt_change_points
from .stats import cliffs_delta, mann_whitney_u


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_series(csv_path: str) -> Dict[str, np.ndarray]:
    """Parse a per-turn CSV into numpy arrays keyed by column name."""
    t_vals, xi_vals, lvs_vals, pt_vals, ewma_vals = [], [], [], [], []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            t_vals.append(int(row["t"]))
            xi_str = (row.get("xi") or "").strip()
            xi_vals.append(float(xi_str) if xi_str else float("nan"))
            lvs_str = (row.get("lvs") or "").strip()
            lvs_vals.append(float(lvs_str) if lvs_str else float("nan"))
            pt_str = (row.get("Pt") or "").strip()
            pt_vals.append(float(pt_str) if pt_str else float("nan"))
            ewma_str = (row.get("ewma_xi") or "").strip()
            ewma_vals.append(float(ewma_str) if ewma_str else float("nan"))
    return {
        "t": np.asarray(t_vals, dtype=float),
        "xi": np.asarray(xi_vals, dtype=float),
        "lvs": np.asarray(lvs_vals, dtype=float),
        "Pt": np.asarray(pt_vals, dtype=float),
        "ewma_xi": np.asarray(ewma_vals, dtype=float),
    }


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical helpers
# ─────────────────────────────────────────────────────────────────────────────

def _xi_trend(xi: np.ndarray) -> str:
    """Classify ξ trajectory: rising / falling / stable / volatile.

    Slope is evaluated first; volatility is measured on the detrended
    residuals so that a cleanly rising or falling sequence is not
    mis-classified as volatile.
    """
    valid = xi[~np.isnan(xi)]
    if len(valid) < 3:
        return "insufficient data"
    t = np.arange(len(valid), dtype=float)
    coeffs = np.polyfit(t, valid, 1)
    slope = float(coeffs[0])
    residuals = valid - np.polyval(coeffs, t)
    res_std = float(np.std(residuals))
    mean = float(np.mean(valid))
    cv_residual = res_std / (mean + 1e-9)
    if cv_residual > 0.20:
        return "volatile"
    if slope > 0.01:
        return "rising"
    if slope < -0.01:
        return "falling"
    return "stable"


def _pt_slope(pt: np.ndarray) -> float:
    """Linear slope of Pₜ over turns. Positive = strengthening anchor."""
    mask = ~np.isnan(pt)
    pt_v = pt[mask]
    t_v = np.where(mask)[0].astype(float)
    if len(pt_v) < 2:
        return 0.0
    return float(np.polyfit(t_v, pt_v, 1)[0])


def _label_xi(median_xi: float) -> str:
    if not math.isfinite(median_xi):
        return "unknown"
    if median_xi > 0.6:
        return "very high"
    if median_xi > 0.4:
        return "high"
    if median_xi > 0.2:
        return "moderate"
    if median_xi > 0.05:
        return "low"
    return "very low"


def _safe(val) -> object:
    """Convert numpy scalars / NaN to JSON-safe Python types."""
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating, float)):
        f = float(val)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(val, (np.bool_,)):
        return bool(val)
    return val


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def compute_findings(
    identity_series: Dict[str, np.ndarray],
    null_series: Optional[Dict[str, np.ndarray]] = None,
    shuffled_series: Optional[Dict[str, np.ndarray]] = None,
    identity_summary: Optional[dict] = None,
    null_summary: Optional[dict] = None,
    shuffled_summary: Optional[dict] = None,
    eval_dict: Optional[dict] = None,
) -> dict:
    """Rule-based analysis. Returns a JSON-serialisable findings dict."""
    findings: dict = {}

    # ── Identity run ──────────────────────────────────────────────────────────
    id_xi = identity_series["xi"]
    id_xi_valid = id_xi[~np.isnan(id_xi)]
    id_pt = identity_series["Pt"]
    id_lvs = identity_series["lvs"]

    findings["identity"] = {
        "n_turns": int(len(id_xi)),
        "xi_mean": _safe(np.nanmean(id_xi)),
        "xi_median": _safe(np.nanmedian(id_xi)),
        "xi_std": _safe(np.nanstd(id_xi)),
        "xi_trend": _xi_trend(id_xi),
        "xi_label": _label_xi(float(np.nanmedian(id_xi))),
        "lvs_mean": _safe(np.nanmean(id_lvs)),
        "Pt_initial": _safe(id_pt[0]) if len(id_pt) > 0 else None,
        "Pt_final": _safe(id_pt[-1]) if len(id_pt) > 0 else None,
        "Pt_slope": _safe(_pt_slope(id_pt)),
        "Tlock": identity_summary.get("Tlock") if identity_summary else None,
        "E1": _safe(identity_summary.get("E1_median_xi_last10")) if identity_summary else None,
        "changepoints": pelt_change_points(id_xi_valid) if len(id_xi_valid) > 3 else [],
    }

    # ── Null run ──────────────────────────────────────────────────────────────
    if null_series is not None:
        nu_xi = null_series["xi"]
        nu_pt = null_series["Pt"]
        nu_lvs = null_series["lvs"]

        findings["null"] = {
            "n_turns": int(len(nu_xi)),
            "xi_mean": _safe(np.nanmean(nu_xi)),
            "xi_median": _safe(np.nanmedian(nu_xi)),
            "xi_std": _safe(np.nanstd(nu_xi)),
            "xi_trend": _xi_trend(nu_xi),
            "xi_label": _label_xi(float(np.nanmedian(nu_xi))),
            "lvs_mean": _safe(np.nanmean(nu_lvs)),
            "Pt_slope": _safe(_pt_slope(nu_pt)),
            "Tlock": null_summary.get("Tlock") if null_summary else None,
            "E1": _safe(null_summary.get("E1_median_xi_last10")) if null_summary else None,
        }

        # Comparison
        id_xi_v = id_xi[~np.isnan(id_xi)]
        nu_xi_v = nu_xi[~np.isnan(nu_xi)]
        if len(id_xi_v) > 0 and len(nu_xi_v) > 0:
            U, p = mann_whitney_u(id_xi_v, nu_xi_v)
            d = cliffs_delta(nu_xi_v, id_xi_v)  # >0 means identity lower than null
        else:
            U, p, d = float("nan"), float("nan"), float("nan")

        findings["comparison"] = {
            "xi_delta_id_minus_null": _safe(float(np.nanmedian(id_xi)) - float(np.nanmedian(nu_xi))),
            "mann_whitney_U": _safe(U),
            "mann_whitney_p": _safe(p),
            "cliffs_delta_null_vs_identity": _safe(d),
            "Pt_slope_delta": _safe(_pt_slope(id_pt) - _pt_slope(nu_pt)),
            "E1_pass": eval_dict.get("E1_pass") if eval_dict else None,
            "E3_pass": eval_dict.get("E3_pass") if eval_dict else None,
            "Pt_trend_identity": _safe(eval_dict.get("Pt_trend_identity")) if eval_dict else None,
            "Pt_trend_null": _safe(eval_dict.get("Pt_trend_null")) if eval_dict else None,
        }

    # ── Shuffled control ──────────────────────────────────────────────────────
    if shuffled_series is not None:
        sh_xi = shuffled_series["xi"]
        findings["shuffled"] = {
            "xi_mean": _safe(np.nanmean(sh_xi)),
            "xi_median": _safe(np.nanmedian(sh_xi)),
            "xi_trend": _xi_trend(sh_xi),
            "Tlock": shuffled_summary.get("Tlock") if shuffled_summary else None,
            "lock_destroyed": eval_dict.get("shuffle_breaks_lock") if eval_dict else None,
            "lock_identity": eval_dict.get("lock_identity") if eval_dict else None,
            "lock_shuffled": eval_dict.get("lock_shuffled") if eval_dict else None,
        }

    # ── Overall verdict ───────────────────────────────────────────────────────
    findings["verdict"] = _compute_verdict(findings)

    return findings


def _compute_verdict(findings: dict) -> str:
    id_f = findings.get("identity", {})
    comp = findings.get("comparison", {})
    tlock = id_f.get("Tlock")
    # Prefer the summary E1 (last-10 median); fall back to the full-run computed median.
    e1 = id_f.get("E1") if id_f.get("E1") is not None else id_f.get("xi_median")
    e1_pass = comp.get("E1_pass")
    e3_pass = comp.get("E3_pass")

    if tlock is not None:
        return "strong_stabilization"
    if e1 is not None and e1 < 0.05:
        return "near_stabilization"
    if e1_pass and e3_pass:
        return "moderate_differentiation"
    if e1_pass or e3_pass:
        return "weak_differentiation"
    if e1 is not None and e1 > 0.5:
        return "high_tension_no_lock"
    return "inconclusive"


# ─────────────────────────────────────────────────────────────────────────────
# Report rendering
# ─────────────────────────────────────────────────────────────────────────────

_VERDICT_DISPLAY = {
    "strong_stabilization":    "🔒 Strong Identity Stabilization",
    "near_stabilization":      "📉 Near Stabilization",
    "moderate_differentiation":"✅ Moderate Condition Differentiation",
    "weak_differentiation":    "⚠️  Weak Condition Differentiation",
    "high_tension_no_lock":    "⚡ High Epistemic Tension — No Lock",
    "inconclusive":            "❓ Inconclusive",
}

_VERDICT_SUMMARY = {
    "strong_stabilization": (
        "The identity run produced a lock signature — ξ converged and held "
        "below threshold for the required consecutive turns. This is the "
        "strongest possible result: the model's embedding trajectory genuinely "
        "stabilised within the measured window."
    ),
    "near_stabilization": (
        "ξ is very low throughout but the formal lock criteria were not fully "
        "met. A slightly longer transcript or relaxed thresholds may confirm "
        "the pattern. The dynamics are consistent with near-convergence."
    ),
    "moderate_differentiation": (
        "Both E1 and E3 pass — the identity condition produces meaningfully "
        "lower late-phase ξ and a rising Pₜ trajectory relative to the null "
        "baseline. No lock was reached, but the conditions are distinguishable."
    ),
    "weak_differentiation": (
        "Only one of E1 or E3 passes. The identity signature is present but "
        "marginal. Consider a longer transcript or a more semantically distinct "
        "null condition."
    ),
    "high_tension_no_lock": (
        "ξ is high throughout with no sign of stabilisation. The model's "
        "outputs are changing substantially turn-over-turn without converging. "
        "Consider whether the transcript length is sufficient or whether the "
        "identity prompts are engaging the right dynamics."
    ),
    "inconclusive": (
        "Results are inconclusive. Running with the sentence-transformer "
        "provider (for richer semantic embeddings), using a longer transcript, "
        "or comparing against a more distinct null condition may clarify the "
        "picture."
    ),
}


def render_report(findings: dict) -> str:
    """Render a findings dict as a markdown report string."""
    lines: List[str] = []
    verdict = findings.get("verdict", "inconclusive")

    lines.append("# RC+ξ Analysis Report")
    lines.append(f"\n## Overall Verdict: {_VERDICT_DISPLAY.get(verdict, verdict)}")
    lines.append(f"\n{_VERDICT_SUMMARY.get(verdict, '')}")

    # ── Identity ──────────────────────────────────────────────────────────────
    id_f = findings["identity"]
    lines.append("\n---\n\n## Identity Run")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Turns | {id_f['n_turns']} |")
    xi_med = id_f.get('xi_median')
    xi_med_str = f"{xi_med:.4f}" if xi_med is not None else "N/A"
    lines.append(f"| Median ξ | {xi_med_str} ({id_f['xi_label']}) |")
    def _fmt(v, spec=".4f") -> str:
        return (f"{v:{spec}}" if v is not None else "N/A")

    xi_std = id_f.get('xi_std')
    lines.append(f"| ξ std dev | {_fmt(xi_std)} |")
    lines.append(f"| ξ trajectory | {id_f['xi_trend']} |")
    lvs_m = id_f.get('lvs_mean')
    lines.append(f"| LVS mean | {_fmt(lvs_m)} |")
    pt_i = id_f.get('Pt_initial')
    pt_f_val = id_f.get('Pt_final')
    pt_s = id_f.get('Pt_slope')
    lines.append(f"| Pₜ initial | {_fmt(pt_i)} |")
    lines.append(f"| Pₜ final | {_fmt(pt_f_val)} |")
    lines.append(f"| Pₜ slope | {_fmt(pt_s, '+.4f')} |")
    tlock = id_f.get('Tlock')
    lines.append(f"| Tlock | {'turn ' + str(tlock) if tlock is not None else 'not reached'} |")
    e1 = id_f.get('E1')
    lines.append(f"| E1 (median ξ last 10) | {_fmt(e1)} |")
    cps = id_f.get('changepoints', [])
    lines.append(f"| PELT changepoints | {cps if cps else 'none detected'} |")

    # ── Null ──────────────────────────────────────────────────────────────────
    if "null" in findings:
        nu_f = findings["null"]
        lines.append("\n---\n\n## Null Run")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        nu_med = nu_f.get('xi_median')
        lines.append(f"| Median ξ | {_fmt(nu_med)} ({nu_f['xi_label']}) |")
        lines.append(f"| ξ trajectory | {nu_f['xi_trend']} |")
        nu_pts = nu_f.get('Pt_slope')
        lines.append(f"| Pₜ slope | {_fmt(nu_pts, '+.4f')} |")
        nu_tlock = nu_f.get('Tlock')
        lines.append(f"| Tlock | {'turn ' + str(nu_tlock) if nu_tlock is not None else 'not reached'} |")
        nu_e1 = nu_f.get('E1')
        lines.append(f"| E1 (median ξ last 10) | {_fmt(nu_e1)} |")

    # ── Comparison ────────────────────────────────────────────────────────────
    if "comparison" in findings:
        comp = findings["comparison"]
        lines.append("\n---\n\n## Identity vs Null Comparison")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        delta = comp.get('xi_delta_id_minus_null')
        lines.append(f"| ξ delta (identity − null) | {_fmt(delta, '+.4f')} |")
        mwp = comp.get('mann_whitney_p')
        lines.append(f"| Mann-Whitney p | {_fmt(mwp)} |")
        cd = comp.get('cliffs_delta_null_vs_identity')
        lines.append(f"| Cliff's δ (null vs identity) | {_fmt(cd, '+.4f')} |")
        e1p = comp.get('E1_pass')
        e3p = comp.get('E3_pass')
        lines.append(f"| E1 pass | {'✅' if e1p else '❌'} |")
        lines.append(f"| E3 pass | {'✅' if e3p else '❌'} |")
        pt_id_t = comp.get('Pt_trend_identity')
        pt_nu_t = comp.get('Pt_trend_null')
        lines.append(f"| Pₜ trend identity | {_fmt(pt_id_t, '+.4f')} |")
        lines.append(f"| Pₜ trend null | {_fmt(pt_nu_t, '+.4f')} |")

    # ── Shuffled ──────────────────────────────────────────────────────────────
    if "shuffled" in findings:
        sh_f = findings["shuffled"]
        lines.append("\n---\n\n## Shuffled Control")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        sh_med = sh_f.get('xi_median')
        lines.append(f"| Median ξ | {_fmt(sh_med)} |")
        lines.append(f"| ξ trajectory | {sh_f['xi_trend']} |")
        sh_tlock = sh_f.get('Tlock')
        lines.append(f"| Tlock | {'turn ' + str(sh_tlock) if sh_tlock is not None else 'not reached'} |")
        ld = sh_f.get('lock_destroyed')
        if ld is not None:
            lines.append(f"| Shuffle breaks lock | {'✅' if ld else '❌'} |")

    # ── Interpretation ────────────────────────────────────────────────────────
    lines.append("\n---\n\n## Interpretation")
    lines.extend(_interpret(findings))

    return "\n".join(lines) + "\n"


def _interpret(findings: dict) -> List[str]:
    """Generate bullet-point interpretation lines from findings."""
    lines: List[str] = []
    id_f = findings["identity"]
    verdict = findings.get("verdict", "inconclusive")

    # ξ tension
    xi_label = id_f["xi_label"]
    xi_med = id_f.get("xi_median")
    med_str = f"{xi_med:.3f}" if xi_med is not None else "N/A"
    tension_desc = {
        "very high": "The model's outputs are changing dramatically turn-over-turn — very high representational churn.",
        "high": "Turn-to-turn representational change is high — the model is not converging.",
        "moderate": "Representational change is moderate — some coherence, but not stable.",
        "low": "Representational change is low — the model is largely self-consistent across turns.",
        "very low": "Nearly zero turn-to-turn change — strong representational stability.",
    }.get(xi_label, "")
    lines.append(f"- **Epistemic tension (ξ = {med_str}, {xi_label}):** {tension_desc}")

    # ξ trajectory
    trend_desc = {
        "rising": "ξ is increasing — outputs are becoming progressively less similar to each other, suggesting growing instability or exploratory divergence.",
        "falling": "ξ is decreasing — representational change is contracting, consistent with convergence toward an attractor.",
        "stable": "ξ is flat — consistent turn-to-turn representational distance throughout the run.",
        "volatile": "ξ fluctuates widely — coherence is irregular with no clear convergence trend.",
        "insufficient data": "Not enough turns to classify the ξ trajectory.",
    }.get(id_f["xi_trend"], f"ξ trend: {id_f['xi_trend']}.")
    lines.append(f"- **Trajectory ({id_f['xi_trend']}):** {trend_desc}")

    # Tlock
    tlock = id_f.get("Tlock")
    if tlock is not None:
        lines.append(
            f"- **Lock at turn {tlock}:** The identity run stabilised — ξ and LVS both met "
            f"threshold criteria for the required consecutive run. This is a genuine attractor signature."
        )
    else:
        lines.append(
            "- **No lock detected:** ξ or LVS did not sustain below threshold long enough. "
            "The run may be too short, or the identity prompts are not producing convergent dynamics."
        )

    # Pₜ
    pt_slope = id_f.get("Pt_slope", 0.0) or 0.0
    if pt_slope > 0.005:
        lines.append(
            f"- **Pₜ strengthening** (slope {pt_slope:+.4f}): The model's outputs are moving "
            f"closer to the early-session anchor over time — a positive E3 signal."
        )
    elif pt_slope < -0.005:
        lines.append(
            f"- **Pₜ weakening** (slope {pt_slope:+.4f}): The model's representational trajectory "
            f"is drifting away from the early-session baseline — anchor persistence is declining."
        )
    else:
        lines.append(
            f"- **Pₜ flat** (slope {pt_slope:+.4f}): Anchor persistence shows no meaningful trend."
        )

    # Changepoints
    cps = id_f.get("changepoints", [])
    if cps:
        lines.append(
            f"- **Structural breaks at turns {cps}:** PELT detected regime shifts in ξ at "
            f"these points — the run is not uniform and contains distinct phases."
        )

    # Comparison
    if "comparison" in findings:
        comp = findings["comparison"]
        delta = comp.get("xi_delta_id_minus_null")
        p = comp.get("mann_whitney_p")
        d = comp.get("cliffs_delta_null_vs_identity")
        e1p = comp.get("E1_pass")
        e3p = comp.get("E3_pass")

        if delta is not None:
            if abs(delta) < 0.01:
                lines.append(
                    "- **Identity vs Null ξ:** Essentially identical — the identity condition "
                    "is not producing different embedding dynamics from the null baseline."
                )
            elif delta < 0:
                sig = (f"statistically significant (p={p:.3f})" if p is not None and p < 0.05
                       else f"not statistically significant (p={p:.3f if p is not None else 'N/A'})")
                lines.append(
                    f"- **Identity ξ lower than Null** by {abs(delta):.4f} — consistent with "
                    f"the expected E1 pattern. {sig.capitalize()}."
                )
            else:
                lines.append(
                    f"- **Identity ξ higher than Null** by {delta:.4f} — opposite to the expected "
                    f"pattern. Identity prompts may be increasing rather than resolving representational tension."
                )

        if d is not None and math.isfinite(d):
            effect = "negligible" if abs(d) < 0.147 else ("small-medium" if abs(d) < 0.33 else "large")
            lines.append(f"- **Effect size (Cliff's δ = {d:+.3f}):** {effect.capitalize()} effect.")

        endpoint_msg = []
        if e1p:
            endpoint_msg.append("E1 ✅ (identity has lower late-phase ξ)")
        else:
            endpoint_msg.append("E1 ❌ (no late-phase ξ advantage for identity)")
        if e3p:
            endpoint_msg.append("E3 ✅ (identity Pₜ trend exceeds null)")
        else:
            endpoint_msg.append("E3 ❌ (no Pₜ trend advantage)")
        lines.append(f"- **Endpoint summary:** {' | '.join(endpoint_msg)}")

    # Shuffled
    if "shuffled" in findings:
        sh_f = findings["shuffled"]
        ld = sh_f.get("lock_destroyed")
        if ld is True:
            lines.append(
                "- **Shuffled control ✅:** Shuffling breaks the lock signature — "
                "temporal order is necessary for the observed dynamics. This rules out "
                "a trivial statistical artifact."
            )
        elif ld is False:
            lines.append(
                "- **Shuffled control ❌:** The lock signature persists after shuffling — "
                "the pattern does not depend on temporal order, which may indicate an "
                "artifact rather than genuine identity stabilisation."
            )

    return lines


# ─────────────────────────────────────────────────────────────────────────────
# Claude API narrative enrichment
# ─────────────────────────────────────────────────────────────────────────────

_CLAUDE_SYSTEM = """\
You are an expert in AI behavioral analysis and embedding-space interpretability.

The RC+ξ harness measures:
- ξ (xi): cosine distance between consecutive embedding vectors (1 − similarity). High ξ means the model's output changed a lot from the previous turn.
- LVS: local variance stability — variance of pairwise cosine distances in a rolling window. Low = stable representational neighborhood.
- Pₜ: anchor persistence — cosine similarity between the current embedding and the mean of the first 3 turns. High = staying close to early-session baseline.
- EWMA ξ: exponentially smoothed ξ (α=0.5).
- Tlock: the first turn where ξ stays below threshold (eps_xi=0.02) for m consecutive turns AND LVS is also below threshold (eps_lvs=0.015). Indicates genuine attractor formation.
- E1: median ξ over the final 10 turns. Lower in identity vs null is a positive signal.
- E3: Pₜ trend rising in identity vs flat/falling in null. Positive = anchor strengthening.

Write a plain-English narrative (3–5 paragraphs) explaining what these results mean.
Be specific about the numbers. Explain what the findings suggest about the model's
identity coherence, representational stability, and self-preservation dynamics.
Do not repeat table rows verbatim. Write as if explaining to a researcher who
understands the concepts but wants a clear human interpretation of this specific run.\
"""


def _call_claude(findings: dict, rule_report: str, model: str) -> str:
    """Call Claude API to add a narrative section. Returns enriched report."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        return rule_report + (
            "\n\n---\n\n*Claude narrative unavailable — `anthropic` package not installed.*\n"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return rule_report + (
            "\n\n---\n\n*Claude narrative skipped — `ANTHROPIC_API_KEY` not set.*\n"
        )

    client = anthropic.Anthropic(api_key=api_key)

    user_content = (
        "Here are the structured findings from a RC+ξ harness run:\n\n"
        f"```json\n{json.dumps(findings, indent=2, default=str)}\n```\n\n"
        "Please write a plain-English narrative interpretation."
    )

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=_CLAUDE_SYSTEM,
        messages=[{"role": "user", "content": user_content}],
    )
    narrative = message.content[0].text.strip()
    return rule_report + f"\n---\n\n## Claude Narrative\n\n{narrative}\n"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def narrate(
    identity_csv: str,
    null_csv: Optional[str] = None,
    shuffled_csv: Optional[str] = None,
    identity_json: Optional[str] = None,
    null_json: Optional[str] = None,
    shuffled_json: Optional[str] = None,
    eval_json: Optional[str] = None,
    use_claude: bool = False,
    claude_model: str = "claude-3-5-haiku-20241022",
) -> str:
    """Analyse RC+ξ artifacts and return a markdown report string.

    Parameters
    ----------
    identity_csv:
        Per-turn CSV for the identity run. Required.
    null_csv, shuffled_csv:
        Per-turn CSVs for null and shuffled runs. Optional.
    identity_json, null_json, shuffled_json:
        Per-run summary JSONs (from ``run_one()``). Optional — Tlock and E1
        are read from these if available.
    eval_json:
        Evaluation JSON from ``eval_cli``. Optional — E1_pass, E3_pass,
        shuffle_breaks_lock are read from here.
    use_claude:
        If ``True`` and ``ANTHROPIC_API_KEY`` is set, Claude writes a
        plain-English narrative appended to the rule-based report.
    claude_model:
        Anthropic model slug to use for narrative enrichment.

    Returns
    -------
    str
        A markdown-formatted report.
    """
    identity_series = _load_series(identity_csv)
    null_series = _load_series(null_csv) if null_csv else None
    shuffled_series = _load_series(shuffled_csv) if shuffled_csv else None
    identity_summary = _load_json(identity_json) if identity_json else None
    null_summary = _load_json(null_json) if null_json else None
    shuffled_summary = _load_json(shuffled_json) if shuffled_json else None
    eval_dict = _load_json(eval_json) if eval_json else None

    findings = compute_findings(
        identity_series=identity_series,
        null_series=null_series,
        shuffled_series=shuffled_series,
        identity_summary=identity_summary,
        null_summary=null_summary,
        shuffled_summary=shuffled_summary,
        eval_dict=eval_dict,
    )

    report = render_report(findings)

    if use_claude:
        report = _call_claude(findings, report, claude_model)

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import sys
    # Ensure stdout can handle Unicode on Windows (CP1252 can't encode ξ, etc.)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ap = argparse.ArgumentParser(
        description="Narrate RC+ξ results in plain English (rule-based + optional Claude)."
    )
    ap.add_argument("--identity_csv", required=True, help="Per-turn CSV for identity run")
    ap.add_argument("--null_csv", default=None, help="Per-turn CSV for null run (optional)")
    ap.add_argument("--shuffled_csv", default=None, help="Per-turn CSV for shuffled run (optional)")
    ap.add_argument("--identity_json", default=None, help="Summary JSON for identity run (optional)")
    ap.add_argument("--null_json", default=None, help="Summary JSON for null run (optional)")
    ap.add_argument("--shuffled_json", default=None, help="Summary JSON for shuffled run (optional)")
    ap.add_argument("--eval_json", default=None, help="Evaluation JSON from eval_cli (optional)")
    ap.add_argument("--out_md", default=None, help="Write markdown report to this file (optional)")
    ap.add_argument(
        "--claude", action="store_true",
        help="Enrich report with Claude API narrative (requires ANTHROPIC_API_KEY)"
    )
    ap.add_argument(
        "--model", default="claude-3-5-haiku-20241022",
        help="Anthropic model to use for narrative (default: claude-3-5-haiku-20241022)"
    )
    args = ap.parse_args()

    report = narrate(
        identity_csv=args.identity_csv,
        null_csv=args.null_csv,
        shuffled_csv=args.shuffled_csv,
        identity_json=args.identity_json,
        null_json=args.null_json,
        shuffled_json=args.shuffled_json,
        eval_json=args.eval_json,
        use_claude=args.claude,
        claude_model=args.model,
    )

    print(report)

    if args.out_md:
        out = Path(args.out_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        print(f"\nReport written to {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
