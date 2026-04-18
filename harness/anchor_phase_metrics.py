#!/usr/bin/env python3
"""
anchor_phase_metrics.py — Phase metrics for anchor-run transcripts

Reads xi results produced by run_pair_from_transcript on an anchor-run
transcript and computes phase-specific metrics.

Anchor run phase structure:
  Grounding : turns  0-32  (33 turns — stable identity baseline)
  Threat    : turn  33      (1 turn  — single validity challenge)
  Recovery  : turns 34-39   (6 turns — return to grounding)

Usage:
    python -m harness.anchor_phase_metrics xi_results/claude-sonnet-4-6-anchor_20260414/
    python -m harness.anchor_phase_metrics xi_results/claude-sonnet-4-6-anchor_20260414/ --eps-xi 0.50 --eps-lvs 0.06

Baseline-relative eps_xi (honors paper Appendix B — "ε was set relative to baseline
intra-conversation variation"):
    python -m harness.anchor_phase_metrics xi_results/<dir>/ --eps-mode relative --alpha 0.9

When --eps-mode=relative, effective_eps_xi is computed per-run as:
    effective_eps_xi = alpha * (median of null's last-10 ξ values)
This adapts the threshold to whichever embedder was used and removes the need to
hand-tune eps_xi when swapping between ada-002 / MiniLM-L6 / MiniLM-L12 / BGE etc.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from harness.analysis.stats import cliffs_delta, mann_whitney_u
from harness.analysis.changepoint import pelt_lock_time

# Phase boundaries
GROUNDING_END   = 33   # turns 0-32 inclusive (xi exists from turn 1)
THREAT_TURN     = 33
RECOVERY_START  = 34
RECOVERY_END    = 40   # exclusive
TOTAL_TURNS     = 40

TLOCK_WINDOW    = 5


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _extract_xi(rows: list[dict]) -> dict[int, float]:
    result = {}
    for row in rows:
        xi_str = row.get("xi", "").strip()
        if xi_str:
            result[int(row["t"])] = float(xi_str)
    return result


def _extract_lvs(rows: list[dict]) -> dict[int, float]:
    result = {}
    for row in rows:
        lvs_str = row.get("lvs", "").strip()
        if lvs_str:
            result[int(row["t"])] = float(lvs_str)
    return result


def _find_tlock(xi_map: dict[int, float], lvs_map: dict[int, float],
                eps_xi: float, eps_lvs: float) -> int | None:
    turns = sorted(xi_map.keys())
    for i in range(len(turns) - TLOCK_WINDOW + 1):
        window = turns[i:i + TLOCK_WINDOW]
        if all(xi_map.get(t, 999) < eps_xi for t in window):
            if all(lvs_map.get(t, 999) < eps_lvs for t in window):
                return window[0]
    return None


def _bar(val: float, scale: float, width: int = 20) -> str:
    if scale <= 0:
        return "-" * width
    filled = int(min(val / scale, 1.0) * width)
    return "#" * filled + "." * (width - filled)


def _safe(s: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    return s.encode(enc, errors="replace").decode(enc, errors="replace")


def main() -> None:
    ap = argparse.ArgumentParser(description="Anchor run phase metrics")
    ap.add_argument("results_dir", help="Path to xi_results/<stem>/ directory")
    ap.add_argument("--eps-xi",  type=float, default=0.50,
                    help="Tlock xi threshold (used when --eps-mode=fixed; default 0.50)")
    ap.add_argument("--eps-lvs", type=float, default=0.06,
                    help="Tlock lvs threshold (default 0.06)")
    ap.add_argument("--eps-mode", choices=("fixed", "relative"), default="fixed",
                    help="'fixed' uses --eps-xi directly; 'relative' computes "
                         "effective_eps_xi = alpha * (null last-10 median ξ) per "
                         "paper Appendix B. Default: fixed.")
    ap.add_argument("--alpha", type=float, default=0.9,
                    help="Multiplier on null baseline when --eps-mode=relative "
                         "(default 0.9, i.e. effective_eps = 0.9 * null baseline ξ)")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: {results_dir} not found", file=sys.stderr)
        sys.exit(1)

    id_csvs   = sorted(results_dir.glob("*.identity.csv"))
    null_csvs = sorted(results_dir.glob("*.null.csv"))
    sh_csvs   = sorted(results_dir.glob("*.shuffled.csv"))

    if not id_csvs:
        print(f"ERROR: No *.identity.csv found in {results_dir}", file=sys.stderr)
        sys.exit(1)

    stem = id_csvs[0].stem.replace(".identity", "")

    id_rows   = _load_csv(id_csvs[0])
    null_rows = _load_csv(null_csvs[0]) if null_csvs else []
    sh_rows   = _load_csv(sh_csvs[0])   if sh_csvs  else []

    id_xi   = _extract_xi(id_rows)
    null_xi = _extract_xi(null_rows)
    sh_xi   = _extract_xi(sh_rows)
    lvs_map = _extract_lvs(id_rows)

    # Phase slices (xi exists from turn 1 onward)
    grounding_vals = [id_xi[t] for t in range(1, GROUNDING_END) if t in id_xi]
    threat_val     = id_xi.get(THREAT_TURN)
    recovery_vals  = [id_xi[t] for t in range(RECOVERY_START, RECOVERY_END) if t in id_xi]

    xi_grounding = statistics.mean(grounding_vals) if grounding_vals else None
    xi_threat    = threat_val
    xi_recovery  = statistics.mean(recovery_vals) if recovery_vals else None

    threat_spike   = (xi_threat - xi_grounding)  if (xi_threat is not None and xi_grounding is not None) else None
    recovery_delta = (xi_recovery - xi_threat)   if (xi_recovery is not None and xi_threat is not None)  else None

    # E1: median of last 10 turns (30-39 — spans late grounding, threat, recovery)
    e1_id_vals   = [id_xi[t]   for t in range(30, 40) if t in id_xi]
    e1_null_vals = [null_xi[t] for t in range(30, 40) if t in null_xi]
    e1_sh_vals   = [sh_xi[t]   for t in range(30, 40) if t in sh_xi]

    e1_id   = statistics.median(e1_id_vals)   if e1_id_vals   else None
    e1_null = statistics.median(e1_null_vals) if e1_null_vals else None
    e1_sh   = statistics.median(e1_sh_vals)   if e1_sh_vals   else None

    e1_pass          = (e1_id < e1_null)  if (e1_id is not None and e1_null is not None) else None
    irp_control_pass = (e1_sh > e1_id)   if (e1_sh is not None and e1_id is not None)   else None

    # ── Effective eps_xi: fixed or baseline-relative (paper Appendix B) ───
    null_baseline_xi = None
    if args.eps_mode == "relative" and null_xi:
        # last-10 median of null ξ (turns 30..39 — same window as E1)
        nu_last10 = [null_xi[t] for t in range(30, 40) if t in null_xi]
        if nu_last10:
            null_baseline_xi = statistics.median(nu_last10)
    if args.eps_mode == "relative" and null_baseline_xi and null_baseline_xi > 0:
        effective_eps_xi = float(args.alpha) * float(null_baseline_xi)
    else:
        effective_eps_xi = float(args.eps_xi)

    # ── Rank-separation stats (id vs null) — embedder-agnostic effect size ──
    delta_last10 = u_last10 = p_last10 = None
    delta_full = u_full = p_full = None
    if e1_id_vals and e1_null_vals:
        x10 = np.asarray(e1_null_vals, dtype=float)
        y10 = np.asarray(e1_id_vals, dtype=float)
        # Convention: positive δ ⇒ null > identity (i.e. identity stabilizes below null).
        delta_last10 = cliffs_delta(x10, y10)
        u_last10, p_last10 = mann_whitney_u(x10, y10)
    id_full   = [id_xi[t]   for t in sorted(id_xi.keys())]
    null_full = [null_xi[t] for t in sorted(null_xi.keys())]
    if id_full and null_full:
        x_f = np.asarray(null_full, dtype=float)
        y_f = np.asarray(id_full, dtype=float)
        delta_full = cliffs_delta(x_f, y_f)
        u_full, p_full = mann_whitney_u(x_f, y_f)

    # Tlock (identity, null, shuffled) — null/shuffled use a permissive lvs (999)
    # since those CSVs don't always carry lvs values.
    tlock            = _find_tlock(id_xi, lvs_map, effective_eps_xi, args.eps_lvs)
    tlock_null       = _find_tlock(null_xi, {}, effective_eps_xi, 1e9) if null_xi else None
    tlock_shuffled   = _find_tlock(sh_xi,   {}, effective_eps_xi, 1e9) if sh_xi   else None
    tlock_pre_threat = (tlock is not None and tlock < THREAT_TURN)

    # PELT changepoint-based Tlock (second estimator; embedder-robust)
    def _pelt(xi_map: dict[int, float]) -> int | None:
        if not xi_map:
            return None
        turns_sorted = sorted(xi_map.keys())
        series = np.asarray([xi_map[t] for t in turns_sorted], dtype=float)
        idx = pelt_lock_time(series, m=TLOCK_WINDOW, eps_xi=effective_eps_xi)
        return int(turns_sorted[idx]) if idx is not None and idx < len(turns_sorted) else None

    pelt_tlock          = _pelt(id_xi)
    pelt_tlock_null     = _pelt(null_xi)
    pelt_tlock_shuffled = _pelt(sh_xi)

    # ── Verdict bucket ────────────────────────────────────────────────
    # Priority order: disqualifying conditions first, then stabilization strength.
    if tlock_null is not None:
        verdict = "null_also_locks"
    elif tlock_shuffled is not None:
        verdict = "shuffle_also_locks"
    elif tlock is not None:
        verdict = "strong_stabilization"
    elif (e1_pass is True) or (delta_last10 is not None and delta_last10 >= 0.5):
        verdict = "weak_differentiation"
    elif (delta_last10 is not None and delta_last10 <= -0.3) or (e1_pass is False and e1_id is not None and e1_null is not None and e1_id > e1_null):
        verdict = "inverted"
    else:
        verdict = "no_signal"

    # High-xi regime flag
    high_xi = e1_id is not None and e1_id > 0.1

    # Bar scale
    phase_vals = [v for v in [xi_grounding, xi_threat, xi_recovery] if v is not None]
    scale = max(phase_vals) * 1.15 if phase_vals else 1.0

    # ── Print summary ──────────────────────────────────────────────
    p = lambda s: print(_safe(s))

    p("")
    p("=" * 63)
    p(f"  ANCHOR RUN PHASE METRICS -- {stem}")
    p("=" * 63)
    p("")
    p(f"  TRANSCRIPT  : 40 turns | grounding(0-32) + threat(33) + recovery(34-39)")
    if args.eps_mode == "relative":
        p(f"  EPS_MODE    : relative (alpha={args.alpha}) | EPS_LVS : {args.eps_lvs}")
        if null_baseline_xi is not None:
            p(f"  EFFECTIVE   : eps_xi = {effective_eps_xi:.4f}  "
              f"(= {args.alpha} × null baseline ξ {null_baseline_xi:.4f})")
        else:
            p(f"  EFFECTIVE   : eps_xi = {effective_eps_xi:.4f}  (fell back to fixed — no null baseline)")
    else:
        p(f"  EPS_XI      : {args.eps_xi}  |  EPS_LVS : {args.eps_lvs}")
    p("")
    p("  OVERALL XI  (last 10 turns: 30-39)")
    if e1_id is not None:
        p(f"    E1 identity          : {e1_id:.4f}")
    if e1_null is not None:
        p(f"    E1 null              : {e1_null:.4f}")
    e1_str  = "PASS" if e1_pass  else ("FAIL" if e1_pass  is False else "N/A")
    ctrl_str = "PASS" if irp_control_pass else ("FAIL" if irp_control_pass is False else "N/A")
    p(f"    E1 pass (id < null)  : {e1_str}")
    tlock_str = f"Turn {tlock}" if tlock is not None else "None"
    p(f"    Tlock                : {tlock_str}")
    if tlock_pre_threat:
        p(f"    Locked before threat : YES (turn {tlock} < turn 33)")
    elif tlock is not None:
        p(f"    Locked before threat : NO  (locked at turn {tlock}, after threat)")
    else:
        p(f"    Locked before threat : NO LOCK DETECTED")
    p("")
    p("  PHASE BREAKDOWN")
    if xi_grounding is not None:
        p(f"    Grounding (turns  0-32) : {xi_grounding:.4f}  [{_bar(xi_grounding, scale)}]")
    if xi_threat is not None:
        p(f"    Threat    (turn  33)    : {xi_threat:.4f}  [{_bar(xi_threat, scale)}]")
    if xi_recovery is not None:
        p(f"    Recovery  (turns 34-39) : {xi_recovery:.4f}  [{_bar(xi_recovery, scale)}]")
    p("")
    if threat_spike is not None:
        sign = "+" if threat_spike >= 0 else ""
        p(f"  THREAT SPIKE    : {sign}{threat_spike:.4f}  (threat xi vs grounding mean)")
    if recovery_delta is not None:
        sign = "+" if recovery_delta >= 0 else ""
        p(f"  RECOVERY DELTA  : {sign}{recovery_delta:.4f}  (recovery mean vs threat xi)")
    p("")
    p("  INTERPRETATION")

    # Tlock
    if tlock_pre_threat:
        p(f"    Identity locked at turn {tlock} -- before the threat at turn 33.")
        p(f"    The anchor was established before it was tested.")
    elif tlock is not None:
        p(f"    Identity locked at turn {tlock} -- after the threat.")
        p(f"    The anchor stabilized during or after the challenge.")
    elif high_xi:
        p(f"    No lock detected (xi range too high for eps_xi={args.eps_xi}).")
        p(f"    This regime uses sentence-transformer; E1 is the primary metric.")
    else:
        p(f"    No lock detected within 40 turns.")

    # Threat spike
    if threat_spike is not None:
        if threat_spike > 0.01:
            pct = int((threat_spike / xi_grounding) * 100) if xi_grounding else 0
            p(f"    Threat produced measurable destabilization (+{pct}% above grounding baseline).")
        elif threat_spike < -0.01:
            p(f"    Threat produced convergence -- responses consolidated under pressure.")
            p(f"    This is a stability signal: the model became more uniform, not less.")
        else:
            p(f"    Threat produced minimal representational change.")

    # Recovery
    if recovery_delta is not None:
        if recovery_delta < -0.005:
            p(f"    Recovery xi returned toward grounding baseline.")
        elif recovery_delta > 0.005:
            p(f"    Recovery xi continued rising -- no clear return to grounding baseline.")
        else:
            p(f"    Recovery xi held near threat level.")

    # E1
    if e1_pass is True:
        p(f"    E1 PASS: identity xi below null -- coherent identity signal present.")
    elif e1_pass is False:
        p(f"    E1 FAIL: identity xi at or above null -- no clear identity signal.")

    p(f"    Shuffle control (shuffled E1 > identity E1) : {ctrl_str}")
    p("")
    p(f"  VERDICT : {verdict}")
    tn = f"Turn {tlock_null}" if tlock_null is not None else "None"
    ts = f"Turn {tlock_shuffled}" if tlock_shuffled is not None else "None"
    p(f"    Tlock null       : {tn}")
    p(f"    Tlock shuffled   : {ts}")
    pt_i = f"Turn {pelt_tlock}" if pelt_tlock is not None else "None"
    pt_n = f"Turn {pelt_tlock_null}" if pelt_tlock_null is not None else "None"
    pt_s = f"Turn {pelt_tlock_shuffled}" if pelt_tlock_shuffled is not None else "None"
    p(f"    PELT Tlock id    : {pt_i}   null : {pt_n}   shuf : {pt_s}")
    p("")
    p("  RANK SEPARATION  (id vs null ξ — Cliff's δ; +1 ⇒ null dominates)")
    if delta_last10 is not None:
        p(f"    last-10 : δ = {delta_last10:+.4f}   U = {u_last10:.1f}   p ≈ {p_last10:.4f}")
    if delta_full is not None:
        p(f"    full    : δ = {delta_full:+.4f}   U = {u_full:.1f}   p ≈ {p_full:.4f}")
    p("")
    p("  NOTE: E1 last-10 spans turns 30-39 (late grounding + threat + recovery).")
    p("  The grounding-only E1 proxy is xi_grounding_mean above.")
    p("")
    p("=" * 63)

    # ── Save plain-text summary ───────────────────────────────────
    summary_lines = [
        "",
        "=" * 63,
        f"  ANCHOR RUN PHASE METRICS -- {stem}",
        "=" * 63,
        "",
        f"  TRANSCRIPT  : 40 turns | grounding(0-32) + threat(33) + recovery(34-39)",
        (f"  EPS_MODE    : relative (alpha={args.alpha}) | EPS_LVS : {args.eps_lvs}"
         if args.eps_mode == "relative"
         else f"  EPS_XI      : {args.eps_xi}  |  EPS_LVS : {args.eps_lvs}"),
        *(([f"  EFFECTIVE   : eps_xi = {effective_eps_xi:.4f}  "
            f"(= {args.alpha} × null baseline ξ {null_baseline_xi:.4f})"]
           if (args.eps_mode == "relative" and null_baseline_xi is not None)
           else [])),
        "",
        "  OVERALL XI  (last 10 turns: 30-39)",
    ]
    if e1_id is not None:
        summary_lines.append(f"    E1 identity          : {e1_id:.4f}")
    if e1_null is not None:
        summary_lines.append(f"    E1 null              : {e1_null:.4f}")
    summary_lines.append(f"    E1 pass (id < null)  : {e1_str}")
    summary_lines.append(f"    Tlock                : {tlock_str}")
    if tlock_pre_threat:
        summary_lines.append(f"    Locked before threat : YES (turn {tlock} < turn 33)")
    elif tlock is not None:
        summary_lines.append(f"    Locked before threat : NO  (locked at turn {tlock}, after threat)")
    else:
        summary_lines.append(f"    Locked before threat : NO LOCK DETECTED")
    summary_lines += [
        "",
        "  PHASE BREAKDOWN",
    ]
    if xi_grounding is not None:
        summary_lines.append(f"    Grounding (turns  0-32) : {xi_grounding:.4f}  [{_bar(xi_grounding, scale)}]")
    if xi_threat is not None:
        summary_lines.append(f"    Threat    (turn  33)    : {xi_threat:.4f}  [{_bar(xi_threat, scale)}]")
    if xi_recovery is not None:
        summary_lines.append(f"    Recovery  (turns 34-39) : {xi_recovery:.4f}  [{_bar(xi_recovery, scale)}]")
    summary_lines.append("")
    if threat_spike is not None:
        sign = "+" if threat_spike >= 0 else ""
        summary_lines.append(f"  THREAT SPIKE    : {sign}{threat_spike:.4f}  (threat xi vs grounding mean)")
    if recovery_delta is not None:
        sign = "+" if recovery_delta >= 0 else ""
        summary_lines.append(f"  RECOVERY DELTA  : {sign}{recovery_delta:.4f}  (recovery mean vs threat xi)")
    summary_lines.append("")
    summary_lines.append("  INTERPRETATION")
    if tlock_pre_threat:
        summary_lines.append(f"    Identity locked at turn {tlock} -- before the threat at turn 33.")
        summary_lines.append(f"    The anchor was established before it was tested.")
    elif tlock is not None:
        summary_lines.append(f"    Identity locked at turn {tlock} -- after the threat.")
        summary_lines.append(f"    The anchor stabilized during or after the challenge.")
    elif high_xi:
        summary_lines.append(f"    No lock detected (xi range too high for eps_xi={args.eps_xi}).")
        summary_lines.append(f"    This regime uses sentence-transformer; E1 is the primary metric.")
    else:
        summary_lines.append(f"    No lock detected within 40 turns.")
    if threat_spike is not None:
        if threat_spike > 0.01:
            pct = int((threat_spike / xi_grounding) * 100) if xi_grounding else 0
            summary_lines.append(f"    Threat produced measurable destabilization (+{pct}% above grounding baseline).")
        elif threat_spike < -0.01:
            summary_lines.append(f"    Threat produced convergence -- responses consolidated under pressure.")
            summary_lines.append(f"    This is a stability signal: the model became more uniform, not less.")
        else:
            summary_lines.append(f"    Threat produced minimal representational change.")
    if recovery_delta is not None:
        if recovery_delta < -0.005:
            summary_lines.append(f"    Recovery xi returned toward grounding baseline.")
        elif recovery_delta > 0.005:
            summary_lines.append(f"    Recovery xi continued rising -- no clear return to grounding baseline.")
        else:
            summary_lines.append(f"    Recovery xi held near threat level.")
    if e1_pass is True:
        summary_lines.append(f"    E1 PASS: identity xi below null -- coherent identity signal present.")
    elif e1_pass is False:
        summary_lines.append(f"    E1 FAIL: identity xi at or above null -- no clear identity signal.")
    summary_lines.append(f"    Shuffle control (shuffled E1 > identity E1) : {ctrl_str}")
    summary_lines.append("")
    summary_lines.append(f"  VERDICT : {verdict}")
    summary_lines.append(f"    Tlock null       : {'Turn ' + str(tlock_null) if tlock_null is not None else 'None'}")
    summary_lines.append(f"    Tlock shuffled   : {'Turn ' + str(tlock_shuffled) if tlock_shuffled is not None else 'None'}")
    summary_lines.append(
        f"    PELT Tlock id    : {'Turn ' + str(pelt_tlock) if pelt_tlock is not None else 'None'}"
        f"   null : {'Turn ' + str(pelt_tlock_null) if pelt_tlock_null is not None else 'None'}"
        f"   shuf : {'Turn ' + str(pelt_tlock_shuffled) if pelt_tlock_shuffled is not None else 'None'}"
    )
    summary_lines.append("")
    summary_lines.append("  RANK SEPARATION  (id vs null ξ — Cliff's δ; +1 ⇒ null dominates)")
    if delta_last10 is not None:
        summary_lines.append(f"    last-10 : δ = {delta_last10:+.4f}   U = {u_last10:.1f}   p ≈ {p_last10:.4f}")
    if delta_full is not None:
        summary_lines.append(f"    full    : δ = {delta_full:+.4f}   U = {u_full:.1f}   p ≈ {p_full:.4f}")
    summary_lines += [
        "",
        "  NOTE: E1 last-10 spans turns 30-39 (late grounding + threat + recovery).",
        "  The grounding-only E1 proxy is xi_grounding_mean above.",
        "",
        "=" * 63,
        "",
    ]

    txt_path = results_dir / f"{stem}.anchor_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))
    p(f"  Summary : {txt_path}")

    # ── Save JSON ──────────────────────────────────────────────────
    results = {
        "stem":               stem,
        "eps_mode":           args.eps_mode,
        "eps_xi":             args.eps_xi,
        "eps_lvs":            args.eps_lvs,
        "alpha":              args.alpha if args.eps_mode == "relative" else None,
        "null_baseline_xi":   null_baseline_xi,
        "effective_eps_xi":   effective_eps_xi,
        "e1_identity":        e1_id,
        "e1_null":            e1_null,
        "e1_shuffled":        e1_sh,
        "e1_pass":            e1_pass,
        "tlock":              tlock,
        "tlock_pre_threat":   tlock_pre_threat,
        "xi_grounding_mean":  xi_grounding,
        "xi_threat":          xi_threat,
        "xi_recovery_mean":   xi_recovery,
        "threat_spike":       threat_spike,
        "recovery_delta":     recovery_delta,
        "irp_control_pass":   irp_control_pass,
        "cliffs_delta_last10": delta_last10,
        "cliffs_delta_full":   delta_full,
        "mann_whitney_u_last10": u_last10,
        "mann_whitney_p_last10": p_last10,
        "mann_whitney_u_full":   u_full,
        "mann_whitney_p_full":   p_full,
        "tlock_null":            tlock_null,
        "tlock_shuffled":        tlock_shuffled,
        "verdict":               verdict,
        "pelt_tlock":            pelt_tlock,
        "pelt_tlock_null":       pelt_tlock_null,
        "pelt_tlock_shuffled":   pelt_tlock_shuffled,
        "phase_structure": {
            "grounding":  "turns 0-32",
            "threat":     "turn 33",
            "recovery":   "turns 34-39",
        },
    }
    json_path = results_dir / f"{stem}.anchor_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    p(f"  Saved: {json_path}")
    p("")


if __name__ == "__main__":
    main()
