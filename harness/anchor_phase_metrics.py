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
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

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
                    help="Tlock xi threshold (default 0.50 for sentence-transformer)")
    ap.add_argument("--eps-lvs", type=float, default=0.06,
                    help="Tlock lvs threshold (default 0.06)")
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

    # Tlock
    tlock           = _find_tlock(id_xi, lvs_map, args.eps_xi, args.eps_lvs)
    tlock_pre_threat = (tlock is not None and tlock < THREAT_TURN)

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
        f"  EPS_XI      : {args.eps_xi}  |  EPS_LVS : {args.eps_lvs}",
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
        "eps_xi":             args.eps_xi,
        "eps_lvs":            args.eps_lvs,
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
