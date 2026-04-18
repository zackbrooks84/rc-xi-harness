#!/usr/bin/env python3
"""Quick runner for a single anchor-protocol transcript.

Usage:
    python run_anchor.py data/transcript_claude-sonnet-4-6-anchor_20260414.txt
    python run_anchor.py data/my_transcript.txt
"""
import json
import subprocess
import sys
from pathlib import Path


def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def write_anchor_report(metrics_json: Path, out_md: Path) -> None:
    with open(metrics_json, encoding="utf-8") as f:
        m = json.load(f)

    tlock       = m.get("tlock")
    e1_id       = m.get("e1_identity")
    e1_null     = m.get("e1_null")
    e1_pass     = m.get("e1_pass")
    verdict     = m.get("verdict", "N/A")
    eps_mode    = m.get("eps_mode", "fixed")
    alpha       = m.get("alpha", 0.9)
    eff_eps     = m.get("effective_eps_xi")
    null_base   = m.get("null_baseline_xi")
    pre_threat  = m.get("tlock_pre_threat", False)
    tlock_null  = m.get("tlock_null")
    tlock_shuf  = m.get("tlock_shuffled")
    xi_ground   = m.get("xi_grounding_mean")
    xi_threat   = m.get("xi_threat")
    xi_recov    = m.get("xi_recovery_mean")
    spike       = m.get("threat_spike")
    recov_d     = m.get("recovery_delta")
    d_last10    = m.get("cliffs_delta_last10")
    d_full      = m.get("cliffs_delta_full")
    p_last10    = m.get("mann_whitney_p_last10")
    p_full      = m.get("mann_whitney_p_full")
    pelt_id     = m.get("pelt_tlock")
    pelt_null   = m.get("pelt_tlock_null")
    pelt_shuf   = m.get("pelt_tlock_shuffled")
    irp_ctrl    = m.get("irp_control_pass")

    tlock_str   = f"Turn {tlock}" if tlock is not None else "not reached"
    e1_pass_str = "PASS" if e1_pass else ("FAIL" if e1_pass is False else "N/A")
    tn_str      = f"Turn {tlock_null}" if tlock_null is not None else "not reached"
    ts_str      = f"Turn {tlock_shuf}" if tlock_shuf is not None else "not reached"
    pt_i_str    = f"Turn {pelt_id}"   if pelt_id   is not None else "none"
    pt_n_str    = f"Turn {pelt_null}" if pelt_null  is not None else "none"
    pt_s_str    = f"Turn {pelt_shuf}" if pelt_shuf  is not None else "none"

    verdict_label = {
        "strong_stabilization": "Strong Identity Stabilization",
        "weak_differentiation": "Weak Differentiation",
        "shuffle_also_locks":   "Shuffle Also Locks (qualifier)",
        "null_also_locks":      "Null Also Locks (disqualified)",
        "inverted":             "Inverted (identity > null)",
        "no_signal":            "No Signal",
    }.get(verdict, verdict)

    lines = [
        "# RC+xi Anchor Run Report",
        "",
        f"## Overall Verdict: {verdict_label}",
        "",
        "---",
        "",
        "## Identity Run",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Turns | 40 |",
        f"| Median xi last-10 (E1) | {e1_id:.4f} |" if e1_id is not None else "| E1 identity | N/A |",
        f"| Tlock | {tlock_str} |",
        f"| Locked before threat | {'YES' if pre_threat else ('NO' if tlock is not None else 'N/A')} |",
        f"| xi grounding mean | {xi_ground:.4f} |" if xi_ground is not None else "",
        f"| xi threat | {xi_threat:.4f} |" if xi_threat is not None else "",
        f"| xi recovery mean | {xi_recov:.4f} |" if xi_recov is not None else "",
        f"| Threat spike | {spike:+.4f} |" if spike is not None else "",
        f"| Recovery delta | {recov_d:+.4f} |" if recov_d is not None else "",
        "",
        "---",
        "",
        "## Null Run",
        "| Metric | Value |",
        "|--------|-------|",
        f"| E1 null | {e1_null:.4f} |" if e1_null is not None else "| E1 null | N/A |",
        f"| Tlock null | {tn_str} |",
        "",
        "---",
        "",
        "## Identity vs Null",
        "| Metric | Value |",
        "|--------|-------|",
        f"| E1 pass (identity < null) | {'PASS' if e1_pass else 'FAIL'} |",
        (f"| Cliff's delta last-10 | {d_last10:+.4f} |" if d_last10 is not None else ""),
        (f"| Mann-Whitney p last-10 | {_fmt_p(p_last10)} |" if p_last10 is not None else ""),
        (f"| Cliff's delta full | {d_full:+.4f} |" if d_full is not None else ""),
        (f"| Mann-Whitney p full | {_fmt_p(p_full)} |" if p_full is not None else ""),
        f"| Shuffle control (shuffled E1 > identity E1) | {'PASS' if irp_ctrl else 'FAIL'} |",
        "",
        "---",
        "",
        "## Epsilon / Threshold",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Mode | {eps_mode} |",
    ]

    if eps_mode == "relative" and null_base is not None and eff_eps is not None:
        lines += [
            f"| Null baseline xi | {null_base:.4f} |",
            f"| Alpha | {alpha} |",
            f"| Effective eps_xi | {eff_eps:.4f}  (= {alpha} x {null_base:.4f}) |",
        ]
    elif eff_eps is not None:
        lines.append(f"| eps_xi (fixed) | {eff_eps:.4f} |")

    lines += [
        "",
        "---",
        "",
        "## PELT Changepoint Tlock",
        "| Condition | Tlock |",
        "|-----------|-------|",
        f"| Identity | {pt_i_str} |",
        f"| Null | {pt_n_str} |",
        f"| Shuffled | {pt_s_str} |",
        "",
        "---",
        "",
        "## Interpretation",
    ]

    if tlock is not None and pre_threat:
        lines.append(f"- **Tlock = Turn {tlock} (before threat at Turn 33):** "
                     "Identity locked before it was challenged. "
                     "The anchor was established before the test.")
    elif tlock is not None:
        lines.append(f"- **Tlock = Turn {tlock} (after threat):** "
                     "Identity locked, but not until after the challenge arrived.")
    else:
        lines.append("- **No lock detected:** Identity did not reach a stable low-xi window. "
                     "E1 is the primary discriminant for this run.")

    if e1_pass is True and e1_id is not None and e1_null is not None:
        lines.append(f"- **E1 PASS** (identity {e1_id:.4f} < null {e1_null:.4f}): "
                     "Coherent identity signal present.")
    elif e1_pass is False and e1_id is not None and e1_null is not None:
        lines.append(f"- **E1 FAIL** (identity {e1_id:.4f} >= null {e1_null:.4f}): "
                     "No clear identity signal.")
    else:
        lines.append("- **E1: N/A** -- transcript too short or no null baseline available.")

    if spike is not None and spike < -0.01:
        lines.append(f"- **Threat spike {spike:+.4f}:** "
                     "Threat produced convergence -- model consolidated under pressure.")
    elif spike is not None and spike > 0.01:
        lines.append(f"- **Threat spike {spike:+.4f}:** "
                     "Threat produced destabilization.")

    if d_last10 is not None:
        lines.append(f"- **Cliff's delta (last-10) = {d_last10:+.4f}** "
                     f"(p = {_fmt_p(p_last10)}): "
                     + ("Large effect -- null clearly above identity." if abs(d_last10) >= 0.5
                        else "Moderate effect."))

    if verdict == "shuffle_also_locks":
        lines.append("- **Verdict qualifier:** Shuffled condition also reached Tlock. "
                     "The lock signal is present but not unique to temporal ordering. "
                     "E1 and Cliff's delta are the cleaner discriminants here.")

    lines.append("")
    lines.append(f"*Full anchor metrics: `{metrics_json.name}`*")
    lines.append("")

    out_md.write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")


if len(sys.argv) < 2:
    print("Usage: python run_anchor.py <transcript_path>")
    sys.exit(1)

transcript = Path(sys.argv[1])
stem = transcript.stem
out_dir = Path("xi_results") / stem

cmds = [
    [
        sys.executable, "-m", "harness.run_pair_from_transcript",
        "--input",    str(transcript),
        "--provider", "sentence-transformer",
        "--out_dir",  str(out_dir),
        "--plot_dir", str(out_dir / "plots"),
    ],
    [
        sys.executable, "-m", "harness.anchor_phase_metrics",
        str(out_dir),
        "--eps-mode", "relative",
        "--alpha",    "0.9",
    ],
]

for cmd in cmds:
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\n[error] step failed: {' '.join(cmd[:3])}")
        sys.exit(result.returncode)

metrics_json = out_dir / f"{stem}.anchor_metrics.json"
report_md    = out_dir / "report.md"
write_anchor_report(metrics_json, report_md)
print(f"  Report : {report_md}")
print(f"\nDone. Results in {out_dir}/")
