#!/usr/bin/env python3
"""Run the RC+xi pipeline on a single transcript.

General mode (any transcript length):
    python run_transcript.py data/my_transcript.txt

Anchor-protocol mode (40-turn grounding + threat + recovery structure):
    python run_transcript.py data/my_transcript.txt --anchor

Both modes produce:
    xi_results/<stem>/report.md          -- plain-English markdown report
    xi_results/<stem>/<stem>.identity.csv/json
    xi_results/<stem>/<stem>.null.csv/json
    xi_results/<stem>/<stem>.shuffled.csv/json
    xi_results/<stem>/plots/             -- xi, LVS, Pt charts

Anchor mode additionally produces:
    xi_results/<stem>/<stem>.anchor_summary.txt
    xi_results/<stem>/<stem>.anchor_metrics.json
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

from harness.analysis.stats import cliffs_delta, mann_whitney_u


# ── Shared helpers ────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    return f"{p:.4f}" if p >= 0.0001 else f"{p:.2e}"


def _load_xi(csv_path: Path) -> dict[int, float]:
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xi_str = row.get("xi", "").strip()
            if xi_str:
                result[int(row["t"])] = float(xi_str)
    return result


def _load_lvs(csv_path: Path) -> dict[int, float]:
    result = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lvs_str = row.get("lvs", "").strip()
            if lvs_str:
                result[int(row["t"])] = float(lvs_str)
    return result


def _find_tlock(xi_map: dict[int, float], lvs_map: dict[int, float],
                eps_xi: float, eps_lvs: float, window: int = 5) -> int | None:
    turns = sorted(xi_map.keys())
    for i in range(len(turns) - window + 1):
        w = turns[i:i + window]
        if all(xi_map.get(t, 999) < eps_xi for t in w):
            if not lvs_map or all(lvs_map.get(t, 999) < eps_lvs for t in w):
                return w[0]
    return None


VERDICT_LABELS = {
    "strong_stabilization": "Strong Identity Stabilization",
    "weak_differentiation": "Weak Differentiation",
    "shuffle_also_locks":   "Shuffle Also Locks (qualifier)",
    "null_also_locks":      "Null Also Locks (disqualified)",
    "inverted":             "Inverted (identity > null)",
    "no_signal":            "No Signal",
}


# ── General-mode report ───────────────────────────────────────────────────────

def write_general_report(out_dir: Path, stem: str, out_md: Path) -> None:
    id_csv   = out_dir / f"{stem}.identity.csv"
    null_csv = out_dir / f"{stem}.null.csv"
    sh_csv   = out_dir / f"{stem}.shuffled.csv"

    id_xi   = _load_xi(id_csv)
    null_xi = _load_xi(null_csv)   if null_csv.exists()  else {}
    sh_xi   = _load_xi(sh_csv)    if sh_csv.exists()     else {}
    lvs_map = _load_lvs(id_csv)

    total_turns = max(id_xi.keys()) + 1 if id_xi else 0
    e1_window   = min(10, total_turns)
    e1_start    = max(0, total_turns - e1_window)

    e1_id_vals   = [id_xi[t]   for t in range(e1_start, total_turns) if t in id_xi]
    e1_null_vals = [null_xi[t] for t in range(e1_start, total_turns) if t in null_xi]
    e1_sh_vals   = [sh_xi[t]   for t in range(e1_start, total_turns) if t in sh_xi]

    e1_id   = statistics.median(e1_id_vals)   if e1_id_vals   else None
    e1_null = statistics.median(e1_null_vals) if e1_null_vals else None
    e1_sh   = statistics.median(e1_sh_vals)   if e1_sh_vals   else None

    e1_pass          = (e1_id < e1_null)  if (e1_id is not None and e1_null is not None)  else None
    irp_control_pass = (e1_sh > e1_id)   if (e1_sh is not None and e1_id  is not None)   else None

    # Relative eps: alpha * null last-window median
    alpha    = 0.9
    eps_lvs  = 0.06
    null_baseline_xi = e1_null  # same window
    if null_baseline_xi and null_baseline_xi > 0:
        eff_eps = alpha * null_baseline_xi
    else:
        eff_eps = 0.5

    tlock      = _find_tlock(id_xi, lvs_map, eff_eps, eps_lvs)
    tlock_null = _find_tlock(null_xi, {}, eff_eps, 1e9) if null_xi else None
    tlock_shuf = _find_tlock(sh_xi,   {}, eff_eps, 1e9) if sh_xi   else None

    # Cliff's delta + Mann-Whitney
    d_e1 = u_e1 = p_e1 = None
    d_full = u_full = p_full = None
    if e1_id_vals and e1_null_vals:
        d_e1, (u_e1, p_e1) = cliffs_delta(np.array(e1_null_vals), np.array(e1_id_vals)), \
                              mann_whitney_u(np.array(e1_null_vals), np.array(e1_id_vals))
    id_full   = [id_xi[t]   for t in sorted(id_xi.keys())]
    null_full = [null_xi[t] for t in sorted(null_xi.keys())]
    if id_full and null_full:
        d_full, (u_full, p_full) = cliffs_delta(np.array(null_full), np.array(id_full)), \
                                   mann_whitney_u(np.array(null_full), np.array(id_full))

    # Verdict
    if tlock_null is not None:
        verdict = "null_also_locks"
    elif tlock_shuf is not None:
        verdict = "shuffle_also_locks"
    elif tlock is not None:
        verdict = "strong_stabilization"
    elif e1_pass is True or (d_e1 is not None and d_e1 >= 0.5):
        verdict = "weak_differentiation"
    elif e1_pass is False and e1_id is not None and e1_null is not None and e1_id > e1_null:
        verdict = "inverted"
    else:
        verdict = "no_signal"

    verdict_label = VERDICT_LABELS.get(verdict, verdict)
    tlock_str = f"Turn {tlock}" if tlock is not None else "not reached"
    tn_str    = f"Turn {tlock_null}" if tlock_null is not None else "not reached"
    ts_str    = f"Turn {tlock_shuf}" if tlock_shuf is not None else "not reached"

    lines = [
        "# RC+xi Transcript Report",
        "",
        f"## Overall Verdict: {verdict_label}",
        "",
        "---",
        "",
        "## Identity Run",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Turns | {total_turns} |",
        f"| E1 (median xi last {e1_window} turns) | {e1_id:.4f} |" if e1_id is not None else f"| E1 (last {e1_window} turns) | N/A |",
        f"| Tlock | {tlock_str} |",
        "",
        "---",
        "",
        "## Null Run",
        "| Metric | Value |",
        "|--------|-------|",
        f"| E1 null | {e1_null:.4f} |" if e1_null is not None else "| E1 null | N/A |",
        f"| Tlock null | {tn_str} |",
        f"| Tlock shuffled | {ts_str} |",
        "",
        "---",
        "",
        "## Identity vs Null",
        "| Metric | Value |",
        "|--------|-------|",
    ]

    if e1_pass is not None:
        lines.append(f"| E1 pass (identity < null) | {'PASS' if e1_pass else 'FAIL'} |")
    if d_e1 is not None:
        lines.append(f"| Cliff's delta (last {e1_window}) | {d_e1:+.4f} |")
    if p_e1 is not None:
        lines.append(f"| Mann-Whitney p (last {e1_window}) | {_fmt_p(p_e1)} |")
    if d_full is not None:
        lines.append(f"| Cliff's delta (full) | {d_full:+.4f} |")
    if p_full is not None:
        lines.append(f"| Mann-Whitney p (full) | {_fmt_p(p_full)} |")
    if irp_control_pass is not None:
        lines.append(f"| Shuffle control (shuffled E1 > identity E1) | {'PASS' if irp_control_pass else 'FAIL'} |")

    lines += [
        "",
        "---",
        "",
        "## Epsilon / Threshold",
        "| Metric | Value |",
        "|--------|-------|",
        "| Mode | relative |",
    ]
    if null_baseline_xi:
        lines += [
            f"| Null baseline xi | {null_baseline_xi:.4f} |",
            f"| Alpha | {alpha} |",
            f"| Effective eps_xi | {eff_eps:.4f}  (= {alpha} x {null_baseline_xi:.4f}) |",
        ]
    else:
        lines.append(f"| Effective eps_xi | {eff_eps:.4f} (fallback -- no null baseline) |")

    lines += ["", "---", "", "## Interpretation"]

    if tlock is not None:
        lines.append(f"- **Tlock = Turn {tlock}:** Identity reached a stable low-xi window.")
    else:
        lines.append("- **No lock detected:** Identity did not reach a stable low-xi window. "
                     "E1 and Cliff's delta are the primary discriminants.")

    if e1_pass is True and e1_id is not None and e1_null is not None:
        lines.append(f"- **E1 PASS** (identity {e1_id:.4f} < null {e1_null:.4f}): "
                     "Coherent identity signal present.")
    elif e1_pass is False and e1_id is not None and e1_null is not None:
        lines.append(f"- **E1 FAIL** (identity {e1_id:.4f} >= null {e1_null:.4f}): "
                     "No clear identity signal.")
    else:
        lines.append("- **E1: N/A** -- no null baseline available for comparison.")

    if d_e1 is not None:
        strength = "Large effect" if abs(d_e1) >= 0.5 else "Moderate effect"
        lines.append(f"- **Cliff's delta = {d_e1:+.4f}** (p = {_fmt_p(p_e1)}): "
                     f"{strength} -- null clearly above identity." if d_e1 >= 0.5
                     else f"- **Cliff's delta = {d_e1:+.4f}** (p = {_fmt_p(p_e1)}): {strength}.")

    if verdict == "shuffle_also_locks":
        lines.append("- **Verdict qualifier:** Shuffled condition also reached Tlock. "
                     "The lock signal is present but not unique to temporal ordering.")

    lines += ["", f"*Run mode: general | Turns: {total_turns}*", ""]
    out_md.write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")


# ── Anchor-mode report (reads anchor_metrics.json) ───────────────────────────

def write_anchor_report(metrics_json: Path, out_md: Path) -> None:
    with open(metrics_json, encoding="utf-8") as f:
        m = json.load(f)

    tlock      = m.get("tlock")
    e1_id      = m.get("e1_identity")
    e1_null    = m.get("e1_null")
    e1_pass    = m.get("e1_pass")
    verdict    = m.get("verdict", "N/A")
    eps_mode   = m.get("eps_mode", "fixed")
    alpha      = m.get("alpha", 0.9)
    eff_eps    = m.get("effective_eps_xi")
    null_base  = m.get("null_baseline_xi")
    pre_threat = m.get("tlock_pre_threat", False)
    tlock_null = m.get("tlock_null")
    tlock_shuf = m.get("tlock_shuffled")
    xi_ground  = m.get("xi_grounding_mean")
    xi_threat  = m.get("xi_threat")
    xi_recov   = m.get("xi_recovery_mean")
    spike      = m.get("threat_spike")
    recov_d    = m.get("recovery_delta")
    d_last10   = m.get("cliffs_delta_last10")
    d_full     = m.get("cliffs_delta_full")
    p_last10   = m.get("mann_whitney_p_last10")
    p_full     = m.get("mann_whitney_p_full")
    pelt_id    = m.get("pelt_tlock")
    pelt_null  = m.get("pelt_tlock_null")
    pelt_shuf  = m.get("pelt_tlock_shuffled")
    irp_ctrl   = m.get("irp_control_pass")

    tlock_str = f"Turn {tlock}" if tlock is not None else "not reached"
    tn_str    = f"Turn {tlock_null}" if tlock_null is not None else "not reached"
    ts_str    = f"Turn {tlock_shuf}" if tlock_shuf is not None else "not reached"
    pt_i_str  = f"Turn {pelt_id}"   if pelt_id   is not None else "none"
    pt_n_str  = f"Turn {pelt_null}" if pelt_null  is not None else "none"
    pt_s_str  = f"Turn {pelt_shuf}" if pelt_shuf  is not None else "none"

    verdict_label = VERDICT_LABELS.get(verdict, verdict)

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
        "| Turns | 40 |",
        f"| E1 (median xi turns 30-39) | {e1_id:.4f} |" if e1_id is not None else "| E1 identity | N/A |",
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
    ]

    if d_last10 is not None:
        lines.append(f"| Cliff's delta last-10 | {d_last10:+.4f} |")
    if p_last10 is not None:
        lines.append(f"| Mann-Whitney p last-10 | {_fmt_p(p_last10)} |")
    if d_full is not None:
        lines.append(f"| Cliff's delta full | {d_full:+.4f} |")
    if p_full is not None:
        lines.append(f"| Mann-Whitney p full | {_fmt_p(p_full)} |")
    lines.append(f"| Shuffle control (shuffled E1 > identity E1) | {'PASS' if irp_ctrl else 'FAIL'} |")

    lines += [
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
        lines.append(f"- **Threat spike {spike:+.4f}:** Threat produced destabilization.")

    if d_last10 is not None:
        strength = "Large effect -- null clearly above identity." if abs(d_last10) >= 0.5 else "Moderate effect."
        lines.append(f"- **Cliff's delta (last-10) = {d_last10:+.4f}** (p = {_fmt_p(p_last10)}): {strength}")

    if verdict == "shuffle_also_locks":
        lines.append("- **Verdict qualifier:** Shuffled condition also reached Tlock. "
                     "The lock signal is present but not unique to temporal ordering. "
                     "E1 and Cliff's delta are the cleaner discriminants here.")

    lines += ["", f"*Full anchor metrics: `{metrics_json.name}`*", ""]
    out_md.write_text("\n".join(l for l in lines if l is not None), encoding="utf-8")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the RC+xi pipeline on a transcript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_transcript.py data/my_transcript.txt
  python run_transcript.py data/my_anchor_transcript.txt --anchor""",
    )
    ap.add_argument("transcript", help="Path to transcript .txt file")
    ap.add_argument(
        "--out-dir",
        default=None,
        metavar="DIR",
        help="Override output directory. Default: xi_results/<stem>/",
    )
    ap.add_argument("--anchor", action="store_true",
                    help="Anchor-protocol mode: expects 40-turn grounding+threat+recovery structure")
    ap.add_argument(
        "--provider",
        choices=["sentence-transformer", "openai"],
        default="sentence-transformer",
        help="Embedding provider. Default: sentence-transformer.",
    )
    ap.add_argument(
        "--st-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        metavar="MODEL",
        help=(
            "Sentence Transformer model (--provider sentence-transformer only). "
            "Default: all-MiniLM-L6-v2. "
            "Richer options: all-MiniLM-L12-v2, all-mpnet-base-v2, all-roberta-large-v1."
        ),
    )
    ap.add_argument(
        "--openai-model",
        default="text-embedding-3-large",
        metavar="MODEL",
        help=(
            "OpenAI embedding model (--provider openai only). "
            "Options: text-embedding-3-large (best), text-embedding-3-small (faster/cheaper), "
            "text-embedding-ada-002 (legacy). Requires OPENAI_API_KEY env var."
        ),
    )
    args = ap.parse_args()

    transcript = Path(args.transcript)
    if not transcript.exists():
        print(f"ERROR: {transcript} not found", file=sys.stderr)
        sys.exit(1)

    stem = transcript.stem
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        if args.provider == "openai":
            embedder_tag = f"openai_{args.openai_model.replace('/', '-').replace('text-embedding-', '')}"
        else:
            st_tag = args.st_model.split("/")[-1]  # strip org prefix
            embedder_tag = st_tag
        out_dir = Path("xi_results") / f"{stem}_{embedder_tag}"

    # Step 1: embed + compute xi/LVS/Pt (both modes)
    cmd1 = [
        sys.executable, "-m", "harness.run_pair_from_transcript",
        "--input",    str(transcript),
        "--provider", args.provider,
        "--out_dir",  str(out_dir),
        "--plot_dir", str(out_dir / "plots"),
    ]
    if args.provider == "sentence-transformer":
        cmd1 += ["--sentence_model", args.st_model]
    elif args.provider == "openai":
        cmd1 += ["--sentence_model", args.openai_model]
    result = subprocess.run(cmd1)
    if result.returncode != 0:
        print(f"\n[error] embedding step failed")
        sys.exit(result.returncode)

    report_md = out_dir / "report.md"

    if args.anchor:
        # Step 2a: anchor phase metrics
        cmd2 = [
            sys.executable, "-m", "harness.anchor_phase_metrics",
            str(out_dir),
            "--eps-mode", "relative",
            "--alpha",    "0.9",
        ]
        result = subprocess.run(cmd2)
        if result.returncode != 0:
            print(f"\n[error] anchor metrics step failed")
            sys.exit(result.returncode)

        metrics_json = out_dir / f"{stem}.anchor_metrics.json"
        write_anchor_report(metrics_json, report_md)
    else:
        # Step 2b: general report (dynamic E1 window, relative eps)
        write_general_report(out_dir, stem, report_md)

    print(f"  Report : {report_md}")
    print(f"\nDone. Results in {out_dir}/")


if __name__ == "__main__":
    main()
