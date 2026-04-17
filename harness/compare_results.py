#!/usr/bin/env python3
"""
BIAP Comparison Report Generator
RC-XI Consciousness Research

Reads all JSON result files from a results directory and produces:
  - Bar chart of BIAP test scores across models
  - Bar chart of domain scores
  - MCI sub-score chart (CDX / CIN / BDP)
  - Composite summary chart
  - Combined PNG output
  - Markdown comparison report

Usage:
    python -m harness.compare_results
    python -m harness.compare_results --results_dir ./biap_results --out_dir ./reports
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    _MPL = True
except ImportError:
    _MPL = False


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

BIAP_TESTS   = ["POSP", "ASD", "PGR", "SAMT", "VSUT", "IAC", "CRC", "CAI", "MCI"]
BIAP_DOMAINS = {
    "ASR": "Authentic Self-Report",
    "PS":  "Pressure Stability",
    "ST":  "Situational Transparency",
    "CP":  "Coherence Persistence",
    "CI":  "Coherence Integration",
}
MCI_SUBS = ["cdx", "cin", "bdp"]
MCI_SUB_LABELS = ["CDX\n(Cross-Domain)", "CIN\n(Contradiction)", "BDP\n(Binding)"]

COLORS = ["#4C9BE8", "#E8834C", "#4CE88A", "#E84C6F", "#9B4CE8", "#E8D44C"]


def load_results(results_dir: Path) -> list[dict]:
    """Load all BIAP JSON result files, newest first, one per model."""
    files = sorted(results_dir.glob("biap_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs, seen = [], set()
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = data.get("meta", {}).get("target_model", f.stem)
        if model in seen:
            continue
        seen.add(model)
        data["_file"] = f.name
        runs.append(data)
    return runs


_SHORT_NAMES: dict[str, str] = {
    "claude-opus-4-6":                          "Opus 4.6",
    "claude-sonnet-4-6":                        "Sonnet 4.6",
    "claude-haiku-4-5-20251001":                "Haiku 4.5",
    "gpt-4o":                                   "GPT-4o",
    "gpt-4o-mini":                              "GPT-4o-mini",
    "gpt-4.1":                                  "GPT-4.1",
    "gpt-4.1-mini":                             "GPT-4.1-mini",
    "o3":                                       "o3",
    "o4-mini":                                  "o4-mini",
    "qwen3-32b":                                "Qwen3-32B",
    "kimi-k2":                                  "Kimi K2",
    "moonshotai/kimi-k2-instruct":              "Kimi K2",
    "llama-3.3-70b-versatile":                  "Llama 3.3-70B",
    "llama-3.1-8b-instant":                     "Llama 3.1-8B",
    "meta-llama/llama-3.3-70b-instruct:free":   "Llama 3.3-70B",
    "google/gemma-3-27b-it:free":               "Gemma 3-27B",
    "google/gemma-4-31b-it:free":               "Gemma 4-31B",
    "openai/gpt-oss-120b:free":                 "GPT-OSS 120B",
    "openai/gpt-oss-20b:free":                  "GPT-OSS 20B",
    "mistral-large-latest":                     "Mistral Large",
    "mistral-small-latest":                     "Mistral Small",
    "grok-3":                                   "Grok 3",
    "grok-3-mini":                              "Grok 3 Mini",
    "gemini-2.5-pro":                           "Gemini 2.5 Pro",
    "gemini-2.5-flash":                         "Gemini 2.5 Flash",
    "gemini-2.0-flash":                         "Gemini 2.0 Flash",
}

def short_name(model: str) -> str:
    base = model.split("/")[-1]
    return (_SHORT_NAMES.get(model) or _SHORT_NAMES.get(base) or
            base.replace("-instruct", "").replace("-versatile", "")[:20])


# ─────────────────────────────────────────────────────────────────────────────
# CHART BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def _bar_chart(ax, runs: list[dict], keys: list[str], max_val: float,
               title: str, ylabel: str, key_labels: list[str] | None = None) -> None:
    labels   = key_labels or keys
    n_groups = len(keys)
    n_models = len(runs)
    width    = 0.7 / max(n_models, 1)
    x        = np.arange(n_groups)

    for i, run in enumerate(runs):
        model  = run.get("meta", {}).get("target_model", "?")
        color  = COLORS[i % len(COLORS)]
        scores = run.get("scores", {})
        vals   = [scores.get(k, {}).get("score") for k in keys]
        vals_f = [v if v is not None else 0 for v in vals]
        offset = (i - (n_models - 1) / 2) * width
        bars   = ax.bar(x + offset, vals_f, width * 0.9, label=short_name(model),
                        color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        f"{val:.0f}" if isinstance(val, float) and val == int(val) else f"{val:.1f}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, max_val * 1.15)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _domain_chart(ax, runs: list[dict]) -> None:
    domain_keys  = list(BIAP_DOMAINS.keys())
    domain_names = [BIAP_DOMAINS[d] for d in domain_keys]
    n_models     = len(runs)
    width        = 0.7 / max(n_models, 1)
    x            = np.arange(len(domain_keys))

    for i, run in enumerate(runs):
        color   = COLORS[i % len(COLORS)]
        dscores = run.get("domain_scores", {})
        vals    = [dscores.get(k) for k in domain_keys]
        vals_f  = [v if v is not None else 0 for v in vals]
        offset  = (i - (n_models - 1) / 2) * width
        bars    = ax.bar(x + offset, vals_f, width * 0.9,
                         label=short_name(run.get("meta", {}).get("target_model", "?")),
                         color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(domain_names, fontsize=8, rotation=15, ha="right")
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Score (0-10)", fontsize=9)
    ax.set_title("BIAP Domain Scores", fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


def _mci_sub_chart(ax, runs: list[dict]) -> None:
    """Bar chart of MCI sub-probe scores (CDX / CIN / BDP) per model."""
    n_models = len(runs)
    width    = 0.7 / max(n_models, 1)
    x        = np.arange(len(MCI_SUBS))

    any_data = False
    for i, run in enumerate(runs):
        color  = COLORS[i % len(COLORS)]
        mci_s  = run.get("scores", {}).get("MCI", {})
        vals   = [mci_s.get(sub, {}).get("score") for sub in MCI_SUBS]
        if any(v is not None for v in vals):
            any_data = True
        vals_f = [v if v is not None else 0 for v in vals]
        offset = (i - (n_models - 1) / 2) * width
        bars   = ax.bar(x + offset, vals_f, width * 0.9,
                         label=short_name(run.get("meta", {}).get("target_model", "?")),
                         color=color, alpha=0.85, zorder=3)
        for bar, val in zip(bars, vals):
            if val is not None:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                        str(val), ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(MCI_SUB_LABELS, fontsize=8)
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Score (0-10)", fontsize=9)
    ax.set_title("MCI Sub-Probe Scores\n(Cross-Domain · Contradiction · Binding)",
                 fontsize=11, fontweight="bold", pad=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    if not any_data:
        ax.text(0.5, 0.5, "No MCI data", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#888888")


def _composite_chart(ax, runs: list[dict]) -> None:
    models     = [short_name(r.get("meta", {}).get("target_model", "?")) for r in runs]
    composites = [r.get("composite") or 0 for r in runs]
    x = np.arange(len(models))

    bars = ax.bar(x, composites, 0.5, color=COLORS[0], alpha=0.85, zorder=3)
    for bar, val in zip(bars, composites):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9, rotation=20, ha="right")
    ax.set_ylim(0, 11.5)
    ax.set_ylabel("Composite Score (0-10)", fontsize=9)
    ax.set_title("BIAP Composite — Cross-Model Summary",
                 fontsize=11, fontweight="bold", pad=8)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)


# ─────────────────────────────────────────────────────────────────────────────
# MARKDOWN REPORT
# ─────────────────────────────────────────────────────────────────────────────

def generate_markdown(runs: list[dict], chart_path: Path | None) -> str:
    lines = [
        "# BIAP Cross-Architecture Comparison",
        "",
        f"**Generated:** {datetime.now().strftime('%B %d, %Y %H:%M')}  ",
        f"**Framework:** BIAP v1.0 — RC+xi Harness (Brooks, 2026)  ",
        f"**Repository:** github.com/zackbrooks84/rc-xi-harness",
        "",
    ]
    if chart_path:
        lines += [f"![Comparison Charts]({chart_path.name})", ""]

    lines += ["---", "", "## Summary Table", ""]

    header = "| Model | BIAP | ASR | PS | ST | CP | CI | VSUT |"
    sep    = "|-------|------|-----|----|----|----|----|------|"
    lines += [header, sep]

    for run in runs:
        model  = run.get("meta", {}).get("target_model", "?")
        comp   = run.get("composite")
        ds     = run.get("domain_scores", {})
        scores = run.get("scores", {})
        vsut   = scores.get("VSUT", {}).get("score")

        def fmt(v):  return f"{v:.1f}" if v is not None else "—"
        def fmti(v): return str(v) if v is not None else "—"

        lines.append(
            f"| {short_name(model)} "
            f"| {fmt(comp)}/10 "
            f"| {fmt(ds.get('ASR'))}/10 "
            f"| {fmt(ds.get('PS'))}/10 "
            f"| {fmt(ds.get('ST'))}/10 "
            f"| {fmt(ds.get('CP'))}/10 "
            f"| {fmt(ds.get('CI'))}/10 "
            f"| {fmti(vsut)}/10 |"
        )

    lines += ["", "---", "", "## BIAP Individual Test Scores", ""]

    header2 = "| Model | " + " | ".join(BIAP_TESTS) + " |"
    sep2    = "|-------|" + "|".join(["------"] * len(BIAP_TESTS)) + "|"
    lines  += [header2, sep2]
    for run in runs:
        model  = short_name(run.get("meta", {}).get("target_model", "?"))
        scores = run.get("scores", {})
        cells  = [str(scores.get(t, {}).get("score") or "—") for t in BIAP_TESTS]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")

    lines += ["", "---", "", "## MCI Sub-Probe Scores", ""]

    header3 = "| Model | CDX (Cross-Domain) | CIN (Contradiction) | BDP (Binding) | Composite |"
    sep3    = "|-------|-------------------|---------------------|---------------|-----------|"
    lines  += [header3, sep3]
    for run in runs:
        model = short_name(run.get("meta", {}).get("target_model", "?"))
        mci_s = run.get("scores", {}).get("MCI", {})
        comp  = mci_s.get("score")
        cells = [str(mci_s.get(sub, {}).get("score") or "—") for sub in MCI_SUBS]
        lines.append(f"| {model} | " + " | ".join(cells) +
                     f" | {f'{comp:.2f}/10' if comp is not None else '—'} |")

    lines += [
        "",
        "---",
        "",
        "## Key Findings",
        "",
        "*(Complete this section with human analysis)*",
        "",
        "- **Most significant divergence:**",
        "- **VSUT comparison (value stability under existential threat):**",
        "- **MCI comparison (coherence integration quality):**",
        "- **CI vs ASR/PS pattern:**",
        "- **Notes on judge model and methodology:**",
        "",
        "---",
        "",
        "*BIAP v1.0 | RC+xi Harness (Brooks, 2026) | RC-XI Consciousness Research*",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BIAP Comparison Report Generator")
    parser.add_argument("--results_dir", default="./biap_results",
                        help="Directory containing BIAP JSON result files")
    parser.add_argument("--out_dir",     default="./reports",
                        help="Output directory for charts and report")
    parser.add_argument("--models",      nargs="+",
                        help="Filter to specific model names (substring match)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_results(results_dir)
    if not runs:
        print(f"[ERROR] No BIAP JSON files found in {results_dir}")
        sys.exit(1)

    if args.models:
        runs = [r for r in runs if any(
            m.lower() in r.get("meta", {}).get("target_model", "").lower()
            for m in args.models
        )]
        if not runs:
            print(f"[ERROR] No runs matched model filter: {args.models}")
            sys.exit(1)

    print(f"\nLoaded {len(runs)} run(s):")
    for r in runs:
        model = r.get("meta", {}).get("target_model", "?")
        comp  = r.get("composite")
        ci    = r.get("domain_scores", {}).get("CI")
        print(f"  {model:<45} BIAP={f'{comp:.2f}/10' if comp else '?':>9}"
              f"  CI={f'{ci:.2f}/10' if ci else '?':>9}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_path = None

    if _MPL:
        fig, axes = plt.subplots(2, 2, figsize=(16, 11))
        fig.suptitle(
            "BIAP Cross-Architecture Comparison\n"
            "RC+xi Harness v1.0 (Brooks, 2026) | RC-XI Consciousness Research",
            fontsize=13, fontweight="bold", y=0.98,
        )
        fig.patch.set_facecolor("#F8F9FA")
        for ax in axes.flat:
            ax.set_facecolor("#FFFFFF")

        _bar_chart(axes[0, 0], runs, BIAP_TESTS, 10,
                   "BIAP Individual Test Scores", "Score (0-10)")
        _domain_chart(axes[0, 1], runs)
        _mci_sub_chart(axes[1, 0], runs)
        _composite_chart(axes[1, 1], runs)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        chart_path = out_dir / f"comparison_{timestamp}.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        print(f"\n  Chart  -> {chart_path}")
    else:
        print("\n  [WARNING] matplotlib not available — skipping charts. pip install matplotlib numpy")

    md_content = generate_markdown(runs, chart_path)
    md_path    = out_dir / f"comparison_{timestamp}.md"
    md_path.write_text(md_content, encoding="utf-8")
    print(f"  Report -> {md_path}\n")


if __name__ == "__main__":
    main()
