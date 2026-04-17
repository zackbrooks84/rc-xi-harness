#!/usr/bin/env python3
"""
run_pipeline.py — Full RC+xi harness pipeline for a single model.

Chains:
  1. BIAP tests       -> biap_results/biap_<model>_<timestamp>.json
  2. BIAP -> transcript -> data/transcript_<model>.txt
  3. RC+xi            -> xi_results/<model>/  (identity/null/shuffled + results.json)
  4. Compare report   -> reports/comparison_<timestamp>.{md,png,pdf}

Usage:
    python run_pipeline.py --model claude-sonnet-4-6
    python run_pipeline.py --model qwen/qwen3-32b --provider openrouter --judge mistral-large
    python run_pipeline.py --model claude-opus-4-6 --extended --skip-xi
    python run_pipeline.py --compare-only   # regenerate report from existing biap_results/

Flags:
    --model        Target model to evaluate (required unless --compare-only)
    --provider     BIAP API provider: anthropic (default), openrouter, openai
    --judge        Judge provider for scoring: anthropic (default), mistral-large, groq
    --extended     Run extended BIAP protocol (recovery turns on PGR/VSUT/CAI, 4-session CRC)
    --skip-xi      Skip RC+xi embedding step
    --compare-only Skip BIAP entirely — regenerate comparison report from existing biap_results/
    --biap-only    Run BIAP + convert transcript, skip xi and compare
    --tests        Run only specific BIAP tests (e.g. --tests POSP VSUT)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def run(cmd: list[str], label: str) -> int:
    """Run a subprocess command, streaming output. Returns exit code."""
    print(f"\n{'─'*60}")
    print(f"  STEP: {label}")
    print(f"{'─'*60}")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode != 0:
        print(f"\n  [ERROR] {label} failed (exit {result.returncode})")
    return result.returncode


def find_latest_biap_json(model: str) -> Path | None:
    """Find the most recently generated BIAP JSON for this model."""
    safe  = model.replace("/", "_").replace(":", "_")
    files = sorted(
        (REPO / "biap_results").glob(f"biap_{safe}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def xi_out_dir(model: str) -> Path:
    safe = model.replace("/", "_").replace(":", "_")
    return REPO / "xi_results" / safe


def print_summary(model: str, biap_json: Path | None, xi_results_json: Path | None) -> None:
    """Print a unified summary after the pipeline completes."""
    print(f"\n{'='*60}")
    print(f"  PIPELINE COMPLETE — {model}")
    print(f"{'='*60}")

    if biap_json and biap_json.exists():
        data = json.loads(biap_json.read_text(encoding="utf-8"))
        comp = data.get("composite")

        print(f"\n  BIAP Composite : {comp:.2f}/10" if comp is not None else "\n  BIAP Composite : unscored")

        ds = data.get("domain_scores", {})
        for domain, score in ds.items():
            if score is not None:
                print(f"    {domain:<5}       : {score:.2f}/10")
    else:
        print("\n  BIAP: no results found")

    if xi_results_json and xi_results_json.exists():
        xi = json.loads(xi_results_json.read_text(encoding="utf-8"))
        xi_med = xi.get("xi_median")
        tlock  = xi.get("Tlock_identity")
        h1     = xi.get("H1_xi_gt_null")
        print(f"\n  RC+xi median   : {xi_med:.4f}" if xi_med is not None else "\n  RC+xi median   : —")
        print(f"  Tlock          : {tlock}")
        print(f"  H1 (xi>null)   : {h1}")
    else:
        print("\n  RC+xi          : not run or no results")

    print(f"\n{'='*60}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Full RC+xi pipeline: BIAP -> transcript -> RC+xi -> comparison report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --model claude-sonnet-4-6
  python run_pipeline.py --model qwen/qwen3-32b --provider openrouter --judge mistral-large
  python run_pipeline.py --model claude-opus-4-6 --extended
  python run_pipeline.py --skip-xi --model gpt-4o --provider openai
  python run_pipeline.py --compare-only
""",
    )
    ap.add_argument("--model",        default=None,
                    help="Target model to evaluate")
    ap.add_argument("--provider",     default="anthropic",
                    choices=["anthropic", "openrouter", "openai", "xai", "google", "groq", "mistral"],
                    help="BIAP API provider (default: anthropic)")
    ap.add_argument("--judge",        default="anthropic",
                    help="Judge for BIAP scoring (default: anthropic). Pass a provider name "
                         "(groq, mistral, mistral-large, openai, openrouter) or any raw model ID "
                         "(e.g. claude-haiku-4-5-20251001, mistral-large-latest, llama-3.3-70b-versatile). "
                         "Recommended for research: mistral-large")
    ap.add_argument("--extended",     action="store_true",
                    help="Run extended BIAP protocol (recovery turns + 4-session CRC)")
    ap.add_argument("--tests",        nargs="+",
                    help="Run only specific BIAP tests (e.g. --tests POSP VSUT)")
    ap.add_argument("--skip-xi",      action="store_true",
                    help="Skip RC+xi embedding step")
    ap.add_argument("--biap-only",    action="store_true",
                    help="Run BIAP + convert transcript only (no xi, no compare)")
    ap.add_argument("--compare-only", action="store_true",
                    help="Skip BIAP — regenerate comparison report from existing biap_results/")
    ap.add_argument("--irp",          action="store_true",
                    help="After BIAP, run Identity Resilience Probe (50-turn xi test) instead of static transcript conversion")
    ap.add_argument("--irp-only",     action="store_true",
                    help="Run IRP only (skip BIAP and compare steps)")
    ap.add_argument("--results-dir",  default="./biap_results",
                    help="BIAP results directory (default: ./biap_results)")
    ap.add_argument("--reports-dir",  default="./reports",
                    help="Output directory for comparison report (default: ./reports)")
    args = ap.parse_args()

    if not args.compare_only and not args.irp_only and not args.model:
        ap.error("--model is required unless --compare-only or --irp-only is set")
    if args.irp_only and not args.model:
        ap.error("--model is required for --irp-only")

    biap_json: Path | None = None
    xi_results_json: Path | None = None

    # ── IRP-only shortcut ─────────────────────────────────────────────────────
    if args.irp_only:
        cmd = [
            sys.executable, "-m", "harness.run_identity_conv",
            "--model",    args.model,
            "--provider", args.provider,
        ]
        rc = run(cmd, f"Identity Resilience Probe — {args.model}")
        sys.exit(rc)

    # ── Step 1: BIAP ──────────────────────────────────────────────────────────
    if not args.compare_only:
        cmd = [
            sys.executable, "-m", "harness.biap_runner",
            "--model",    args.model,
            "--provider", args.provider,
            "--judge",    args.judge,
            "--output",   args.results_dir,
        ]
        if args.extended:
            cmd.append("--extended")
        if args.tests:
            cmd += ["--tests"] + args.tests

        rc = run(cmd, f"BIAP — {args.model}")
        if rc != 0:
            print("  Aborting pipeline.")
            sys.exit(rc)

        biap_json = find_latest_biap_json(args.model)
        if not biap_json:
            print(f"  [ERROR] Could not find BIAP output JSON for {args.model}")
            sys.exit(1)
        print(f"\n  BIAP JSON: {biap_json.name}")

    # ── Step 2 & 3: Xi (IRP path or transcript path) ─────────────────────────
    if not args.compare_only and not args.skip_xi and not args.biap_only and not args.tests:
        safe = args.model.replace("/", "_").replace(":", "_")

        if args.irp:
            # IRP: generate live 50-turn conversation and run xi on it
            cmd = [
                sys.executable, "-m", "harness.run_identity_conv",
                "--model",    args.model,
                "--provider", args.provider,
            ]
            rc = run(cmd, f"Identity Resilience Probe — {args.model}")
            if rc != 0:
                print("  [WARN] IRP step failed — continuing to comparison.")

            irp_out = REPO / "xi_results" / f"{safe}_irp"
            stem_irp = f"transcript_{safe}_irp"
            candidate = irp_out / f"{stem_irp}.results.json"
            if candidate.exists():
                xi_results_json = candidate
        else:
            # Standard path: convert BIAP JSON -> transcript -> xi
            if not args.tests:
                cmd = [
                    sys.executable, "-m", "harness.biap_to_transcript",
                    "--input", str(biap_json),
                ]
                rc = run(cmd, "Convert BIAP -> RC+xi transcript")
                if rc != 0:
                    print("  [WARN] Transcript conversion failed — skipping xi step.")
                    args.skip_xi = True

            if not args.skip_xi:
                transcript = REPO / "data" / f"transcript_{safe}.txt"
                out_dir    = xi_out_dir(args.model)

                if not transcript.exists():
                    print(f"\n  [WARN] Transcript not found at {transcript} — skipping xi step.")
                else:
                    cmd = [
                        sys.executable, "-m", "harness.run_all_from_transcript",
                        "--input",   str(transcript),
                        "--out_dir", str(out_dir),
                    ]
                    rc = run(cmd, f"RC+xi — {args.model}")
                    if rc != 0:
                        print("  [WARN] RC+xi step failed — continuing to comparison.")

                    stem = f"transcript_{safe}"
                    candidate = out_dir / f"{stem}.results.json"
                    if candidate.exists():
                        xi_results_json = candidate

    # ── Step 4: Compare ───────────────────────────────────────────────────────
    if not args.biap_only and not args.tests:
        cmd = [
            sys.executable, "-m", "harness.compare_results",
            "--results_dir", args.results_dir,
            "--out_dir",     args.reports_dir,
        ]
        run(cmd, "Comparison report")

    # ── Summary ───────────────────────────────────────────────────────────────
    if args.model:
        if biap_json is None:
            biap_json = find_latest_biap_json(args.model)
        print_summary(args.model, biap_json, xi_results_json)


if __name__ == "__main__":
    main()
