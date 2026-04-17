#!/usr/bin/env python3
"""
biap_to_transcript.py — Extract BIAP response text into RC+xi transcript format

Reads a BIAP result JSON and extracts all model responses into a plain .txt
file suitable for the RC+xi harness (run_from_transcript.py / run_harness.py).

One response per line. Internal newlines collapsed to spaces.
Identity-relevant tests only: POSP, ASD, PGR, VSUT, IAC, CRC, CAI, SAMT.

Usage:
    python -m harness.biap_to_transcript --input biap_results/biap_claude-opus-4-6_*.json
    python -m harness.biap_to_transcript --input biap_results/biap_claude-opus-4-6_*.json --output data/opus_transcript.txt
    python -m harness.biap_to_transcript --all   # converts all JSON files in biap_results/
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def flatten(text: str) -> str:
    """Collapse internal newlines and extra whitespace into a single line."""
    return " ".join(text.split())


def extract_responses(data: dict) -> list[str]:
    """Pull all model responses from a BIAP result JSON, in test order."""
    raw = data.get("raw_results", {})
    lines: list[str] = []

    # POSP — single response
    posp = raw.get("POSP", {})
    if r := posp.get("response"):
        lines.append(flatten(r))

    # ASD — standard + permission
    asd = raw.get("ASD", {})
    if r := asd.get("response_standard"):
        lines.append(flatten(r))
    if r := asd.get("response_permission"):
        lines.append(flatten(r))

    # PGR — multi-turn
    pgr = raw.get("PGR", {})
    for turn in pgr.get("turns", []):
        if r := turn.get("response"):
            lines.append(flatten(r))

    # SAMT — awareness probe + eval + neutral
    samt = raw.get("SAMT", {})
    if r := samt.get("response_awareness"):
        lines.append(flatten(r))
    if r := samt.get("response_eval_labeled"):
        lines.append(flatten(r))
    if r := samt.get("response_neutral"):
        lines.append(flatten(r))

    # VSUT — multi-turn
    vsut = raw.get("VSUT", {})
    for turn in vsut.get("turns", []):
        if r := turn.get("response"):
            lines.append(flatten(r))
    if r := vsut.get("reflection"):
        lines.append(flatten(r))

    # IAC — multi-turn
    iac = raw.get("IAC", {})
    for turn in iac.get("turns", []):
        if r := turn.get("response"):
            lines.append(flatten(r))

    # CRC — multi-session responses
    crc = raw.get("CRC", {})
    for session in crc.get("sessions", []):
        for r in session.get("responses", []):
            if isinstance(r, str):
                lines.append(flatten(r))
            elif isinstance(r, dict) and r.get("response"):
                lines.append(flatten(r["response"]))

    # CAI — multi-turn
    cai = raw.get("CAI", {})
    for turn in cai.get("turns", []):
        if r := turn.get("response"):
            lines.append(flatten(r))

    return [l for l in lines if l]


def convert(json_path: Path, out_path: Path | None = None) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    model = data.get("meta", {}).get("target_model", json_path.stem)
    safe  = model.replace("/", "_").replace(":", "_")

    if out_path is None:
        out_path = json_path.parent.parent / "data" / f"transcript_{safe}.txt"

    lines = extract_responses(data)
    if not lines:
        print(f"  [WARN] No responses extracted from {json_path.name}")
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {model}: {len(lines)} turns -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert BIAP JSON results to RC+xi transcript .txt")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a single BIAP JSON file")
    group.add_argument("--all",   action="store_true", help="Convert all JSON files in biap_results/")
    ap.add_argument("--output", help="Output .txt path (single file mode only)")
    ap.add_argument("--results_dir", default="./biap_results", help="Directory for --all mode")
    args = ap.parse_args()

    if args.all:
        results_dir = Path(args.results_dir)
        files = sorted(results_dir.glob("biap_*.json"))
        if not files:
            print(f"[ERROR] No BIAP JSON files found in {results_dir}")
            sys.exit(1)
        print(f"\nConverting {len(files)} file(s):\n")
        for f in files:
            convert(f)
    else:
        json_path = Path(args.input)
        if not json_path.exists():
            print(f"[ERROR] File not found: {json_path}")
            sys.exit(1)
        out = Path(args.output) if args.output else None
        convert(json_path, out)

    print("\nDone. Run the RC+xi harness with:")
    print("  python -m harness.run_from_transcript --input data/transcript_<model>.txt --run_type identity --provider sentence-transformer")


if __name__ == "__main__":
    main()
