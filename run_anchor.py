#!/usr/bin/env python3
"""Anchor-protocol shortcut -- equivalent to: python run_transcript.py <transcript> --anchor

Use run_transcript.py directly for full control:
    python run_transcript.py data/my_anchor_transcript.txt --anchor  # anchor protocol
    python run_transcript.py data/my_transcript.txt                  # general mode
"""
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python run_anchor.py <transcript_path>")
    print("       python run_transcript.py <transcript_path> --anchor  (equivalent)")
    sys.exit(1)

subprocess.run([sys.executable, "run_transcript.py", sys.argv[1], "--anchor"])
