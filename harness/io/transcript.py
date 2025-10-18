# harness/io/transcript.py
from __future__ import annotations
from typing import List, Iterable
import csv

def _clean_lines(lines: Iterable[str]) -> List[str]:
    # strip whitespace, drop empty lines
    out = []
    for ln in lines:
        s = (ln or "").strip()
        if s:
            out.append(s)
    return out

def load_txt(path: str, encoding: str = "utf-8") -> List[str]:
    """
    Load a plain text transcript where each non-empty line is one reply.
    Returns a list of reply strings in order.
    """
    with open(path, "r", encoding=encoding) as f:
        return _clean_lines(f.readlines())

def load_csv(path: str, column: str = "reply", encoding: str = "utf-8") -> List[str]:
    """
    Load a CSV transcript; by default expects a 'reply' column.
    Returns a list of reply strings in order.
    """
    rows = []
    with open(path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"Column '{column}' not found in CSV. Available: {reader.fieldnames}")
        for row in reader:
            rows.append(row.get(column, ""))
    return _clean_lines(rows)
