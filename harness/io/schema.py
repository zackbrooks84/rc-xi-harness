# harness/io/schema.py
from __future__ import annotations
import csv
from typing import Iterable, Dict, Any

COLUMNS = ["t", "xi", "lvs", "Pt", "ewma_xi", "run_type", "provider"]

def write_rows(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in COLUMNS})
