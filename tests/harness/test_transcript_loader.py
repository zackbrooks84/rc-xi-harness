# tests/harness/test_transcript_loader.py
from __future__ import annotations
from harness.io.transcript import load_txt, load_csv
import csv

def test_load_txt(tmp_path):
    p = tmp_path / "t.txt"
    p.write_text("first reply\n\n second reply \n third reply\n", encoding="utf-8")
    out = load_txt(str(p))
    assert out == ["first reply", "second reply", "third reply"]

def test_load_csv_default_column(tmp_path):
    p = tmp_path / "t.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reply", "meta"])
        w.writeheader()
        w.writerow({"reply": "r1", "meta": "a"})
        w.writerow({"reply": "", "meta": "b"})     # empty line gets dropped
        w.writerow({"reply": " r3 ", "meta": "c"})  # trims whitespace
    out = load_csv(str(p))  # default column='reply'
    assert out == ["r1", "r3"]

def test_load_csv_custom_column(tmp_path):
    p = tmp_path / "u.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utterance"])
        w.writeheader()
        for x in ["a", "b", "c"]:
            w.writerow({"utterance": x})
    out = load_csv(str(p), column="utterance")
    assert out == ["a", "b", "c"]
