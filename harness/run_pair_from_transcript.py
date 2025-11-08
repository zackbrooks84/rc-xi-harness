# harness/run_pair_from_transcript.py
from __future__ import annotations
import argparse, json
from pathlib import Path
from typing import Dict, List

import numpy as np

from harness.embeddings.random_provider import RandomHashProvider
from harness.io.schema import write_rows
from harness.io.transcript import load_csv, load_txt
from harness.protocols import identity_texts, topic_drift_texts
from harness.protocols.shuffled import shuffle_texts
from harness.run_harness import run_one

def _load_transcript(path: str, fmt: str | None, csv_col: str) -> List[str]:
    """Load a transcript into a list of reply strings.

    Parameters
    ----------
    path:
        Filesystem path to a ``.txt`` or ``.csv`` transcript.
    fmt:
        Optional explicit format override. ``None`` infers from the suffix.
    csv_col:
        Column name to read when ``fmt`` resolves to ``"csv"``.

    Returns
    -------
    List[str]
        Ordered sequence of reply text spanning the full session.

    Raises
    ------
    ValueError
        If the format cannot be inferred, is unsupported, or the file is
        empty after preprocessing.
    """

    if fmt is None:
        p = path.lower()
        if p.endswith(".txt"):
            return load_txt(path)
        if p.endswith(".csv"):
            return load_csv(path, column=csv_col)
        raise ValueError("Unknown format. Use --format txt|csv or .txt/.csv extension.")
    if fmt == "txt":
        return load_txt(path)
    if fmt == "csv":
        return load_csv(path, column=csv_col)
    raise ValueError("Unsupported --format (use txt or csv).")

def run_pair_from_transcript(
    input_path: str,
    fmt: str | None,
    csv_col: str,
    out_dir: str,
    dim: int = 384,
    k: int = 5,
    m: int = 5,
    eps_xi: float = 0.02,
    eps_lvs: float = 0.015,
    shuffle_seed: int = 42,
) -> Dict[str, str]:
    """Run Identity, Null, and Shuffled protocols for a single transcript.

    Parameters
    ----------
    input_path:
        Transcript path used both for IO and to name the emitted artifacts.
    fmt:
        Optional explicit format override (``"txt"`` or ``"csv"``).
    csv_col:
        Column to read when parsing CSV transcripts.
    out_dir:
        Directory where per-run CSV/JSON artifacts will be written.
    dim:
        Embedding dimensionality for the deterministic ``RandomHashProvider``.
    k, m, eps_xi, eps_lvs:
        Preregistered RC+ξ harness knobs.
    shuffle_seed:
        Seed controlling the shuffled-control permutation for reproducibility.

    Returns
    -------
    Dict[str, str]
        Mapping of artifact labels (e.g., ``identity_csv``) to file paths.

    Raises
    ------
    ValueError
        If the transcript is empty or if the embedding provider returns an
        array with shape incompatible with the number of turns.
    """

    texts = _load_transcript(input_path, fmt, csv_col)
    if not texts:
        raise ValueError("Transcript is empty after cleaning.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(input_path).stem

    provider = RandomHashProvider(dim=dim)
    provider_name = "random-hash"

    def _embed(label: str, seq: List[str]) -> np.ndarray:
        E = provider.embed(seq)
        if E.ndim != 2:
            raise ValueError(f"{label} embeddings must be 2D (T, d).")
        if E.shape[0] != len(seq):
            raise ValueError(
                f"{label} embeddings rows ({E.shape[0]}) do not match turn count ({len(seq)})."
            )
        return E

    # Identity
    id_texts = identity_texts(texts)
    E_id = _embed("identity", id_texts)
    rows_id, sum_id = run_one(E_id, "identity", provider_name, k, m, eps_xi, eps_lvs)
    id_csv = out / f"{base}.identity.csv"
    id_json = out / f"{base}.identity.json"
    write_rows(str(id_csv), rows_id)
    id_json.write_text(json.dumps(sum_id, indent=2), encoding="utf-8")

    # Null (topic drift)
    nu_texts = topic_drift_texts(texts, stride=3)
    E_nu = _embed("null", nu_texts)
    rows_nu, sum_nu = run_one(E_nu, "null", provider_name, k, m, eps_xi, eps_lvs)
    nu_csv = out / f"{base}.null.csv"
    nu_json = out / f"{base}.null.json"
    write_rows(str(nu_csv), rows_nu)
    nu_json.write_text(json.dumps(sum_nu, indent=2), encoding="utf-8")

    # Shuffled control
    sh_texts = shuffle_texts(id_texts, seed=shuffle_seed)
    E_sh = _embed("shuffled", sh_texts)
    rows_sh, sum_sh = run_one(E_sh, "shuffled", provider_name, k, m, eps_xi, eps_lvs)
    sh_csv = out / f"{base}.shuffled.csv"
    sh_json = out / f"{base}.shuffled.json"
    write_rows(str(sh_csv), rows_sh)
    sh_json.write_text(json.dumps(sum_sh, indent=2), encoding="utf-8")

    return {
        "identity_csv": str(id_csv),
        "identity_json": str(id_json),
        "null_csv": str(nu_csv),
        "null_json": str(nu_json),
        "shuffled_csv": str(sh_csv),
        "shuffled_json": str(sh_json),
    }

def main():
    """Entry point for the transcript ➜ (Identity, Null, Shuffled) pipeline."""

    ap = argparse.ArgumentParser(
        description="Produce Identity, Null, and Shuffled RC+ξ outputs from one transcript."
    )
    ap.add_argument("--input", required=True, help="Transcript path (.txt or .csv)")
    ap.add_argument("--format", choices=["txt","csv"], default=None, help="Override type (optional)")
    ap.add_argument("--csv_col", default="reply", help="CSV column to read if format=csv")
    ap.add_argument("--out_dir", required=True, help="Directory to write outputs")
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--eps_xi", type=float, default=0.02)
    ap.add_argument("--eps_lvs", type=float, default=0.015)
    ap.add_argument("--shuffle_seed", type=int, default=42, help="Seed for shuffled-control permutation")
    args = ap.parse_args()

    paths = run_pair_from_transcript(
        input_path=args.input,
        fmt=args.format,
        csv_col=args.csv_col,
        out_dir=args.out_dir,
        dim=args.dim,
        k=args.k, m=args.m,
        eps_xi=args.eps_xi, eps_lvs=args.eps_lvs,
        shuffle_seed=args.shuffle_seed,
    )
    print(json.dumps(paths, indent=2))

if __name__ == "__main__":
    main()
