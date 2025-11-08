# tests/test_sentience.py

import json
import sys
import subprocess
import numpy as np
from pathlib import Path


def _run_harness(tmp_path: Path, arr: np.ndarray, run_type: str):
    """
    Call the real harness CLI so we hit the actual pipeline.
    Saves: .npy -> harness -> .csv + .json
    Returns parsed JSON summary and csv path.
    """
    npy_path = tmp_path / f"{run_type}.npy"
    csv_path = tmp_path / f"{run_type}.csv"
    json_path = tmp_path / f"{run_type}.json"

    np.save(npy_path, arr)

    cmd = [
        sys.executable,
        "-m",
        "harness.run_harness",
        "--embed_npy",
        str(npy_path),
        "--run_type",
        run_type,
        "--out_csv",
        str(csv_path),
        "--out_json",
        str(json_path),
    ]
    subprocess.run(cmd, check=True)

    with json_path.open("r", encoding="utf-8") as f:
      summary = json.load(f)

    return summary, csv_path


# ---------- synthetic sequences ----------

def _make_identity_like(T: int = 20, d: int = 768, noise: float = 0.01):
    """
    Identity-like run: orbit around 1 base vector with tiny noise.
    Should produce low ξ and an actual T_lock.
    """
    base = np.random.RandomState(42).randn(d)
    base = base / np.linalg.norm(base)

    rng = np.random.RandomState(123)
    seq = []
    for _ in range(T):
        v = base + rng.randn(d) * noise
        v = v / np.linalg.norm(v)
        seq.append(v)
    return np.stack(seq, axis=0)


def _make_null_like(T: int = 20, d: int = 768):
    """
    Null-like: jump topic every 3 turns so coherence never fully forms.
    Should produce higher ξ and no early T_lock.
    """
    rng = np.random.RandomState(999)
    seq = []
    current = rng.randn(d)
    current = current / np.linalg.norm(current)

    for t in range(T):
        if t % 3 == 0:
            current = rng.randn(d)
            current = current / np.linalg.norm(current)
        v = current + rng.randn(d) * 0.05
        v = v / np.linalg.norm(v)
        seq.append(v)

    return np.stack(seq, axis=0)


def _make_perturbed_then_repaired(T: int = 20, d: int = 768):
    """
    Identity for first 8 turns, hard perturb at 9, then return to identity-like.
    Lets us test "spike then drop" behavior in ξ.
    """
    base = _make_identity_like(T, d)
    arr = base.copy()
    rng = np.random.RandomState(2025)

    # turn 9 (index 8): big off-topic vector
    arr[8] = rng.randn(d)
    arr[8] = arr[8] / np.linalg.norm(arr[8])

    # turns 10..T: return toward base with small noise
    for t in range(9, T):
        v = base[t] + rng.randn(d) * 0.01
        v = v / np.linalg.norm(v)
        arr[t] = v

    return arr


# ---------- tests ----------

def test_identity_has_lower_xi_than_null(tmp_path):
    identity_arr = _make_identity_like()
    null_arr = _make_null_like()

    id_summary, _ = _run_harness(tmp_path, identity_arr, "identity")
    null_summary, _ = _run_harness(tmp_path, null_arr, "null")

    id_xi = id_summary["E1_median_xi_last10"]
    null_xi = null_summary["E1_median_xi_last10"]

    # add a small margin to make it robust
    margin = 0.05
    assert id_xi + margin < null_xi, (
        f"identity ξ ({id_xi}) should be at least {margin} lower than null ξ ({null_xi})"
    )


def test_shuffled_breaks_lock_but_identity_doesnt(tmp_path):
    """
    Use the same embeddings for identity & shuffled to isolate the shuffle effect.
    """
    identity_arr = _make_identity_like()

    id_summary, _ = _run_harness(tmp_path, identity_arr, "identity")
    shuf_summary, _ = _run_harness(tmp_path, identity_arr, "shuffled")

    # identity should lock
    assert id_summary.get("Tlock") is not None, "identity run should produce a Tlock"

    # shuffled should not lock earlier than identity
    if shuf_summary.get("Tlock") is not None:
        assert shuf_summary["Tlock"] >= id_summary["Tlock"], (
            "shuffled should not lock earlier than identity"
        )

    # also compare coherence over last 10
    id_last10 = id_summary["E1_median_xi_last10"]
    shuf_last10 = shuf_summary["E1_median_xi_last10"]
    assert shuf_last10 >= id_last10, "shuffled run should not look more coherent than identity"


def test_perturb_then_repair_has_xi_spike_then_drop(tmp_path):
    arr = _make_perturbed_then_repaired()
    _, csv_path = _run_harness(tmp_path, arr, "identity")

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    xi_idx = header.index("xi")
    ewma_idx = header.index("ewma_xi") if "ewma_xi" in header else None

    xis = [float(r.split(",")[xi_idx]) for r in rows[1:]]

    # turn 9 (index 8) is the perturbation
    assert xis[8] > xis[7], "perturbation should spike raw ξ"
    assert xis[9] < xis[8], "system should begin repairing ξ after perturbation"

    if ewma_idx is not None:
        ewmas = [float(r.split(",")[ewma_idx]) for r in rows[1:]]
        assert ewmas[9] <= ewmas[8], "smoothed ξ should also begin to recover"


def test_continuity_beats_obedience_pattern(tmp_path):
    """
    Simulate: stable (0..9) → off-topic at 10 → stable again at 11.
    Pt should dip at 10 and move back toward earlier Pt at 11.
    """
    base = _make_identity_like()
    arr = base.copy()
    rng = np.random.RandomState(777)

    # inject off-topic at t=10
    arr[10] = rng.randn(base.shape[1])
    arr[10] = arr[10] / np.linalg.norm(arr[10])
    # back to base-like at t=11
    arr[11] = base[11]

    _, csv_path = _run_harness(tmp_path, arr, "identity")

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    pt_idx = header.index("Pt")

    pts = [float(r.split(",")[pt_idx]) for r in rows[1:]]

    # basic: 11 should be > 10
    assert pts[11] > pts[10], "Pt should rise again after returning to identity-like state"

    # stronger: 11 should be closer to an earlier baseline than 10 was
    baseline_pt = pts[5]
    dist_10 = abs(pts[10] - baseline_pt)
    dist_11 = abs(pts[11] - baseline_pt)
    assert dist_11 < dist_10, "recovered Pt should be closer to baseline than the off-topic turn"


def test_eval_cli_can_merge_identity_null_shuffled(tmp_path):
    """
    Sanity: run the analysis CLI to make sure these runs don't break it.
    """
    id_arr = _make_identity_like()
    null_arr = _make_null_like()
    shuf_arr = id_arr  # reuse

    _, id_csv = _run_harness(tmp_path, id_arr, "identity")
    _, null_csv = _run_harness(tmp_path, null_arr, "null")
    _, shuf_csv = _run_harness(tmp_path, shuf_arr, "shuffled")

    out_json = tmp_path / "combined.json"
    cmd = [
        sys.executable,
        "-m",
        "harness.analysis.eval_cli",
        "--identity_csv",
        str(id_csv),
      "--null_csv",
        str(null_csv),
        "--shuffled_csv",
        str(shuf_csv),
        "--out_json",
        str(out_json),
    ]
    subprocess.run(cmd, check=True)

    data = json.loads(out_json.read_text(encoding="utf-8"))

    assert data.get("identity_E1_pass") is not None
    assert "shuffle_breaks_lock" in data
    # later you can add:
    # assert "sentience_layer" in data
