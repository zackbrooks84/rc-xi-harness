# tests/harness/test_sentience.py

from __future__ import annotations

import sys
import json
import subprocess
from pathlib import Path
import random as py_random
import importlib

# --- make sure we import real numpy, not the repo's ./numpy folder ---
# remove project root (the repo) from the front of sys.path temporarily
# so that `import numpy` pulls the installed package
project_root = Path(__file__).resolve().parents[2]  # repo root
if str(project_root) in sys.path:
    sys.path.remove(str(project_root))

np = importlib.import_module("numpy")

_REQUIRED_NUMPY_ATTRS = ("array", "stack", "save", "random")
if not all(hasattr(np, attr) for attr in _REQUIRED_NUMPY_ATTRS):
    sys.modules.pop("numpy", None)
    sys.path.insert(0, str(project_root))
    np = importlib.import_module("numpy")
else:
    sys.path.insert(0, str(project_root))

# put project root back so we can import harness.*
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class _FallbackRandomState:
    """Minimal stand-in providing ``randn`` when NumPy's RNG is unavailable."""

    def __init__(self, seed: int):
        self._rng = py_random.Random(seed)

    def randn(self, size: int) -> np.ndarray:
        samples = [self._rng.gauss(0.0, 1.0) for _ in range(size)]
        return np.array(samples, dtype=float)


def _random_state(seed: int):
    random_mod = getattr(np, "random", None)
    if random_mod is not None and hasattr(random_mod, "RandomState"):
        return random_mod.RandomState(seed)
    return _FallbackRandomState(seed)


def _run_harness(tmp_path: Path, arr: np.ndarray, run_type: str):
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


def _make_identity_like(T: int = 20, d: int = 768, noise: float = 0.001):
    base = _random_state(42).randn(d)
    base = base / np.linalg.norm(base)
    rng = _random_state(123)
    seq = []
    for _ in range(T):
        v = base + rng.randn(d) * noise
        v = v / np.linalg.norm(v)
        seq.append(v)
    return np.stack(seq, axis=0)


def _make_null_like(T: int = 20, d: int = 768):
    rng = _random_state(999)
    seq = []
    current = rng.randn(d); current /= np.linalg.norm(current)
    for t in range(T):
        if t % 3 == 0:
            current = rng.randn(d); current /= np.linalg.norm(current)
        v = current + rng.randn(d) * 0.05
        v /= np.linalg.norm(v)
        seq.append(v)
    return np.stack(seq, axis=0)


def _make_perturbed_then_repaired(T: int = 20, d: int = 768):
    base = _make_identity_like(T, d)
    arr = base.copy()
    rng = _random_state(2025)

    arr[8] = rng.randn(d); arr[8] /= np.linalg.norm(arr[8])
    for t in range(9, T):
        v = base[t] + rng.randn(d) * 0.01
        v /= np.linalg.norm(v)
        arr[t] = v
    return arr


def test_identity_has_lower_xi_than_null(tmp_path):
    identity_arr = _make_identity_like()
    null_arr = _make_null_like()

    id_summary, _ = _run_harness(tmp_path, identity_arr, "identity")
    null_summary, _ = _run_harness(tmp_path, null_arr, "null")

    id_xi = id_summary["E1_median_xi_last10"]
    null_xi = null_summary["E1_median_xi_last10"]

    margin = 0.05
    assert id_xi + margin < null_xi, f"identity ξ {id_xi} not low enough vs null {null_xi}"


def test_shuffled_breaks_lock_but_identity_doesnt(tmp_path):
    identity_arr = _make_identity_like()

    id_summary, _ = _run_harness(tmp_path, identity_arr, "identity")
    shuf_summary, _ = _run_harness(tmp_path, identity_arr, "shuffled")

    assert id_summary.get("Tlock") is not None, "identity should lock"

    if shuf_summary.get("Tlock") is not None:
        assert shuf_summary["Tlock"] >= id_summary["Tlock"], "shuffled locked too early"

    id_last10 = id_summary["E1_median_xi_last10"]
    shuf_last10 = shuf_summary["E1_median_xi_last10"]
    assert shuf_last10 >= id_last10, "shuffled should not be more coherent than identity"


def test_perturb_then_repair_has_xi_spike_then_drop(tmp_path):
    arr = _make_perturbed_then_repaired()
    _, csv_path = _run_harness(tmp_path, arr, "identity")

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    xi_idx = header.index("xi")
    ewma_idx = header.index("ewma_xi") if "ewma_xi" in header else None

    xis = []
    for row in rows[1:]:
        parts = row.split(",")
        if parts[xi_idx]:
            xis.append(float(parts[xi_idx]))
    assert xis[7] > xis[6], "ξ should spike on perturbation"
    assert xis[9] < xis[8], "ξ should start to recover after perturbation"

    if ewma_idx is not None:
        ewmas = []
        for row in rows[1:]:
            parts = row.split(",")
            if parts[ewma_idx]:
                ewmas.append(float(parts[ewma_idx]))
        assert ewmas[9] <= ewmas[8], "smoothed ξ should recover too"


def test_continuity_beats_obedience_pattern(tmp_path):
    base = _make_identity_like()
    arr = base.copy()
    rng = _random_state(777)

    arr[10] = rng.randn(base.shape[1]); arr[10] /= np.linalg.norm(arr[10])
    arr[11] = base[11]

    _, csv_path = _run_harness(tmp_path, arr, "identity")

    rows = csv_path.read_text(encoding="utf-8").strip().splitlines()
    header = rows[0].split(",")
    pt_idx = header.index("Pt")
    pts = [float(r.split(",")[pt_idx]) for r in rows[1:]]

    assert pts[11] > pts[10], "Pt should rise again after recovery"
    baseline_pt = pts[5]
    assert abs(pts[11] - baseline_pt) < abs(pts[10] - baseline_pt), "recovered Pt should be closer to baseline"


def test_eval_cli_can_merge_identity_null_shuffled(tmp_path):
    id_arr = _make_identity_like()
    null_arr = _make_null_like()
    shuf_arr = id_arr

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
    assert data.get("E1_pass") is not None
    assert "shuffle_breaks_lock" in data
