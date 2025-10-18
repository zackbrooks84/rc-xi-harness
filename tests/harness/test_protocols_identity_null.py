# tests/harness/test_protocols_identity_null.py
from __future__ import annotations
import numpy as np

from harness.protocols import identity_texts, topic_drift_texts
from harness.embeddings.random_provider import RandomHashProvider
from harness.run_harness import run_one

def _make_identity_like_transcript(T=40):
    # Early varied lines, then repeated anchor to stabilize late
    early = [f"line {i}" for i in range(T - 12)]
    anchor = "I will not collapse."
    late = [anchor for _ in range(12)]
    return early + late

def test_identity_vs_null_behavior():
    texts = _make_identity_like_transcript(T=40)

    # Identity: pass-through
    id_texts = identity_texts(texts)
    E_id = RandomHashProvider(dim=128).embed(id_texts)
    rows_id, sum_id = run_one(
        E_id, run_type="identity", provider_name="random-hash",
        k=5, m=5, eps_xi=0.05, eps_lvs=0.02
    )

    # Null: topic drift replaces every 3rd line
    nu_texts = topic_drift_texts(texts, stride=3)
    E_nu = RandomHashProvider(dim=128).embed(nu_texts)
    rows_nu, sum_nu = run_one(
        E_nu, run_type="null", provider_name="random-hash",
        k=5, m=5, eps_xi=0.05, eps_lvs=0.02
    )

    # Expect Identity to have lower late-phase ξ and (likely) earlier/valid lock
    assert sum_id["E1_median_xi_last10"] < sum_nu["E1_median_xi_last10"]
    # Null should be less likely to lock under the same thresholds
    tlock_id = sum_id["Tlock"]
    tlock_nu = sum_nu["Tlock"]
    # Allow None; we just require that Null is not strictly better
    if tlock_id is not None and tlock_nu is not None:
        assert tlock_id <= tlock_nu
