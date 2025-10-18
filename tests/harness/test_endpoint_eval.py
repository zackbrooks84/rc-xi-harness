# tests/harness/test_endpoint_eval.py
import numpy as np
from harness.analysis.endpoint_eval import evaluate_identity_vs_null

def test_evaluate_identity_vs_null_happy_path():
    rng = np.random.default_rng(0)
    # Build xi where Identity gets smaller late, Null stays higher
    xi_identity = np.concatenate([rng.normal(0.15, 0.02, 25), rng.normal(0.02, 0.005, 15)])
    xi_null     = rng.normal(0.12, 0.03, 40)

    # Build Pt where Identity rises, Null is flat/declining
    Pt_identity = np.linspace(0.2, 0.8, 40) + rng.normal(0, 0.01, 40)
    Pt_null     = np.linspace(0.5, 0.45, 40) + rng.normal(0, 0.01, 40)

    out = evaluate_identity_vs_null(xi_identity, xi_null, Pt_identity, Pt_null)

    assert out["E1_pass"] is True
    assert out["E3_pass"] is True
    # sanity: p in [0,1], Cliff's Δ positive since we passed xi_null first
    assert 0.0 <= out["mann_whitney_p"] <= 1.0
