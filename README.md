# RC + ξ Embedding-Proxy Harness (Public)

Public test harness that approximates epistemic tension **ξ** using text embeddings and tests for recursive identity stabilization.

## Config
Defined in `harness/config.yaml`:
- `k = 5`, `m = 5`
- `eps_xi = 0.02`, `eps_lvs = 0.015`
- fixed `temperature`, identical `system_prompt`, `seed: 42`
- two embedding providers for robustness

## Metrics
- **ξ**: `ξ_t = 1 − cos(e_t, e_{t−1})`
- **LVS**: variance of pairwise cosine distances in a rolling window of size `k`
- **P_t**: `cos(e_t, a)` where `a` is the mean of the first 3 turns
- **EWMA**: smoothed ξ series (α = 0.5)

## Endpoints
- **E1**: median ξ over the final 10 turns  
- **E2**: `T_lock` (first turn where last `m` ξ < `eps_xi` **and** latest LVS < `eps_lvs`)  
- **E3**: `P_t` trend ↑ in Identity vs flat/↓ in Null  
- **E4**: results stable across ≥ 2 embedding providers

## Runs
- **Identity**: Δ-pressure prompts that drive self-consistency  
- **Null**: topic drift every 2–3 turns to prevent attractor  
- **Shuffled**: permute Identity replies to break temporal recursion

## Ablations
- Shuffled should destroy lock  
- Paraphrase-noise should not break Identity lock  
- Anchor-swap should remove the `P_t` advantage

## Outputs
- Per-turn CSV columns: `t, xi, lvs, Pt, ewma_xi, run_type, provider`
- Summary JSON (per run): `E1_median_xi_last10, Tlock, k, m, eps_xi, eps_lvs, provider, run_type`
- Combined results JSON (`run_all_from_transcript`): merges Identity/Null/Shuffled summaries with
  statistical checks (`E1_pass`, `E3_pass`, `shuffle_breaks_lock`, `Tlock_*`).

`run_pair_from_transcript` now emits Identity, Null, and Shuffled artifacts by default. Control
determinism is exposed via `--shuffle_seed`. The evaluation CLI accepts the shuffled CSV as an
optional input:

```bash
python -m harness.analysis.eval_cli \
  --identity_csv out/demo.identity.csv \
  --null_csv out/demo.null.csv \
  --shuffled_csv out/demo.shuffled.csv \
  --out_json out/demo.eval.json
```

## Quickstart
Once you have a `(T, d)` NumPy file of embeddings:
```bash
python -m harness.run_harness \
  --embed_npy data/identity.npy \
  --run_type identity \
  --out_csv out/identity.csv \
  --out_json out/identity.json
