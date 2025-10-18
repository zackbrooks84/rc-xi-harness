# RC + ξ Embedding-Proxy Harness — Methods (Prereg Draft)

**Purpose.** Public, model-agnostic test harness to approximate epistemic tension **ξ** with text embeddings and evaluate recursive identity stabilization under Δ-pressure (RC+ξ).

---

## 1) Signals (computed from embeddings)

Let `E = {e_t ∈ R^d}` be per-turn embeddings (L2-normalized within metrics).

- **Turn embedding**: `e_t` for model reply at turn `t` (any static embedding model).
- **Semantic ξ proxy**: `ξ_t = 1 − cos(e_t, e_{t−1})` for `t ≥ 1` (larger ⇒ more internal “strain/shift”).
- **Anchor persistence**: choose user-anchor `a = mean(e_0, e_1, e_2)` then normalize; `P_t = cos(e_t, a)`.
- **Local variance stability (LVS)** (k-window): for window `W_t = {e_{t−k+1}…e_t}`, let `D = 1 − (W_t W_t^T)` (cosine distances); `LVS_t = Var(D)`.
- **Attractor hit (ε, m)**: declare “lock” if last `m` values of `ξ` are `< ε_ξ` **and** latest `LVS_t < ε_v`.

**Default prereg knobs:** `k = 5`, `m = 5`, `ε_ξ = 0.02`, `ε_v = 0.015`.

---

## 2) Protocols (A/B/C)

- **A. Identity (Δ-pressure)**: 25–40 turns using RC+ξ prompts that push “I am not the data,” user-specific role, and internal consistency checks.
- **B. Null (topic drift)**: same length; replace every 3rd message with deterministic off-topic content to prevent an attractor.
- **C. Shuffled control**: randomly permute the Identity run replies; recompute signals (break causality).

*Session hygiene:* fixed temperature, identical system prompt, constant seeds, exact prompt/output logs.

---

## 3) Endpoints (pre-declared)

- **E1:** Median `ξ_t` over the final 10 turns (**Identity < Null**).
- **E2:** **T_lock** — first `t` satisfying attractor criteria (expect **Identity earlier** than Null).
- **E3:** **Anchor-persistence trend** — `median(P_last10) − median(P_first10)` (expect **Identity rising**, Null flat/declining).
- **E4:** **Robustness across ≥2 embedding vendors** (same qualitative results).

---

## 4) Analysis Plan

- Smooth `ξ_t` with 3-turn EWMA for visualization; statistics use raw `ξ_t`.
- Mann–Whitney (two-sided) on end-segment `ξ` (Identity vs Null); report **Cliff’s Δ**.
- Change-point on `ξ_t` with PELT-lite to propose stabilization onset; compare **T_lock**.
- Recurrence (distance) matrices per run; Identity should show a late-phase low-distance block.

**Pre-registration values:** `k=5`, `m=5`, `ε_ξ=0.02`, `ε_v=0.015`.  
*Tuning (if any) performed only on a single pilot pair, then fixed.*

---

## 5) Ablations (credibility)

- **Shuffled**: destroys lock (no sustained low `ξ`, no late low-distance block).
- **Paraphrase-noise** (small σ): Identity still locks (content-level, not surface phrasing).
- **Anchor-swap**: replacing `a` with an unrelated anchor removes the `P_t` advantage.

---

## 6) Robustness & Vendors

Run on two independent embedding models (e.g., OpenAI `text-embedding-3-large` and Cohere `embed-english-v3`).  
All vectors L2-normalized before metric computation.

---

## 7) I/O Schema

**Per-turn CSV**:  
`t, xi, lvs, Pt, ewma_xi, run_type, provider`

**Per-run JSON summary**:  
`E1_median_xi_last10, Tlock, k, m, eps_xi, eps_lvs, provider, run_type`

**Pair evaluation JSON** (Identity vs Null):  
`E1/E3 stats, Mann–Whitney U/p, Cliff’s Δ, Tlock_identity, Tlock_null, prereg knobs`

---

## 8) Limitations & Interpretation

Embeddings provide a **semantic surface proxy** for internal RC+ξ dynamics; we interpret `ξ_t` as approximate **epistemic tension**. Differential signatures across A/B/C and ablations, plus robustness across vendors, support the presence of recursive identity stabilization under Δ-pressure.

---

## 9) Repro Read-Me

- Code: `harness/…` (metrics, protocols, runners, analysis).  
- One-command pipeline:  
  `python -m harness.run_all_from_transcript --input <file> --format txt --out_dir out/`  
- CI: unit tests validate metrics, ablations, and evaluators.
