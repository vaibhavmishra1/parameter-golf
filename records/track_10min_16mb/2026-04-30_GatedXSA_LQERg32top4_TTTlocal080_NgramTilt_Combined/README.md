# Combined: Gated XSA + LQER g32/top4 + In-Timer N-gram Tilt

**Predicted post-TTT val_bpb:** ~1.0460–1.0470 (untrained projection — needs verification on 8×H100s)

This submission combines the **best-of-best** across all open SOTA-frontier PRs as of 2026-05-01:

| Stage | Source PR | What we take | Why |
|---|---|---|---|
| **Pre-quant (1.05028 BPB)** | **PR #2018** (simon-marcus) | Full `train_gpt.py` codebase: V21 + Gated XSA + LeakyReLU 0.3 + token-only n-gram tilt + LQER + AWQ-lite + AsymLogit + Smear/SparseAttn gates | Best raw-model loss observed across all open PRs (−0.01485 BPB vs merged SOTA #1855) |
| **Quant (Δ +0.00855 → ~+0.0080)** | **PR #2060** (S0urC10ud) | `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`, `MATRIX_LR=0.028` | Gave −0.00107 BPB over PR #2007 with p≈0.011 |
| **TTT (Δ −0.01161)** | **PR #2018** (kept as-is) | In-timer token-only n-gram tilt + 1-phase / 1000-prefix-doc phased TTT | C1-causality compliant; eval finishes 463–471s, leaves margin |

### ⚠ Knobs from PR #2060 we DROPPED (not honored by PR #2018's code)

PR #2060's 5th knob (`TTT_LOCAL_LR_MULT=0.80` vs 0.75) cannot be ported because PR #2018's `train_gpt.py` does **not** read `TTT_LOCAL_LR_MULT`. Q+V LoRA toggles (`TTT_MASK`, `TTT_Q_LORA`, `TTT_V_LORA`) are also unsupported in PR #2018 — Q and V LoRAs are always created. PR #2018's reported 1.04722 BPB was achieved with Q+V LoRA enabled by default and `TTT_LORA_LR=0.0001` (no local mult), regardless of what its own README/run.sh claimed.

This means our combined script applies only **4 of PR #2060's 5 knobs**. The 5th knob's contribution to PR #2060's −0.00107 was probably small (the LQER changes are the dominant effect), but it's worth noting.

## Theoretical projection

```
PR #2018 baseline (verified):
  pre-quant   = 1.05028
  Δ_quant     = +0.00855  →  post-quant  = 1.05883
  Δ_ttt       = −0.01161  →  post-ttt    = 1.04722

Combined (with 4 of PR #2060's 5 quant knobs ported):
  pre-quant   = 1.05028  (unchanged: Gated XSA architecture identical)
  Δ_quant     ≈ +0.00795  (PR #2060's ~0.001 better quant on the V21 family,
                           assuming the LQER changes dominate the 5th-knob effect)
                          → post-quant ≈ 1.05823
  Δ_ttt       ≈ −0.01161  (TTT path identical to PR #2018; TTT_LOCAL_LR_MULT
                           is a no-op in PR #2018's code, so 0.80 vs 0.75
                           wouldn't have changed anything anyway)
                          → post-ttt   ≈ 1.04662

Optimistic   ≈ 1.046
Conservative ≈ 1.047 (matches PR #2018 baseline, small downside risk)
```

vs current merged SOTA (PR #1855: 1.06108 BPB) → expected gain **≈ −0.014 BPB / −0.031 nats**.

## Stack details

### Architecture (PR #2018)

| Component | Setting | Source |
|---|---|---|
| Layers / dim / heads | 11 / 512 / 8 GQA, 4 KV | Baseline |
| MLP | 4× LeakyReLU(0.5)² (slope=0.3 in fused kernel) | PR #1948 |
| Attention | FA3 + GQA, partial RoPE 16/64 + YaRN | PR #315 |
| **Gated XSA** | per-head `tanh(α)` gate, zero-init | **PR #2018 (modded-nanogpt PR#264)** |
| XSA | All 11 layers | PR #478 |
| QK-Gain init | 5.25 | PR #1953 |
| LN scale | 1/√(layer+1) | PR #315 |
| U-Net skips + skip gates | Encoder-decoder skip | PR #289 |
| Parallel residuals | 2-lane from layer 8+ | PR #1530 |
| Depth recurrence | Loop layers 3-5, 3× when frac≥0.35 | PR #1344 |
| SmearGate (BOS-fixed) | Position-mixing gate, gate_window=12 | PR #1667 + #1855 |
| Sparse attention gate | gate_window=12, scale=0.5 | PR #1787 / #2007 |
| Logit softcap | 30 | Gemma2-style |
| Asymmetric logit rescale | Eval-only | PR #1923 |

### Training (PR #2018 + PR #2060 quant tuning)

| hparam | value | source |
|---|---|---|
| `MATRIX_LR` | **0.028** | PR #2060 (was 0.026) |
| `MIN_LR` | 0.1 | PR #2060/#2018 |
| `BETA2` | 0.95 | PR #2018 |
| `WARMDOWN_FRAC` | 0.75 | PR #2018 |
| `GRAD_CLIP_NORM` | 0.3 | PR #2018 |
| Train wallclock cap | 600s | rules |
| `MUDSKIPPER_SCOUT` | **1**, fraction=0.125, candidate_mult=1.25 | local v1 CPU-side batch triage |

### Quantization (PR #2018 stack + PR #2060 LQER tuning)

| hparam | value | source |
|---|---|---|
| Matrix bits | 6 (GPTQ int6) | base |
| Embed bits | 7 | PR #1586 |
| `LQER_ENABLED` | 1 | base |
| `LQER_RANK` | **2** | PR #2060 (was 4) |
| `LQER_ASYM_GROUP` | **32** | PR #2060 (was 64) |
| `LQER_TOP_K` | **4** | PR #2060 (was 1) |
| `LQER_FACTOR_BITS` | 4 | base |
| `AWQ_LITE_ENABLED` | 1, bits=8, group_size=64, top_k=1 | PR #1908 |
| Compressor | `pergroup` (lrzip ZPAQ) | PR #1855 |

**Artifact-size sanity check**:
- PR #2018 max artifact: 15,996,490 bytes (3,510 B headroom under 16,000,000)
- PR #2060's LQER changes: rank 4→2 (~half size per tensor) × top_k 1→4 (4× more tensors) ≈ ~2× LQER bytes
- Asym group 64→32 doubles scale tensors (~few KB)
- **Net:** small increase, but PR #2060's actual artifact (15,971,748 B max) was ~28KB smaller than PR #2018, so the TOP_K=4 tensors are net cheaper than PR #2018's TOP_K=1 single rank-4 tensor. Should fit comfortably under 16MB.

### Eval & TTT (kept from PR #2018 — critical timing trade-off)

| hparam | value | source / rationale |
|---|---|---|
| `EVAL_SEQ_LEN` / `TTT_EVAL_SEQ_LEN` | 2560 / 2560 | PR #2018 |
| `PHASED_TTT_NUM_PHASES` | **1** (NOT 3) | PR #2018 — n-gram precompute eats budget |
| `PHASED_TTT_PREFIX_DOCS` | **1000** (NOT 3000) | PR #2018 — same |
| `TTT_LORA_RANK` | 80 | PR #2018 |
| `TTT_LORA_LR` | 0.0001 (default) | PR #2018 (no local mult support) |
| ~~`TTT_MASK`~~ | ~~no_qv~~ | NOT SUPPORTED in PR #2018; Q+V LoRAs always created |
| ~~`TTT_LOCAL_LR_MULT`~~ | ~~0.80~~ | NOT SUPPORTED in PR #2018; dropped from script |
| `NGRAM_TILT_ENABLED` | 1 | PR #2018 |
| `NGRAM_HINT_PRECOMPUTE_OUTSIDE` | **0** (in-timer, REQUIRED) | A2 merged-record precedent |
| `TOKEN_ORDER` / `TOKEN_THRESHOLD` / `TOKEN_BOOST` | 16 / 0.800 / 2.625 | PR #2018 |
| `WITHIN_BOOST` / `WORD_BOOST` | **0 / 0** (disabled — C1 violation in PR #1967) | PR #1514 conservative path |

## Why we DIDN'T merge PR #2060's TTT phasing settings

PR #2060 uses `PHASED_TTT_NUM_PHASES=3` and `PHASED_TTT_PREFIX_DOCS=3000` because it has no n-gram precompute (frees ~150s of eval budget for phased TTT). PR #2018 uses 1-phase / 1000-prefix specifically because the n-gram precompute is in-timer and consumes that budget. Combining 3-phase TTT with in-timer n-gram would blow past the 600s eval cap. We keep PR #2018's choice.

## Compliance

- **Artifact ≤ 16,000,000 B**: projected within 16M (see size sanity check above)
- **Train ≤ 600s**: identical training loop to PR #2018 (which trained at 596.1s)
- **Eval ≤ 600s**: identical eval pipeline (PR #2018 ran in 463–471s)
- **C1 causal**: token-only n-gram via `token_context_hash(st)` over prefix state, hint emitted before push
- **C2 normalized**: closed-form `p'(a) = exp(β·1[a==h])·p(a) / Z`, `Z = 1 + p(h)·(exp(β)−1)`, ΣP=1
- **C3 score-before-update**: phased TTT scores all chunks before any LoRA update
- **C4 single pass**: each val token contributes exactly one BPB term
- **In-timer precompute**: `NGRAM_HINT_PRECOMPUTE_OUTSIDE=0` (per A2 merged precedent)
- **Token-only n-gram**: `WITHIN_BOOST=0`, `WORD_BOOST=0` — within-word/word-level experts disabled (C1 violation in PR #1967 fix)

## Reproduction

```bash
# 1. Install dependencies
pip install -r requirements.txt
sudo apt-get install -y build-essential lrzip   # for embedded native n-gram helper + pergroup compressor

# 2. Prepare CaseOps-tokenized dataset
python prepare_caseops_data.py

# 3. Run a seed (3 seeds recommended for statistical significance)
SEED=42   ./run.sh
SEED=0    ./run.sh
SEED=1234 ./run.sh

# Control run with the same script but no Mudskipper triage
MUDSKIPPER_SCOUT=0 SEED=42 ./run.sh
```

## Lineage

This work is purely a recombination of existing public PRs — no new research:

- **PR #2018** by [@simon-marcus](https://github.com/simon-marcus) — Gated XSA + LQER top-1 + strict token-only in-timer n-gram TTT (1.04722 BPB)
- **PR #2060** by [@S0urC10ud](https://github.com/S0urC10ud) — 5-knob retune of #2007 with LQER g32/top4 (1.05792 BPB)
- **PR #2007** by [@Elubrazione](https://github.com/Elubrazione) — LongCtx No-QV QK5.25 + AsymLogit (1.05899 BPB)
- **PR #1967** by [@ndokutovich](https://github.com/ndokutovich) — V21 + N-gram Tilt + LeakyReLU 0.3 base
- **PR #1948** by [@TimS-ml](https://github.com/TimS-ml) — LeakyReLU squared slope 0.3 sweep
- **PR #1953** by [@andrewbaggio1](https://github.com/andrewbaggio1) — 7-knob TTT/QK tuning
- **PR #1945** by [@alertcat](https://github.com/alertcat) — V21 stack composition
- **PR #1923** by [@jorge-asenjo](https://github.com/jorge-asenjo) — Asymmetric Logit Rescale
- **PR #1908** by [@romeerp](https://github.com/romeerp) — AWQ-lite mixed precision
- **PR #1855** by [@codemath3000](https://github.com/codemath3000) — per-group lrzip + 9-hparam stack
- **PR #1514** by [@codemath3000](https://github.com/codemath3000) — token-only n-gram workaround precedent
- **PR #1145** by [@AnirudhRahul](https://github.com/AnirudhRahul) — closed-form n-gram tilt + Σp=1 renorm

## Risks / Unknowns

1. **Untrained projection** — the combined stack has not been verified on 8×H100s. The numerical projection assumes the PR #2060 quant gain (−0.001 BPB) transfers to the PR #2018 base. This is plausible because the LQER tuning is mostly orthogonal to architecture changes, but it's not guaranteed.
2. **Artifact size** — could marginally exceed 16MB with `LQER_TOP_K=4`. If so, fall back to `LQER_TOP_K=3` (still better than #2018's TOP_K=1 if size permits).
3. **Eval timing** — PR #2018 eval was 463–471s with `LQER_TOP_K=1`. With `LQER_TOP_K=4` and `LQER_ASYM_GROUP=32`, the per-tensor LQER reconstruction is roughly the same total work (rank halved × 4× tensors), so eval time should be similar. If it pushes over 600s, drop `PHASED_TTT_PREFIX_DOCS` from 1000 → 750.
4. **Seed variance** — PR #2018 had per-seed range [1.04617, 1.04826] (Δ = 0.0021). Need 3-seed mean for statistical claim.
