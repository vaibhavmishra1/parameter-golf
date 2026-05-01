# Mudskipper V1 Combined Submission

## What This Submission Does

This record starts from the strong Gated XSA + LQER + in-timer n-gram tilt stack, then adds **Mudskipper v1** as a training-time data selection layer.

Mudskipper v1 does not change the model architecture, optimizer, quantization path, EMA, TTT, or validation logic. It only changes how training batches are assembled:

1. The loader reads extra candidate training windows on CPU.
2. A cheap online scout scores those windows using running token statistics.
3. Most of the batch remains in normal stream order.
4. A small fraction of windows is replaced with higher-scoring candidate windows.
5. The selected windows are packed back into the same varlen `x/y/cu_seqlens` format expected by the original training loop.

The default settings are intentionally conservative:

```bash
MUDSKIPPER_SCOUT=1
MUDSKIPPER_SCOUT_FRACTION=0.125
MUDSKIPPER_CANDIDATE_MULT=1.25
MUDSKIPPER_BUCKET_COUNT=16
MUDSKIPPER_RECENT_HASHES=8192
```

The scout score is based on cheap CPU features:

- running unigram surprise
- rare-token fraction
- bucket coverage
- repetition / junk penalties
- recent-window fingerprint penalty

The goal is to spend the same GPU training budget on slightly more useful token windows, while using otherwise idle CPU time.

## PRs And Ideas Used

This submission combines ideas from the following public Parameter Golf PRs:

| Source | Contribution Used |
|---|---|
| PR #2018 | Main codebase: Gated XSA, token-only in-timer n-gram tilt, TTT path, strong V21-family stack |
| PR #2060 | Quant tuning: `LQER_RANK=2`, `LQER_ASYM_GROUP=32`, `LQER_TOP_K=4`, `MATRIX_LR=0.028` |
| PR #2007 | Long-context / asym-logit lineage used by the parent stack |
| PR #1967 | V21 + n-gram tilt + LeakyReLU 0.3 lineage |
| PR #1948 | LeakyReLU squared slope 0.3 tuning |
| PR #1953 | QK / TTT tuning lineage |
| PR #1945 | V21 stack composition |
| PR #1923 | Asymmetric logit rescale |
| PR #1908 | AWQ-lite mixed precision |
| PR #1855 | Per-group compressor and strong 10min/16MB baseline lineage |
| PR #1514 | Token-only n-gram workaround precedent |
| PR #1145 | Closed-form n-gram tilt with normalized probability correction |

Local addition:

| Source | Contribution Used |
|---|---|
| `mudskipper_2/trian_gpt_original_v1.py` | CPU-side Mudskipper v1 scout data-loader heuristic |

## Results

Results are intentionally left blank until the final 8xH100 runs finish.

| Run | Seed | Mudskipper | Train Time | Post-Quant BPB | Post-TTT BPB | Artifact Bytes | Notes |
|---|---:|---:|---:|---:|---:|---:|---|
|  | 42 | on |  |  |  |  |  |
|  | 0 | on |  |  |  |  |  |
|  | 1234 | on |  |  |  |  |  |
|  | 42 | off |  |  |  |  | same-script control |

## Reproduce

From the repository root:

```bash
cd /workspace/parameter-golf
```

Install dependencies:

```bash
pip install -r records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/requirements.txt
sudo apt-get install -y build-essential lrzip
```

Prepare the CaseOps-tokenized dataset if it is not already present:

```bash
cd records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined
python prepare_caseops_data.py
cd /workspace/parameter-golf
```

Run the Mudskipper-enabled submission:

```bash
SEED=42 \
RUN_ID=combined_mud_v1_s42 \
records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/run.sh
```

Recommended seed set:

```bash
SEED=42   RUN_ID=combined_mud_v1_s42   records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/run.sh
SEED=0    RUN_ID=combined_mud_v1_s0    records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/run.sh
SEED=1234 RUN_ID=combined_mud_v1_s1234 records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/run.sh
```

Same-script control without Mudskipper:

```bash
MUDSKIPPER_SCOUT=0 \
SEED=42 \
RUN_ID=combined_nomud_s42 \
records/track_10min_16mb/2026-04-30_GatedXSA_LQERg32top4_TTTlocal080_NgramTilt_Combined/run.sh
```

Useful diagnostics in the log:

```text
mudskipper scout:...
resources proc_cpu:... cuda_mem:... cuda_util:...
```

These lines show candidate counts, skipped windows, CPU scoring cost, H2D copy time, and basic CPU/GPU utilization while training runs.

## Notes

- `MUDSKIPPER_SCOUT=0` should preserve the same training path as the combined submission without CPU triage.
- The eval-time n-gram helper is embedded directly inside `train_gpt.py`; the standalone helper files are no longer required.
- The Mudskipper scout is a heuristic. Treat the same-script no-Mudskipper run as the required control before claiming a gain.
