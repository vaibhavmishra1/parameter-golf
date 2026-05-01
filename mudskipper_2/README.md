# Mudskipper 2: CPU-Side Online Data Triage

## One-line thesis

Use otherwise-idle CPU time to maintain a cheap memory of what training has already covered, then bias the GPU loader toward windows that are useful, non-redundant, and not obvious junk, without adding any extra GPU forward pass.

## Why this might work

The current frontier already uses strong architecture, tokenizer, quantization, and eval-time tricks. Pure static data selection has mostly failed because FineWeb is already filtered and hard chunk selection removes diversity. The narrow opening is not "pick the best data once." The opening is:

```text
During the 10-minute run, spend slightly more GPU gradient budget on data modes
that are still undercovered by the current run, while preserving broad diversity.
```

The CPU can track this cheaply while the GPU trains. The GPU should remain the expensive student. The CPU scout is only a notebook that tracks coverage and proposes mildly better next pages.

## What this is not

- Not TTT.
- Not validation-set training.
- Not a learned data selector trained offline.
- Not hard top-k chunk filtering.
- Not a second GPU model pass to score candidates.
- Not "maximally far apart" static diversity selection.
- Not a full LSTM/RNN scout in the first version.

The first version must be boring, cheap, and hard to break.

## Main bet

The best first version is a non-parametric CPU scout, not a CPU LSTM.

A CPU LSTM sounds attractive because CPU is partly free, but it has three problems:

1. It is sequential and may become a data-loader bottleneck.
2. Its learned difficulty estimate can drift away from what the transformer actually needs.
3. Debugging it adds too many degrees of freedom before we know if the signal exists.

Instead, Mudskipper 2 should use cheap token statistics and coverage memory:

- unigram coverage
- hashed bigram coverage
- rare-token coverage
- window fingerprint novelty
- repetition and boilerplate filters
- rolling cluster budgets
- bounded sampling bias

If this wins, a tiny learned scout can be tried later. The first question is whether online coverage tracking has signal at all.

## Core algorithm

Start from the current strong coprime multi-shard loader. For each training batch, the loader already samples many candidate windows across shards. Mudskipper 2 changes candidate selection, not the model.

For every candidate window, compute a cheap CPU score:

```text
interest =
    learnable_surprise
  + coverage_bonus
  + novelty_bonus
  - redundancy_penalty
  - junk_penalty
  - quota_penalty
```

Then select windows with a mild mixture:

```text
80-90% normal coprime loader behavior
10-20% scout-biased windows
```

The scout should never control the whole batch. If the scout is wrong, the baseline data distribution should still dominate.

## Features

### 1. Learnable surprise

Use a CPU n-gram-ish model, not the transformer:

```text
surprise = average negative log probability under online unigram/bigram counts
```

But clip it into a middle band.

Very low surprise means the window is probably redundant or too easy. Very high surprise may be noise, formatting junk, code blobs, weird markup, or rare byte garbage. The useful zone is medium-high surprise.

Suggested shape:

```text
learnable_surprise = triangular_score(cpu_nll, low, mid, high)
```

Where:

- below `low`: easy/redundant
- near `mid`: most useful
- above `high`: suspicious/noisy

### 2. Coverage bonus

Track how much of each token family has been trained recently.

Cheap families:

- rare token buckets
- capitalization/CaseOps marker buckets if available
- punctuation-heavy buckets
- digit/math buckets
- URL/markup-ish buckets
- short-word/common-text buckets
- high-token-entropy buckets
- low-token-entropy/repetitive buckets

Each bucket has a target budget. If a candidate contains undercovered buckets, give it a bonus.

### 3. Novelty bonus

Use SimHash or MinHash-style fingerprints over token shingles.

Maintain a small rolling table of fingerprints from recent accepted windows. Candidate windows too close to recent ones get downweighted.

This avoids spending the 10-minute budget rereading the same kind of page.

### 4. Junk penalty

Reject or downweight windows with obvious bad structure:

- excessive repetition
- extreme single-token dominance
- too many replacement/unknown/special tokens
- too many non-text formatting tokens
- extremely high CPU n-gram NLL
- very low unique-token ratio
- pathological long runs of punctuation/digits

This is important because naive high-loss sampling tends to chase junk.

### 5. Quota penalty

Maintain a rolling budget across coarse clusters/buckets.

If the loader has already spent too much on one bucket, lower that bucket's score until the rest of the stream catches up.

This is the "do not fill the chit in the middle of the book" mechanism.

## Candidate selection

For each global batch:

1. Draw `K` candidate windows from the existing coprime loader logic.
2. Score candidates on CPU.
3. Preserve most windows from the baseline distribution.
4. Replace only a small fraction with the highest-scoring candidates after quota penalties.
5. Update scout memory only for accepted windows.

Initial values:

```text
candidate_multiplier = 1.25
scout_fraction = 0.125
fingerprint_table_size = 8192 to 32768
feature_decay_half_life = 200 to 500 steps
```

Avoid `2x` oversampling at first. PR #737 already showed that selection overhead can erase any signal.

## Throughput constraint

This experiment lives or dies on throughput.

Hard rule:

```text
GPU step time regression must be <= 1.5%
```

If step time regresses more than that, kill or simplify the scout.

The CPU scout must run inside the existing prefetch path. The training loop should see a ready batch exactly as before.

## Why this is better than pre-storing maximally apart content

Static max-diversity selection has three weaknesses:

1. It can pick outliers instead of useful examples.
2. It cannot react to what the current model has already learned.
3. It commits the whole training run before seeing training dynamics.

Mudskipper 2 is better only if it remains adaptive:

- If a mode is already covered, it backs off.
- If a mode is undercovered, it increases sampling.
- If a mode looks like junk, it rejects the temptation.
- If the scout is uncertain, baseline sampling dominates.

The goal is not maximum distance. The goal is maximum marginal training value per GPU step.

## Why this is different from failed prior PRs

### PR #772: static data ordering and selection

That work tested shard-level and chunk-level scoring. Hard chunk selection got worse because it removed diversity. Mudskipper 2 should not hard select a small subset. It should keep the base distribution and only apply a mild online bias.

### PR #737: online entropy curriculum

That work loaded 2x data and filtered by unigram entropy, costing step time. Mudskipper 2 should avoid 2x oversampling and should score inside the CPU prefetch budget. It also should use multiple signals, not entropy alone.

### Coprime loader

The coprime loader already gives cheap diversity across shards. Mudskipper 2 should be implemented as a small extension of this idea, not a replacement. If Mudskipper 2 cannot beat coprime loader, it should die quickly.

## Experimental phases

### Phase 0: Instrument only

No sampling change.

Add logging for candidate/window statistics:

- CPU n-gram NLL distribution
- unique token ratio
- repetition score
- bucket coverage
- fingerprint collision rate
- step time
- final prequant BPB

Purpose: confirm the features are cheap and not degenerate.

Go criterion:

```text
feature computation adds <= 0.5% step-time overhead
```

### Phase 1: Junk guard only

Keep normal coprime sampling, but reject only extreme junk windows and replace from baseline candidates.

Purpose: test whether obvious bad-window avoidance helps without changing distribution much.

Go criterion:

```text
prequant BPB improves by >= 0.0005 on 1xH100
or no BPB loss with no step-time loss
```

### Phase 2: Mild coverage bias

Turn on 10-12.5% scout-biased replacement.

Use:

- learnable surprise band
- bucket undercoverage bonus
- repetition penalty
- rolling quotas

Go criterion:

```text
prequant BPB improves by >= 0.0015 on 1xH100
step-time regression <= 1.5%
```

### Phase 3: Ablation grid

Small grid only:

```text
SCOUT_FRACTION: 0.0625, 0.125, 0.20
CANDIDATE_MULTIPLIER: 1.125, 1.25, 1.5
SURPRISE_MODE: unigram, hashed_bigram, mixed
QUOTA_DECAY: 200, 500
```

Do not run a giant search. If the idea needs huge tuning, it is not robust enough for this contest.

### Phase 4: 8xH100 validation

Only scale if 1xH100 shows signal without throughput loss.

Need compare:

- same seed baseline
- same seed Mudskipper 2
- ideally 3 seeds if first result survives

Acceptance target:

```text
mean gain >= 0.005 nats/token equivalent
or enough prequant gain to justify combining with current quant/TTT stack
```

## Metrics to log

Training:

- step time
- number of completed steps
- tokens trained
- train loss
- prequant val BPB
- postquant val BPB if available

Scout:

- candidate count
- accepted from baseline
- accepted from scout
- rejected as junk
- average CPU n-gram NLL
- novelty score mean
- bucket coverage histogram
- quota penalties
- feature compute time
- prefetch wait time

The key metric is not just BPB. The key metric is:

```text
BPB improvement per lost training step
```

If the loader makes better batches but loses enough steps, it fails.

## Kill criteria

Kill immediately if any of these happen:

- step time regression > 3% in smoke testing
- GPU waits on CPU prefetch
- scout selects mostly repetitive or weird-format text
- train loss improves but val BPB worsens
- result depends on one extremely narrow threshold
- 1xH100 prequant gain is below 0.001 BPB after basic tuning

This idea should not become a rabbit hole. It either shows cheap signal fast, or it is probably not worth further spend.

## First implementation target

Implement as a loader-level feature flag:

```text
MUDSKIPPER_SCOUT=0/1
MUDSKIPPER_SCOUT_FRACTION=0.125
MUDSKIPPER_CANDIDATE_MULT=1.25
MUDSKIPPER_LOG_EVERY=100
```

Default should preserve current behavior exactly.

The first patch should touch only loader/scout code and logging. No architecture, tokenizer, quantization, or TTT changes.

## Recommended first runs

Baseline:

```text
MUDSKIPPER_SCOUT=0
```

Instrumentation:

```text
MUDSKIPPER_SCOUT=0
MUDSKIPPER_INSTRUMENT=1
```

Junk guard:

```text
MUDSKIPPER_SCOUT=1
MUDSKIPPER_SCOUT_FRACTION=0.0
MUDSKIPPER_JUNK_GUARD=1
```

Mild scout:

```text
MUDSKIPPER_SCOUT=1
MUDSKIPPER_SCOUT_FRACTION=0.125
MUDSKIPPER_CANDIDATE_MULT=1.25
```

## Expected outcome

Pessimistic expectation:

```text
0.000 to 0.001 BPB gain, maybe killed by overhead.
```

Optimistic but plausible outcome:

```text
0.0015 to 0.003 BPB prequant gain with negligible step-time loss.
```

Breakthrough outcome:

```text
The scout discovers that current 10-minute training systematically undercovers
specific token/document modes, giving a repeatable >= 0.005 nats gain.
```

The breakthrough case is not the base expectation. The reason to try is that the first test is cheap and the idea attacks a real bottleneck: finite gradient budget, not model capacity.

## Final decision

Yes, this is worth one controlled experiment.

But the implementation must be conservative:

```text
Do not let the scout become the training algorithm.
Let it be a small CPU-side bias on top of the strongest existing loader.
```

If it cannot win as a small, cheap bias, it probably will not win as a bigger complicated system.
