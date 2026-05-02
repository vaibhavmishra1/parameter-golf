# Dynamic Learning Allocator Ideas

This note collects three concrete versions of the same high-level idea:

> Keep the transformer compute path fixed, but allocate more gradient pressure to the parts of the batch that are most useful to learn from.

This is the cleaner successor to the original Mudskipper direction. It does not skip documents, does not use CPU-side handcrafted filtering, and does not change the number of layer recursions per document. The scarce resource it tries to allocate is not wallclock compute directly, but **gradient budget**.

## Implemented Scripts

The folder contains four scripts:

- `train_gpt.py`: untouched baseline copy from repo root.
- `train_gpt_loss_allocator.py`: Idea 1, fixed loss/entropy span allocator.
- `train_gpt_learned_allocator.py`: Idea 2, tiny learned span allocator.
- `train_gpt_gradient_allocator.py`: Idea 3, learned allocator with final-hidden gradient-norm usefulness proxy.

Each variant keeps the baseline architecture, optimizer, quantization path, and validation path intact except for the training loss allocator logic.

Common knobs:

```bash
ALLOC_ENABLED=1
ALLOC_SPAN_SIZE=64
ALLOC_ALPHA=0.10
ALLOC_WARMUP_STEPS=300
```

Variant-specific knobs:

```bash
# train_gpt_loss_allocator.py
ALLOC_USE_ENTROPY=0   # 0 = span CE, 1 = token entropy

# train_gpt_learned_allocator.py
ALLOC_AUX_WEIGHT=0.01

# train_gpt_gradient_allocator.py
ALLOC_AUX_WEIGHT=0.01
ALLOC_GRAD_POWER=1.0
```

## Idea 1: Loss / Entropy Allocator

### Core idea

Use the model's current token loss or entropy to upweight harder spans.

Split each packed batch into fixed-size spans, for example 64 tokens. Compute the normal next-token cross-entropy. Spans with higher average loss receive slightly larger loss weights; spans with lower average loss receive slightly smaller loss weights.

### Why it might work

Uniform CE spends the same gradient pressure on easy spans and hard spans. In a 10-minute training run, many examples are already easy or redundant by the time they appear. Reweighting toward harder spans may improve memorization of underfit parts of the data without changing the model architecture or data order.

This is cheap and easy to test.

### Main problem

High loss does not necessarily mean useful. It can also mean noisy text, tokenization artifacts, formatting junk, or impossible prediction. A pure entropy/loss allocator may overfit hard garbage.

It is also not really learned. It is a fixed rule:

```text
hard span -> more weight
easy span -> less weight
```

### Training algorithm

```text
Hyperparameters:
  span_size = 64
  alpha = 0.10
  warmup_steps = 300

For each training step:

  1. Run the normal transformer forward pass.

     logits = model(x)
     ce_tok = cross_entropy(logits, y, reduction="none")

  2. Split tokens into fixed spans.

     span_loss[i] = mean CE over span i

  3. Normalize span losses within the batch.

     z_i = (span_loss[i] - mean(span_loss)) / (std(span_loss) + eps)

  4. Convert normalized loss into mild weights.

     if step < warmup_steps:
        weight_i = 1
     else:
        weight_i = 1 + alpha * tanh(z_i)
        weight_i = weight_i / mean(weight)

  5. Train with weighted span loss.

     loss = mean_i(weight_i.detach() * span_loss[i])

  6. Backward and optimizer step as usual.
```

Expected weight range with `alpha=0.10` is roughly `0.9x` to `1.1x`. The mean weight is forced to `1.0`, so total gradient scale stays stable.

## Idea 2: Learned Span Allocator

### Core idea

Attach a tiny allocator head to the transformer's final hidden states. The allocator predicts which spans deserve more gradient pressure.

Instead of hardcoding:

```text
weight = function(current_loss)
```

we learn:

```text
span_hidden -> allocation_score
```

The allocator is trained to predict normalized span difficulty. Its output is then used to weight span losses.

### Why it might work

The final hidden state contains a dense model-native representation of the text. A learned head may produce smoother and more useful weights than a raw loss/entropy rule. It can also be implemented with almost no overhead: one small linear head over pooled span representations.

This keeps:

- same data,
- same transformer path,
- same recurrence,
- same training budget,
- no CPU filtering,
- no handcrafted document features.

### Main problem

If trained only to predict current span loss, this may still reduce to a learned version of "hard span = important span." That is better structured than raw entropy, but it does not fully solve the hard-junk problem.

The allocator must also be prevented from directly lowering weights on hard spans to reduce the main loss. The safe version detaches allocator weights from the main CE path and trains the allocator with a separate auxiliary loss.

### Training algorithm

```text
Hyperparameters:
  span_size = 64
  alpha = 0.10
  allocator_loss_weight = 0.01
  warmup_steps = 300

Model addition:
  allocator_head = Linear(model_dim, 1)

For each training step:

  1. Run transformer forward pass and return final hidden states.

     hidden, logits = model(x, return_hidden=True)
     ce_tok = cross_entropy(logits, y, reduction="none")

  2. Split tokens into fixed spans.

     span_loss[i] = mean CE over span i
     span_hidden[i] = mean hidden over span i

  3. Build detached supervision target from span loss.

     target_i = stopgrad(span_loss[i])
     target_z_i = zscore(target_i)

  4. Allocator predicts span score from hidden state.

     score_i = allocator_head(stopgrad(span_hidden[i]))
     score_z_i = zscore(score_i)

  5. Train allocator to predict relative span difficulty.

     allocator_loss = mean((score_z_i - target_z_i)^2)

  6. Convert allocator score into mild loss weights.

     if step < warmup_steps:
        weight_i = 1
     else:
        weight_i = 1 + alpha * tanh(stopgrad(score_z_i))
        weight_i = weight_i / mean(weight)

  7. Main weighted training loss.

     main_loss = mean_i(weight_i * span_loss[i])

  8. Final loss.

     loss = main_loss + allocator_loss_weight * allocator_loss

  9. Backward and optimizer step as usual.
```

The key safety detail is:

```text
weight_i uses stopgrad(score_z_i)
```

So the allocator cannot directly manipulate the main CE loss. It is trained only through `allocator_loss`.

## Idea 3: Gradient-Usefulness Allocator

### Core idea

Train the allocator to predict not just which spans are hard, but which spans produce useful gradients.

The useful signal is:

```text
hard span + gradient agrees with broader learning direction
```

This is meant to distinguish useful hard examples from hard junk.

### Why it might work

Loss or entropy says only:

```text
this span is hard
```

Gradient usefulness asks:

```text
does learning from this span push the model in a direction compatible with the rest of the batch?
```

If a span has high loss but its gradient points in a strange private direction, boosting it may waste memorization on noise. If a span has high loss and its gradient agrees with the batch gradient, it is more likely to be useful.

### Main problem

The exact version is expensive. Per-span gradient alignment requires many span-level gradient computations. If a batch has hundreds of spans, exact alignment can cost far more than the baseline training step.

This idea is the most principled but also the riskiest under the 10-minute budget.

### Exact training algorithm

```text
Hyperparameters:
  span_size = 64
  alpha = 0.10
  allocator_loss_weight = 0.01
  warmup_steps = 500

Model addition:
  allocator_head = Linear(model_dim, 1)

For each training step:

  1. Run transformer forward pass.

     hidden, logits = model(x, return_hidden=True)
     ce_tok = cross_entropy(logits, y, reduction="none")

  2. Split into spans.

     span_loss[i] = mean CE over span i
     span_hidden[i] = mean hidden over span i

  3. Compute full-batch gradient direction with respect to hidden states.

     batch_loss = mean_i(span_loss[i])
     g_batch = grad(batch_loss, hidden, retain_graph=True)

  4. For each span, compute its own hidden-state gradient.

     g_i = grad(span_loss[i], hidden_span_i, retain_graph=True)

  5. Compute gradient alignment.

     alignment_i = cosine(g_i, g_batch over same span)

  6. Build usefulness target.

     target_i = stopgrad(span_loss[i] * relu(alignment_i))
     target_z_i = zscore(target_i)

  7. Train allocator to predict usefulness.

     score_i = allocator_head(stopgrad(span_hidden[i]))
     score_z_i = zscore(score_i)
     allocator_loss = mean((score_z_i - target_z_i)^2)

  8. Convert allocator score into mild weights.

     if step < warmup_steps:
        weight_i = 1
     else:
        weight_i = 1 + alpha * tanh(stopgrad(score_z_i))
        weight_i = weight_i / mean(weight)

  9. Weighted main loss.

     main_loss = mean_i(weight_i * span_loss[i])

  10. Final loss and optimizer step.

      loss = main_loss + allocator_loss_weight * allocator_loss
      loss.backward()
      optimizer.step()
```

### Practical cheaper variant

Exact per-span gradients may be too slow. A cheaper proxy is to use hidden-state gradient magnitude from the normal backward pass.

```text
target_i = stopgrad(span_loss[i] * mean_norm(hidden_grad over span i))
```

Training sketch:

```text
For each step:

  1. Forward pass with hidden states retained.
  2. Compute normal unweighted CE.
  3. Backward enough to obtain hidden.grad.
  4. Build per-span gradient-strength targets from hidden.grad.
  5. Train allocator to predict those targets.
  6. On later steps, use allocator weights in the main loss.
```

This is less clean than true alignment, but it has a better chance of fitting into the training budget.

## Recommended Order Of Experiments

1. **Loss / Entropy Allocator**
   - Establish the cheap baseline.
   - If this hurts, aggressive dynamic weighting is likely dangerous.

2. **Learned Span Allocator**
   - Same mild weight range.
   - Test whether hidden-state prediction beats raw loss/entropy weighting.

3. **Gradient-Usefulness Allocator**
   - Only worth trying if the first two show some signal.
   - Start with the cheap gradient proxy, not exact per-span alignment.

The success criterion should be strict:

```text
allocator must beat the same seed baseline, not just look plausible
```

If it only matches entropy/loss weighting, the learned allocator is not worth its complexity.
