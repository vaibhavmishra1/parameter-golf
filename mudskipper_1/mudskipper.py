"""Mudskipper v1 — soft per-token reweighting for L(N) training.

Training-time only. Not part of the 16MB submission artifact.

The single mechanism: maintain a running on-device bigram count table over
training tokens; for each batch, derive a per-token weight from
`-log P_bigram(y_t | x_t)` raised to a power `alpha`; multiply per-token
cross-entropy by that weight (with mean-1 normalization) before averaging.

This avoids the failure modes documented by:
  - PR #772 (chunk-level hard masking → +0.0072 BPB worse, "removes diverse tail")
  - PR #737 (online entropy-curriculum hard filtering → +0.021 BPB worse,
            verdict: "curriculum at this scale must be zero-overhead")

Mudskipper preserves every token (no masking) and adds essentially zero
per-step overhead (one gather + one elementwise log). See DESIGN.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class MudskipperConfig:
    """Runtime config, populated from environment variables."""

    enabled: bool = False
    alpha: float = 0.5
    smoothing: float = 1.0
    weight_clamp_max: float = 8.0
    log_every: int = 200

    @classmethod
    def from_env(cls) -> "MudskipperConfig":
        return cls(
            enabled=bool(int(os.environ.get("MUDSKIPPER_ENABLED", "0"))),
            alpha=float(os.environ.get("MUDSKIPPER_ALPHA", "0.5")),
            smoothing=float(os.environ.get("MUDSKIPPER_SMOOTHING", "1.0")),
            weight_clamp_max=float(os.environ.get("MUDSKIPPER_CLAMP_MAX", "8.0")),
            log_every=int(os.environ.get("MUDSKIPPER_LOG_EVERY", "200")),
        )


class BigramScorer:
    """Running on-device bigram counter that emits per-token loss weights.

    For vocab=1024 the count table is 1024*1024*4 = 4MB (int32). For vocab=8192
    it would be 256MB — still acceptable but worth noting before scaling up.
    """

    def __init__(
        self,
        vocab_size: int,
        device: torch.device,
        config: MudskipperConfig,
        *,
        init_uniform: bool = True,
    ) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.config = config
        # int32 is plenty for ~10B-token runs (max count per cell <<2^31).
        self.counts = torch.zeros(
            (vocab_size, vocab_size), dtype=torch.int32, device=device
        )
        self.row_sums = torch.zeros((vocab_size,), dtype=torch.int32, device=device)
        if init_uniform:
            # Seed every cell with `smoothing` so the first batch is not all-NaN.
            # Counts stay int32 by storing only integer additions.
            init_count = max(1, int(round(config.smoothing)))
            self.counts.fill_(init_count)
            self.row_sums.fill_(init_count * vocab_size)
        # Diagnostic accumulators.
        self.tokens_observed = 0

    @torch.no_grad()
    def update(self, prev_ids: Tensor, target_ids: Tensor) -> None:
        """Increment bigram counts for a batch of (prev, target) pairs.

        Both inputs are (B, T) int64 token id tensors on the same device.
        """
        flat_prev = prev_ids.reshape(-1).to(torch.int64)
        flat_targ = target_ids.reshape(-1).to(torch.int64)
        # Encode pair as a single linear index for bincount.
        pair_idx = flat_prev * self.vocab_size + flat_targ
        # Bincount, then scatter-add into the 2D table.
        bincounts = torch.bincount(
            pair_idx, minlength=self.vocab_size * self.vocab_size
        ).to(torch.int32)
        self.counts.add_(bincounts.view(self.vocab_size, self.vocab_size))
        prev_bincount = torch.bincount(flat_prev, minlength=self.vocab_size).to(
            torch.int32
        )
        self.row_sums.add_(prev_bincount)
        self.tokens_observed += int(flat_targ.numel())

    @torch.no_grad()
    def surprise(self, prev_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Per-token bigram surprise in nats. Shape matches input (B, T).

        Uses Laplace-smoothed conditional probability:
            P(y | x) = (counts[x,y] + s) / (row_sums[x] + s * V)
        """
        prev_long = prev_ids.to(torch.int64)
        targ_long = target_ids.to(torch.int64)
        # Per-element joint and marginal lookups.
        joint = self.counts[prev_long, targ_long].to(torch.float32)
        margin = self.row_sums[prev_long].to(torch.float32)
        s = self.config.smoothing
        cond = (joint + s) / (margin + s * self.vocab_size)
        # Clamp to avoid -inf if cond underflows for some reason.
        cond = cond.clamp_min(1e-12)
        return -cond.log()

    @torch.no_grad()
    def weights(self, prev_ids: Tensor, target_ids: Tensor) -> Tensor:
        """Per-token loss weight in (~0, clamp_max], with batch mean ≈ 1.0."""
        s = self.surprise(prev_ids, target_ids)
        # alpha=0 returns uniform 1.0, alpha=1 is linear-in-surprise.
        if self.config.alpha == 0.0:
            return torch.ones_like(s)
        w = s.clamp_min(1e-6).pow(self.config.alpha)
        # Normalize to mean 1.0 within the batch — preserves expected gradient
        # magnitude per step against the unweighted baseline.
        w = w / w.mean().clamp_min(1e-12)
        # Cap the upper tail so a single rare-pair token doesn't dominate the
        # batch loss. Renormalize after clamp to keep mean ≈ 1.0.
        if self.config.weight_clamp_max > 0:
            w = w.clamp(max=self.config.weight_clamp_max)
            w = w / w.mean().clamp_min(1e-12)
        return w

    @torch.no_grad()
    def diagnostics(self, w: Tensor, surprise: Tensor) -> dict[str, float]:
        """Return summary stats for logging."""
        return {
            "weight_min": float(w.min().item()),
            "weight_max": float(w.max().item()),
            "weight_mean": float(w.mean().item()),
            "weight_std": float(w.std().item()),
            "surprise_p50": float(surprise.quantile(0.5).item()),
            "surprise_p95": float(surprise.quantile(0.95).item()),
            "tokens_observed": float(self.tokens_observed),
        }
