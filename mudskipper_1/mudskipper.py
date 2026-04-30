"""Mudskipper — soft per-token loss reweighting for L(N) training.

Training-time only. Not part of the 16MB submission artifact.

Two scorers, switchable via MUDSKIPPER_MODE:
  - "byte"   (v2, default): weight ∝ token byte-length. Aligns the training
             objective with the BPB eval metric. Signal is uncorrelated with
             model loss → no gradient-variance amplification. ~Zero overhead.
             See PR #1519 for prior art and DESIGN.md §0 for the v1→v2
             diagnosis.
  - "bigram" (v1, kept for ablation): weight ∝ bigram-surprise. FAILED on
             1×H100 (+0.0103 BPB worse than baseline). Mechanism is correct
             but the signal is positively correlated with per-token loss,
             which amplifies gradient variance.

Both scorers expose `weights(prev_ids, target_ids) -> (B, T) tensor` with
mean ≈ 1.0 inside each batch, so the unweighted training path stays
gradient-magnitude-comparable.
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
    # "byte" (v2 default) or "bigram" (v1, kept for ablation).
    mode: str = "byte"
    alpha: float = 1.0
    # Bigram-only: Laplace smoothing.
    smoothing: float = 1.0
    # Both: clamp the upper tail so a single high-weight token can't dominate.
    weight_clamp_max: float = 8.0
    # Byte-only: minimum byte count assigned to control tokens (BOS etc.) so
    # they still get a unit gradient signal. Matches PR #1519's clamp(min=1).
    byte_min: int = 1
    log_every: int = 200

    @classmethod
    def from_env(cls) -> "MudskipperConfig":
        return cls(
            enabled=bool(int(os.environ.get("MUDSKIPPER_ENABLED", "0"))),
            mode=os.environ.get("MUDSKIPPER_MODE", "byte"),
            # v1 used alpha=0.5 by default; v2 byte mode wants alpha=1.0
            # (linear in bytes, matching PR #1519 and the BPB metric exactly).
            alpha=float(os.environ.get("MUDSKIPPER_ALPHA", "1.0")),
            smoothing=float(os.environ.get("MUDSKIPPER_SMOOTHING", "1.0")),
            weight_clamp_max=float(os.environ.get("MUDSKIPPER_CLAMP_MAX", "8.0")),
            byte_min=int(os.environ.get("MUDSKIPPER_BYTE_MIN", "1")),
            log_every=int(os.environ.get("MUDSKIPPER_LOG_EVERY", "200")),
        )


def _normalize_mean_one(w: Tensor, clamp_max: float) -> Tensor:
    """Rescale to batch mean 1.0, then optionally cap the upper tail.

    The cap protects Muon's gradient-orthogonalization from being driven by a
    single outlier weight when alpha is large.
    """
    w = w / w.mean().clamp_min(1e-12)
    if clamp_max > 0:
        w = w.clamp(max=clamp_max)
        # Re-normalize so mean stays 1 after the cap.
        w = w / w.mean().clamp_min(1e-12)
    return w


class ByteScorer:
    """v2 scorer: per-token weight ∝ (byte count of target)^alpha.

    Uses the *full* byte-count formula that the eval metric uses:

        bytes(y_t | x_t) = base_bytes_lut[y_t]
                         + has_leading_space_lut[y_t] * (1 - is_boundary_token_lut[x_t])

    PR #1519 used only the first term. We use the full formula so the training
    weight per token equals the byte that token actually contributes to BPB.

    Stateless apart from the LUTs. Update is a no-op (kept for API symmetry
    with BigramScorer).
    """

    def __init__(
        self,
        vocab_size: int,
        device: torch.device,
        config: MudskipperConfig,
        *,
        base_bytes_lut: Tensor,
        has_leading_space_lut: Tensor,
        is_boundary_token_lut: Tensor,
    ) -> None:
        self.vocab_size = vocab_size
        self.device = device
        self.config = config
        # Cast to int32 once and keep on device for fast indexing.
        self.base_bytes = base_bytes_lut.to(device=device, dtype=torch.int32)
        self.has_leading_space = has_leading_space_lut.to(
            device=device, dtype=torch.bool
        )
        self.is_boundary_token = is_boundary_token_lut.to(
            device=device, dtype=torch.bool
        )
        self.tokens_observed = 0

    @torch.no_grad()
    def update(self, prev_ids: Tensor, target_ids: Tensor) -> None:
        # Stateless wrt training data; just track count for logging.
        self.tokens_observed += int(target_ids.numel())

    @torch.no_grad()
    def total_bytes(self, prev_ids: Tensor, target_ids: Tensor) -> Tensor:
        prev_long = prev_ids.to(torch.int64)
        targ_long = target_ids.to(torch.int64)
        target_bytes = self.base_bytes[targ_long].to(torch.float32)
        # Adjustment: an extra leading-space byte if the target is a leading-
        # space piece AND the previous token is not a doc boundary marker.
        # This matches eval_val()'s byte-counting exactly (see train_gpt.py:269).
        leading_space_adj = (
            self.has_leading_space[targ_long] & ~self.is_boundary_token[prev_long]
        ).to(torch.float32)
        total = target_bytes + leading_space_adj
        return total.clamp(min=float(self.config.byte_min))

    @torch.no_grad()
    def weights(self, prev_ids: Tensor, target_ids: Tensor) -> Tensor:
        if self.config.alpha == 0.0:
            # Uniform → equivalent to baseline; useful as a control.
            return torch.ones_like(target_ids, dtype=torch.float32)
        bytes_per_token = self.total_bytes(prev_ids, target_ids)
        w = bytes_per_token.pow(self.config.alpha)
        return _normalize_mean_one(w, self.config.weight_clamp_max)

    @torch.no_grad()
    def diagnostics(self, w: Tensor, prev_ids: Tensor, target_ids: Tensor) -> dict[str, float]:
        bpt = self.total_bytes(prev_ids, target_ids)
        targ_long = target_ids.to(torch.int64)
        boundary_frac = self.is_boundary_token[targ_long].float().mean()
        return {
            "weight_min": float(w.min().item()),
            "weight_max": float(w.max().item()),
            "weight_mean": float(w.mean().item()),
            "weight_std": float(w.std().item()),
            "bytes_p05": float(bpt.quantile(0.05).item()),
            "bytes_p50": float(bpt.quantile(0.5).item()),
            "bytes_p95": float(bpt.quantile(0.95).item()),
            "bytes_mean": float(bpt.mean().item()),
            "frac_boundary_targets": float(boundary_frac.item()),
            "tokens_observed": float(self.tokens_observed),
        }


class BigramScorer:
    """v1 scorer (kept for ablation only). Weight ∝ bigram surprise^alpha.

    FAILED on 1×H100: +0.0103 BPB worse than baseline. The signal correlates
    with per-token loss → gradient-variance amplification. See DESIGN.md §0.
    Use MUDSKIPPER_MODE=bigram only for reproducing the failure.
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
        self.counts = torch.zeros(
            (vocab_size, vocab_size), dtype=torch.int32, device=device
        )
        self.row_sums = torch.zeros((vocab_size,), dtype=torch.int32, device=device)
        if init_uniform:
            init_count = max(1, int(round(config.smoothing)))
            self.counts.fill_(init_count)
            self.row_sums.fill_(init_count * vocab_size)
        self.tokens_observed = 0

    @torch.no_grad()
    def update(self, prev_ids: Tensor, target_ids: Tensor) -> None:
        flat_prev = prev_ids.reshape(-1).to(torch.int64)
        flat_targ = target_ids.reshape(-1).to(torch.int64)
        pair_idx = flat_prev * self.vocab_size + flat_targ
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
        prev_long = prev_ids.to(torch.int64)
        targ_long = target_ids.to(torch.int64)
        joint = self.counts[prev_long, targ_long].to(torch.float32)
        margin = self.row_sums[prev_long].to(torch.float32)
        s = self.config.smoothing
        cond = (joint + s) / (margin + s * self.vocab_size)
        cond = cond.clamp_min(1e-12)
        return -cond.log()

    @torch.no_grad()
    def weights(self, prev_ids: Tensor, target_ids: Tensor) -> Tensor:
        if self.config.alpha == 0.0:
            return torch.ones_like(target_ids, dtype=torch.float32)
        s = self.surprise(prev_ids, target_ids)
        w = s.clamp_min(1e-6).pow(self.config.alpha)
        return _normalize_mean_one(w, self.config.weight_clamp_max)

    @torch.no_grad()
    def diagnostics(
        self, w: Tensor, prev_ids: Tensor, target_ids: Tensor
    ) -> dict[str, float]:
        s = self.surprise(prev_ids, target_ids)
        return {
            "weight_min": float(w.min().item()),
            "weight_max": float(w.max().item()),
            "weight_mean": float(w.mean().item()),
            "weight_std": float(w.std().item()),
            "surprise_p50": float(s.quantile(0.5).item()),
            "surprise_p95": float(s.quantile(0.95).item()),
            "tokens_observed": float(self.tokens_observed),
        }


def make_scorer(
    config: MudskipperConfig,
    vocab_size: int,
    device: torch.device,
    *,
    base_bytes_lut: Tensor | None = None,
    has_leading_space_lut: Tensor | None = None,
    is_boundary_token_lut: Tensor | None = None,
) -> ByteScorer | BigramScorer:
    if config.mode == "byte":
        if (
            base_bytes_lut is None
            or has_leading_space_lut is None
            or is_boundary_token_lut is None
        ):
            raise ValueError(
                "ByteScorer requires base_bytes_lut, has_leading_space_lut, "
                "is_boundary_token_lut. Build them with "
                "build_sentencepiece_luts() and pass them in."
            )
        return ByteScorer(
            vocab_size,
            device,
            config,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_leading_space_lut,
            is_boundary_token_lut=is_boundary_token_lut,
        )
    if config.mode == "bigram":
        return BigramScorer(vocab_size, device, config)
    raise ValueError(
        f"Unknown MUDSKIPPER_MODE={config.mode!r}; expected 'byte' or 'bigram'"
    )
