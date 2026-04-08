"""Weight-Tying + Depth-RoPE Transformer for parameter-golf.

COMBINES both ideas from the other two files:

1. WEIGHT TYING (from train_gpt_weight_tying.py)
   - stem (2 unique blocks) → shared_block (×7, weight tied) → tail (2 unique blocks)
   - 5 unique blocks vs 11 baseline → ~55% parameter savings in block weights
   - Freed budget reinvested: increase MODEL_DIM (try 576 or 640)

2. DEPTH ROPE (from train_gpt_depth_rope.py)
   - The shared_block receives depth_idx (0,1,2,...,6) at each recursion pass
   - A DepthRotary applies a fixed sinusoidal rotation to Q and K
   - depth=0 is identity; each subsequent pass gets progressively rotated Q/K
   - Attention patterns can therefore DIFFER across recursion passes despite
     having identical weight matrices

SYNERGY: Weight tying saves parameters and forces the shared block to be
general. Depth-RoPE breaks the symmetry between passes — giving the shared
block a "clock signal" so it can specialise each iteration of computation
even though the weights are shared.

Architecture:
  stem[0] → stem[1]                               (enc, unique)
  shared_block × 3 [depth 0,1,2]                  (enc, shared + depth)
  shared_block × 4 [depth 3,4,5,6] + skip inject  (dec, shared + depth)
  tail[0] + skip inject → tail[1] + skip inject    (dec, unique)

  Effective depth:  2 + 7 + 2 = 11
  Unique blocks:    2 + 1 + 2 = 5  (vs 11 baseline)

Tuning knobs:
  NUM_STEM_LAYERS   (default 2)  — unique front blocks
  NUM_SHARED_LOOPS  (default 7)  — how many times the shared block runs
  NUM_TAIL_LAYERS   (default 2)  — unique back blocks
  MODEL_DIM         (default 512, try 576/640 to reinvest parameter savings)
  DEPTH_ROPE_DIMS   (default 16) — head_dim dimensions used for depth encoding
  DEPTH_ROPE_BASE   (default 100) — frequency base (lower = more spread per step)
"""

import collections
import copy
import glob
import io
import lzma
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import uuid

import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import Tensor, nn

from flash_attn_interface import flash_attn_func as flash_attn_3_func
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

# ----------------------------------------
# Hyperparameters
# ----------------------------------------

class Hyperparameters():
    data_dir = os.environ.get('DATA_DIR', './data/')
    seed = int(os.environ.get('SEED', 1337))
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))

    iterations = int(os.environ.get('ITERATIONS', 20000))
    warmdown_frac = float(os.environ.get('WARMDOWN_FRAC', 0.667))
    warmup_steps = int(os.environ.get('WARMUP_STEPS', 20))
    train_batch_tokens = int(os.environ.get('TRAIN_BATCH_TOKENS', 2048 * 48 * 8))
    train_seq_len = int(os.environ.get('TRAIN_SEQ_LEN', 2048))
    train_log_every = int(os.environ.get('TRAIN_LOG_EVERY', 500))
    max_wallclock_seconds = float(os.environ.get('MAX_WALLCLOCK_SECONDS', 600.0))

    val_batch_tokens = int(os.environ.get('VAL_BATCH_TOKENS', 2048 * 32 * 8))
    eval_seq_len = int(os.environ.get('EVAL_SEQ_LEN', 2048))
    val_loss_every = int(os.environ.get('VAL_LOSS_EVERY', 4000))
    sliding_window_enabled = bool(int(os.environ.get('SLIDING_WINDOW_ENABLED', '1')))

    vocab_size = int(os.environ.get('VOCAB_SIZE', 8192))

    # === WEIGHT-TYING PARAMS ===
    num_stem_layers  = int(os.environ.get('NUM_STEM_LAYERS',  2))
    num_shared_loops = int(os.environ.get('NUM_SHARED_LOOPS', 7))
    num_tail_layers  = int(os.environ.get('NUM_TAIL_LAYERS',  2))
    # With 5 unique blocks vs 11, parameter savings allow wider model.
    # At dim=576: 5 × 11 × 576^2 ≈ 18.2M block params vs SOTA 11 × 11 × 512^2 ≈ 31.5M
    model_dim     = int(os.environ.get('MODEL_DIM', 512))     # try 576 or 640
    embedding_dim = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads  = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads     = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult      = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap  = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base      = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims      = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale       = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init   = float(os.environ.get('QK_GAIN_INIT', 4.0))
    xsa_last_n     = int(os.environ.get('XSA_LAST_N', 2))

    # === DEPTH ROPE PARAMS ===
    # Applied to the SHARED block at each recursion pass.
    # Stem and tail blocks use depth_idx=0 (identity rotation — no effect).
    # depth_rope_dims occupies head_dim slots AFTER seq_rope_dims (no overlap).
    # With head_dim=64, rope_dims=16: 48 free dims → 32 for depth = strong signal.
    depth_rope_dims = int(os.environ.get('DEPTH_ROPE_DIMS', 32))
    # base=10: θ_0 = 1.0 rad per depth step → ~57° between consecutive passes.
    # With num_shared_loops=7 (depths 0-6), this gives highly distinct rotations.
    depth_rope_base = float(os.environ.get('DEPTH_ROPE_BASE', 10.0))
    # max_depth should be >= num_shared_loops
    max_depth = int(os.environ.get('MAX_DEPTH', 16))

    # Optimizer
    min_lr = float(os.environ.get('MIN_LR', 0.0))
    embed_lr = float(os.environ.get('EMBED_LR', 0.6))
    head_lr = float(os.environ.get('HEAD_LR', 0.008))
    tied_embed_lr = float(os.environ.get('TIED_EMBED_LR', 0.03))
    tied_embed_init_std = float(os.environ.get('TIED_EMBED_INIT_STD', 0.005))
    matrix_lr = float(os.environ.get('MATRIX_LR', 0.02))
    scalar_lr = float(os.environ.get('SCALAR_LR', 0.02))
    muon_momentum = float(os.environ.get('MUON_MOMENTUM', 0.99))
    muon_backend_steps = int(os.environ.get('MUON_BACKEND_STEPS', 5))
    muon_momentum_warmup_start = float(os.environ.get('MUON_MOMENTUM_WARMUP_START', 0.92))
    muon_momentum_warmup_steps = int(os.environ.get('MUON_MOMENTUM_WARMUP_STEPS', 1500))
    muon_row_normalize = bool(int(os.environ.get('MUON_ROW_NORMALIZE', '1')))
    beta1 = float(os.environ.get('BETA1', 0.9))
    beta2 = float(os.environ.get('BETA2', 0.95))
    adam_eps = float(os.environ.get('ADAM_EPS', 1e-8))
    grad_clip_norm = float(os.environ.get('GRAD_CLIP_NORM', 0.3))
    eval_stride = int(os.environ.get('EVAL_STRIDE', 64))
    muon_beta2 = float(os.environ.get('MUON_BETA2', 0.95))
    adam_wd = float(os.environ.get('ADAM_WD', 0.02))
    muon_wd = float(os.environ.get('MUON_WD', 0.085))
    embed_wd = float(os.environ.get('EMBED_WD', 0.085))
    ema_decay = float(os.environ.get('EMA_DECAY', 0.997))

    compressor = os.environ.get('COMPRESSOR', 'brotli')
    gptq_calibration_batches = int(os.environ.get('GPTQ_CALIBRATION_BATCHES', 64))
    gptq_reserve_seconds = float(os.environ.get('GPTQ_RESERVE_SECONDS', 12.0))
    matrix_bits = int(os.environ.get('MATRIX_BITS', 6))
    embed_bits = int(os.environ.get('EMBED_BITS', 8))
    matrix_clip_sigmas = float(os.environ.get('MATRIX_CLIP_SIGMAS', 13.5))
    embed_clip_sigmas = float(os.environ.get('EMBED_CLIP_SIGMAS', 20.0))

    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_main_process = rank == 0
    grad_accum_steps = 8 // world_size

    datasets_dir = os.path.join(data_dir, 'datasets', f'fineweb10B_sp{vocab_size}')
    train_files = os.path.join(datasets_dir, 'fineweb_train_*.bin')
    val_files = os.path.join(datasets_dir, 'fineweb_val_*.bin')
    tokenizer_path = os.path.join(data_dir, 'tokenizers', f'fineweb_{vocab_size}_bpe.model')

    logfile = f"logs/{run_id}.txt"
    model_path = "final_model.pt"
    quantized_model_path = "final_model.int6.ptz"

# ----------------------------------------
# Logging
# ----------------------------------------

_logger_hparams = None
def set_logging_hparams(h): global _logger_hparams; _logger_hparams = h
def log(msg, console=True):
    if _logger_hparams is None: print(msg); return
    if _logger_hparams.is_main_process:
        if console: print(msg)
        if _logger_hparams.logfile:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f: print(msg, file=f)

# ----------------------------------------
# Data Loading  (unchanged)
# ----------------------------------------

class ValidationData:
    def __init__(self, h, device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size: raise ValueError("vocab_size mismatch")
        self.val_tokens = load_validation_tokens(h.val_files, h.eval_seq_len)
        self.base_bytes_lut, self.has_leading_space_lut, self.is_boundary_token_lut = (
            build_sentencepiece_luts(self.sp, h.vocab_size, device))

def build_sentencepiece_luts(sp, vocab_size, device):
    sp_vocab_size = int(sp.vocab_size())
    assert sp.piece_to_id("\u2581") != sp.unk_id()
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id): continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id): base_bytes_np[token_id] = 1; continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"): has_leading_space_np[token_id] = True; piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
            torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
            torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device))

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files: raise FileNotFoundError(f"No files: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    return tokens[:(((tokens.numel()-1)//seq_len)*seq_len)+1]

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    return torch.from_numpy(np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes).astype(np.uint16, copy=False))

_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file):
    key = str(file)
    if key not in _SHARD_NTOKENS_CACHE:
        _SHARD_NTOKENS_CACHE[key] = int(np.fromfile(file, dtype="<i4", count=256)[2])
    return _SHARD_NTOKENS_CACHE[key]

def _get_shard_memmap(file):
    key = str(file)
    if key not in _MMAP_CACHE:
        _MMAP_CACHE[key] = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(_read_num_tokens(file),))
    return _MMAP_CACHE[key]

class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size; self.seq_len = h.train_seq_len; self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files: raise FileNotFoundError(f"No files: {h.train_files}")
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds = [[] for _ in self.files]
        for si in range(len(self.files)): self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        self.start_inds[si] = (phase + self.rng.permutation(num_sequences) * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)): self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64); total = remaining.sum()
            si = int(self.rng.choice(len(self.files), p=remaining / total))
            start_ind = self.start_inds[si].pop(); remaining[si] -= 1
            window = torch.as_tensor(np.array(_get_shard_memmap(self.files[si])[start_ind:start_ind+self.seq_len+1], dtype=np.int64))
            x[bi] = window[:-1]; y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ----------------------------------------
# Fused MLP Kernels (unchanged)
# ----------------------------------------

if HAS_TRITON:
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def _fused_leaky_relu_sq_tma_kernel(
        a_desc, b_desc, c_desc, aux_desc, M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    ):
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_n = tl.cdiv(N, BLOCK_N); num_tiles = tl.cdiv(M, BLOCK_M) * num_pid_n
        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n; pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_M; offs_bn = pid_n * BLOCK_N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(tl.cdiv(K, BLOCK_K)):
                a = a_desc.load([offs_am, ki * BLOCK_K]); b = b_desc.load([offs_bn, ki * BLOCK_K])
                accumulator = tl.dot(a, b.T, accumulator)
            acc = tl.permute(tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2)), (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            for acc_half, off_n in [(acc0, offs_bn), (acc1, offs_bn + BLOCK_N // 2)]:
                c_half = acc_half.to(dtype)
                c_ag = tl.where(c_half > 0, 2.0 * c_half, 0.5 * c_half)
                c_desc.store([offs_am, off_n], c_ag)
                aux_desc.store([offs_am, off_n], 0.5 * c_ag * c_half)

    def _triton_fused_leaky_relu_sq(x_flat, fc_weight):
        M, K = x_flat.shape; N, _ = fc_weight.shape
        act_grad = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        post = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        a_desc = TensorDescriptor.from_tensor(x_flat, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(fc_weight, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(act_grad, [BLOCK_M, BLOCK_N // 2])
        aux_desc = TensorDescriptor.from_tensor(post, [BLOCK_M, BLOCK_N // 2])
        def grid(META): return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)
        _fused_leaky_relu_sq_tma_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, num_stages=4, num_warps=8)
        return post, act_grad

    class _FusedMLP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, fc_w, proj_w):
            x_flat = x.reshape(-1, x.shape[-1])
            post, act_grad = _triton_fused_leaky_relu_sq(x_flat, fc_w)
            out = F.linear(post, proj_w)
            ctx.save_for_backward(x_flat, fc_w, proj_w, act_grad, post)
            ctx.orig_shape = x.shape
            return out.reshape(*x.shape[:-1], out.shape[-1])
        @staticmethod
        def backward(ctx, grad_output):
            x_flat, fc_w, proj_w, act_grad, post = ctx.saved_tensors
            go = grad_output.reshape(-1, grad_output.shape[-1])
            dpre = (go @ proj_w) * act_grad
            return (dpre @ fc_w).reshape(ctx.orig_shape), dpre.T @ x_flat, go.T @ post

# ----------------------------------------
# Model Architecture  (KEY CHANGES: both weight tying and depth rope)
# ----------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None): super().__init__(); self.eps = eps
    def forward(self, x): return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    """Sequence-position RoPE (unchanged)."""
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim; self.base = base; self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        self.register_buffer("inv_freq",
            1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims)),
            persistent=False)
        self._seq_len_cached = 0; self._cos_cached = None; self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                inv_freq = 1.0 / ((self.base * (scale ** (rd/(rd-2)))) ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            freqs = torch.outer(torch.arange(seq_len, device=device, dtype=inv_freq.dtype), inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)


class DepthRotary(nn.Module):
    """
    Fixed sinusoidal depth-position encodings for Q/K.

    Encodes recursion depth (0..num_shared_loops-1) as a rotation of a
    dedicated slice of each attention head's dimensions.

    CRITICAL — non-overlapping with sequence RoPE:
      Sequence RoPE uses dims [0 : seq_rope_dims].
      Depth   RoPE uses dims [seq_rope_dims : seq_rope_dims + depth_rope_dims].
    With head_dim=64, seq_rope_dims=16, depth_rope_dims=32:
      [0:16]  → sequence position   (sequence RoPE)
      [16:48] → recursion depth     (depth RoPE)  ← 32 clean dims
      [48:64] → unconstrained

    depth=0 → identity (cos=1, sin=0) — neutral for non-looping passes.
    depth=d → progressive rotation per pass — depth-aware attention.
    Zero trainable parameters.
    """
    def __init__(self, head_dim: int, seq_rope_dims: int = 16, max_depth: int = 16,
                 depth_rope_base: float = 100.0, depth_rope_dims: int = 32):
        super().__init__()
        self.start = seq_rope_dims
        rdim = min(depth_rope_dims, head_dim - seq_rope_dims)
        self.rdim = rdim
        inv_freq = 1.0 / (depth_rope_base ** (torch.arange(0, rdim, 2, dtype=torch.float32) / rdim))
        depths = torch.arange(max_depth, dtype=torch.float32)
        freqs = torch.outer(depths, inv_freq)
        self.register_buffer("depth_cos", freqs.cos(), persistent=False)
        self.register_buffer("depth_sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, depth_idx: int) -> Tensor:
        """
        x: [B, T, H, head_dim] — Q or K after sequence RoPE has been applied.
        depth_idx: Python int, compile-time constant when loop is unrolled.
        Rotates dims [seq_rope_dims : seq_rope_dims+rdim] only.
        """
        rdim = self.rdim; start = self.start
        cos = self.depth_cos[depth_idx]   # [rdim//2]
        sin = self.depth_sin[depth_idx]   # [rdim//2]
        half = rdim // 2
        x_pre  = x[..., :start]
        x_d    = x[..., start:start + rdim]
        x_post = x[..., start + rdim:]
        x1, x2 = x_d[..., :half], x_d[..., half:]
        x_rotated = torch.cat([x1*cos - x2*sin, x1*sin + x2*cos], dim=-1)
        return torch.cat([x_pre, x_rotated, x_post], dim=-1)


def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2; x1, x2 = x_rope[..., :half], x_rope[..., half:]
        return torch.cat([torch.cat([x1*cos+x2*sin, x1*(-sin)+x2*cos], dim=-1), x_pass], dim=-1)
    half = x.size(-1) // 2; x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1*cos+x2*sin, x1*(-sin)+x2*cos], dim=-1)


class CausalSelfAttention(nn.Module):
    """
    Attention that optionally applies depth-aware RoPE to Q and K.
    use_depth_rope=True  → shared block (rotated by depth_idx each pass)
    use_depth_rope=False → stem/tail blocks (depth_idx=0 = identity, no-op)
    """
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
                 use_depth_rope=False, depth_rope_base=100.0, depth_rope_dims=16, max_depth=16):
        super().__init__()
        assert dim % num_heads == 0 and num_heads % num_kv_heads == 0
        self.num_heads = num_heads; self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q  = CastedLinear(dim, dim, bias=False)
        self.c_k  = CastedLinear(dim, kv_dim, bias=False)
        self.c_v  = CastedLinear(dim, kv_dim, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False
        # Depth RoPE — only instantiated for the shared block.
        # seq_rope_dims is 0 here (rope_dims set externally by make_block);
        # make_block rebuilds depth_rotary after setting attn.rope_dims.
        self.use_depth_rope = use_depth_rope
        if use_depth_rope:
            self.depth_rotary = DepthRotary(
                self.head_dim,
                seq_rope_dims=0,   # placeholder; rebuilt by make_block
                max_depth=max_depth,
                depth_rope_base=depth_rope_base, depth_rope_dims=depth_rope_dims)

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape; Hkv = v.size(-2); group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        return (y_g - (y_g * vn).sum(dim=-1, keepdim=True) * vn).reshape(B, T, H, D)

    def forward(self, x: Tensor, depth_idx: int = 0) -> Tensor:
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),)); k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        # Depth rotation: only active on the shared block; depth_idx is a
        # compile-time constant since loops are unrolled by torch.compile.
        if self.use_depth_rope:
            q = self.depth_rotary(q, depth_idx)
            k = self.depth_rotary(k, depth_idx)
        q = q * self.q_gain.to(q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa: y = self._xsa_efficient(y, v)
        return self.proj(y.reshape(bsz, seqlen, dim))


class MLP(nn.Module):
    def __init__(self, dim, mlp_mult):
        super().__init__()
        hidden = int(mlp_mult * dim)
        self.fc   = CastedLinear(dim, hidden, bias=False)
        self.proj = CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
    def forward(self, x):
        if HAS_TRITON and x.is_cuda and self.training:
            return _FusedMLP.apply(x, self.fc.weight.to(x.dtype), self.proj.weight.to(x.dtype))
        return self.proj(F.leaky_relu(self.fc(x), negative_slope=0.5).square())


class Block(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, mlp_mult, rope_base, qk_gain_init,
                 train_seq_len, layer_idx=0, ln_scale=False, parallel=False,
                 use_depth_rope=False, depth_rope_base=100.0, depth_rope_dims=16, max_depth=16):
        super().__init__()
        self.parallel = parallel
        self.attn_norm = RMSNorm(); self.mlp_norm = RMSNorm()
        self.attn = CausalSelfAttention(
            dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len,
            use_depth_rope=use_depth_rope, depth_rope_base=depth_rope_base,
            depth_rope_dims=depth_rope_dims, max_depth=max_depth)
        self.mlp = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0
        # Learned additive depth embedding: depth_emb[d] is added to the residual
        # stream before this block runs at depth d. Zero-init → no effect at start.
        # Affects both attention (via attn_norm) and MLP (via mlp_norm).
        self.depth_emb = nn.Parameter(torch.zeros(max_depth, dim)) if use_depth_rope else None

    def forward(self, x: Tensor, x0: Tensor, depth_idx: int = 0) -> Tensor:
        mix = self.resid_mix.to(x.dtype)
        x_in = mix[0][None,None,:] * x + mix[1][None,None,:] * x0
        # Additive depth signal — informs both attention and MLP which pass this is
        if self.depth_emb is not None:
            x_in = x_in + self.depth_emb[depth_idx].to(x_in.dtype)[None, None, :]
        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed, depth_idx)
        if self.parallel:
            return x_in + self.attn_scale.to(x_in.dtype)[None,None,:] * attn_out \
                        + self.mlp_scale.to(x_in.dtype)[None,None,:] * self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
        x_out = x_in + self.attn_scale.to(x_in.dtype)[None,None,:] * attn_out
        return x_out + self.mlp_scale.to(x_out.dtype)[None,None,:] * self.mlp(self.mlp_norm(x_out) * self.ln_scale_factor)


# === COMBINED: Weight-Tied + Depth-RoPE GPT ===
class GPT(nn.Module):
    """
    stem_blocks (unique, no depth rope)
       ↓
    shared_block × num_shared_loops  [depth_idx = 0,1,...,num_shared_loops-1]
       ↓  (weight tied — same nn.Module, different depth_idx each call)
    tail_blocks (unique, no depth rope)

    The shared_block uses DepthRotary to rotate Q/K differently at each pass,
    breaking the weight-symmetry that would otherwise make all recursion steps
    functionally identical.

    Skip connections span the encoder/decoder midpoint as in the U-Net design.
    """
    def __init__(self, h: Hyperparameters):
        super().__init__()
        assert h.logit_softcap > 0.0
        self.tie_embeddings = h.tie_embeddings; self.logit_softcap = h.logit_softcap
        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj  = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None; self.head_proj = None

        def make_block(layer_idx, parallel=False, use_depth_rope=False):
            b = Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                      h.rope_base, h.qk_gain_init, h.train_seq_len,
                      layer_idx=layer_idx, ln_scale=h.ln_scale, parallel=parallel,
                      use_depth_rope=use_depth_rope,
                      depth_rope_base=h.depth_rope_base,
                      depth_rope_dims=h.depth_rope_dims,
                      max_depth=h.max_depth)
            if h.rope_dims > 0:
                b.attn.rope_dims = h.rope_dims
                b.attn.rotary = Rotary(b.attn.head_dim, base=h.rope_base,
                                       train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
                # Rebuild DepthRotary now that seq_rope_dims is known, so depth
                # dims start AFTER sequence dims — clean non-overlapping encoding.
                if use_depth_rope:
                    b.attn.depth_rotary = DepthRotary(
                        b.attn.head_dim,
                        seq_rope_dims=h.rope_dims,
                        max_depth=h.max_depth,
                        depth_rope_base=h.depth_rope_base,
                        depth_rope_dims=h.depth_rope_dims,
                    )
            return b

        # Stem: unique, no depth rope (always depth_idx=0 → identity rotation)
        self.stem_blocks = nn.ModuleList([make_block(i) for i in range(h.num_stem_layers)])

        # Shared: ONE block with depth rope, applied num_shared_loops times
        # layer_idx set to middle of network for ln_scale
        mid_idx = h.num_stem_layers + h.num_shared_loops // 2
        self.shared_block = make_block(mid_idx, use_depth_rope=True)   # ← depth rope enabled

        # Tail: unique, parallel residual + XSA on the last few
        total_tail = h.num_tail_layers
        tail_parallel_start = max(0, total_tail - 2)
        self.tail_blocks = nn.ModuleList([
            make_block(h.num_stem_layers + h.num_shared_loops + i,
                       parallel=(i >= tail_parallel_start))
            for i in range(total_tail)
        ])
        if h.xsa_last_n > 0:
            for i in range(max(0, total_tail - h.xsa_last_n), total_tail):
                self.tail_blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)

        # Encoder/decoder split for skip connections
        # enc: stem + first half of shared loops
        # dec: second half of shared loops + tail
        self.num_shared_enc = h.num_shared_loops // 2
        self.num_shared_dec = h.num_shared_loops - self.num_shared_enc
        n_enc = h.num_stem_layers + self.num_shared_enc
        n_dec = self.num_shared_dec + h.num_tail_layers
        self.num_skip_weights = min(n_enc, n_dec)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, h.model_dim, dtype=torch.float32))
        self.skip_gates   = nn.Parameter(torch.zeros(self.num_skip_weights, h.model_dim, dtype=torch.float32)) \
                            if h.skip_gates_enabled else None

        self._init_weights(h)

    def _init_weights(self, h):
        if self.tie_embeddings: nn.init.normal_(self.tok_emb.weight, mean=0.0, std=h.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False): nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _inject_skip(self, x, skip_idx, skips):
        scaled = self.skip_weights[skip_idx].to(x.dtype)[None, None, :] * skips.pop()
        if self.skip_gates is not None:
            return torch.lerp(scaled, x, torch.sigmoid(self.skip_gates[skip_idx].to(x.dtype))[None, None, :])
        return x + scaled

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None: x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []

        # === ENCODER: stem (unique, depth=0) ===
        for block in self.stem_blocks:
            x = block(x, x0, 0)   # depth_idx=0, compile-time const
            skips.append(x)

        # === ENCODER: shared block, first half of loops ===
        # Each iteration uses a different depth_idx → different Q/K rotation
        # The loop is unrolled by torch.compile; depth_idx is a compile-time constant per iteration.
        for d in range(self.num_shared_enc):
            x = self.shared_block(x, x0, d)   # d is Python int, compile-time const
            skips.append(x)

        # === DECODER: shared block, second half of loops ===
        skip_idx = 0
        for d in range(self.num_shared_dec):
            depth = self.num_shared_enc + d    # continue counting from where enc left off
            if skip_idx < self.num_skip_weights:
                x = self._inject_skip(x, skip_idx, skips); skip_idx += 1
            x = self.shared_block(x, x0, depth)

        # === DECODER: tail blocks (unique, depth=0) ===
        for block in self.tail_blocks:
            if skip_idx < self.num_skip_weights:
                x = self._inject_skip(x, skip_idx, skips); skip_idx += 1
            x = block(x, x0, 0)

        x = self.final_norm(x)
        if self.head_proj is not None: x = self.head_proj(x)
        logits_proj = F.linear(x, self.tok_emb.weight) if self.tie_embeddings else self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        return F.cross_entropy(self.forward_logits(input_ids).reshape(-1, self.tok_emb.weight.size(0)).float(),
                               target_ids.reshape(-1), reduction="mean")


def classify_param(name):
    if "tok_emb" in name or "lm_head" in name: return "embed"
    if ".mlp." in name: return "mlp"
    if ".attn." in name: return "attn"
    return "other"

# ----------------------------------------
# Optimization
# ----------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = tuple(
    p for p in os.environ.get(
        "CONTROL_TENSOR_NAME_PATTERNS",
        "attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights,skip_gates",
    ).split(",") if p
)

@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16(); X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed: X = X.T
    for _ in range(steps):
        A = X @ X.T; B = b*A + c*(A@A); X = a*X + B@X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True, weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay, row_normalize=row_normalize))
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad(): loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params: continue
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad; state = self.state[p]
                    if "momentum_buffer" not in state: state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]; buf.mul_(group["momentum"]).add_(g)
                    if group["nesterov"]: g = g.add(buf, alpha=group["momentum"])
                    if group.get("row_normalize", False):
                        g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr:curr+p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed: dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            curr = 0
            for p in params:
                if group.get("weight_decay", 0.0) > 0: p.data.mul_(1.0 - group["lr"] * group["weight_decay"])
                p.add_(updates_flat[curr:curr+p.numel()].view_as(p).to(p.dtype), alpha=-group["lr"])
                curr += p.numel()
        return loss


class Optimizers():
    def __init__(self, h, base_model):
        # Collect parameters from all three block groups (stem, shared, tail)
        all_block_named_params: list[tuple[str, nn.Parameter]] = []
        for pfx, mod in [("stem_blocks", base_model.stem_blocks),
                         ("shared_block", base_model.shared_block),
                         ("tail_blocks",  base_model.tail_blocks)]:
            all_block_named_params += [(f"{pfx}.{n}", p) for n, p in mod.named_parameters()]

        # depth_emb is [max_depth, dim] (ndim==2) but is NOT a weight matrix —
        # it's a lookup table. Route it to Adam (scalar_params), not Muon.
        matrix_params = [p for name, p in all_block_named_params
                         if p.ndim == 2
                         and "depth_emb" not in name
                         and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        scalar_params  = [p for name, p in all_block_named_params
                          if p.ndim < 2
                          or "depth_emb" in name
                          or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)]
        if base_model.skip_weights.numel() > 0: scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None:   scalar_params.append(base_model.skip_gates)

        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        self.optimizer_tok = torch.optim.AdamW(
            [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True)
        self.optimizer_muon = Muon(matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd, row_normalize=h.muon_row_normalize)
        for group in self.optimizer_muon.param_groups: group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True)
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizers.insert(1, torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True))

    def __iter__(self): return iter(self.optimizers)
    def zero_grad_all(self):
        for opt in self.optimizers: opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.optimizers: opt.step()
        self.zero_grad_all()

# ----------------------------------------
# Quantization  (unchanged)
# ----------------------------------------

def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear): module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()

def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}; hooks = []
    def make_hook(name):
        def fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return fn
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            if classify_param(name + ".weight") in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hook_mod = model.head_proj if model.head_proj is not None else model.final_norm
        def out_hook(name):
            def fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return fn
        hooks.append(hook_mod.register_forward_hook(out_hook("tok_emb.weight")))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks: hook.remove()
    for name in hessians: hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone(); rows, cols = W_orig.shape; H = H.float().clone()
    dead = torch.diag(H) == 0; H[dead, dead] = 1; H.diagonal().add_(0.01 * H.diag().mean())
    perm = torch.argsort(H.diag(), descending=True); invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone(); W_perm[:, dead[perm]] = 0; H = H[perm][:, perm]
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    sf = (clip_sigmas * W_orig.std(dim=1) / clip_range).clamp_min(1e-10).float()
    s = sf.to(torch.float16); Q = torch.zeros(rows, cols, dtype=torch.int8); W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols); W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]; Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            d = Hinv_block[j, j]; q_col = torch.clamp(torch.round(W_block[:, j] / sf), -clip_range, clip_range)
            Q[:, i1+j] = q_col.to(torch.int8); err = (W_block[:, j] - q_col * sf) / d; Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols: W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s

def gptq_mixed_quantize(state_dict, hessians, h):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"; continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=2**(bits-1)-1)
        result[name + ".q"] = q; result[name + ".scale"] = s; meta[name] = f"gptq (int{bits})"
    log("Quantized weights:"); cats = collections.defaultdict(set)
    for name, cat in meta.items():
        cats[cat].add(re.sub(r'\.\d+$', '', re.sub(r'(stem_blocks|shared_block|tail_blocks)\.\d*', r'\1', name)))
    for cat in sorted(cats): log(f"  {cat}: {', '.join(sorted(cats[cat]))}")
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        if "passthrough" in info:
            t = result[name]
            out[name] = t.to(orig.dtype) if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16) else t
            continue
        q, s = result[name+".q"], result[name+".scale"]
        out[name] = (q.float() * s.float().view(q.shape[0],*([1]*(q.ndim-1)))).to(orig.dtype) if s.ndim > 0 else (q.float()*float(s.item())).to(orig.dtype)
    return out

_BSHF_MAGIC = b"BSHF"

def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8); n = len(src); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        chunk = src[pos::stride]; out[off:off+len(chunk)] = chunk; off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC: return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5); n = len(payload); out = np.empty(n, dtype=np.uint8); off = 0
    for pos in range(stride):
        cl = n//stride + (1 if pos < n%stride else 0); out[pos::stride][:cl] = payload[off:off+cl]; off += cl
    return out.tobytes()

def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "lzma": return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli; return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown: {compressor}")

def _decompress(data, compressor):
    if compressor == "lzma": raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli; raw = brotli.decompress(data)
    else: raise ValueError(f"Unknown: {compressor}")
    return _byte_unshuffle(raw)

def serialize(h, base_model, code):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        log(f"Serialized model: {os.path.getsize(h.model_path)} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    log("GPTQ:collecting Hessians..."); t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, torch.device("cuda", h.local_rank))
    hessians = collect_hessians(base_model, calib_loader, h, torch.device("cuda", h.local_rank), h.gptq_calibration_batches)
    log(f"GPTQ:done {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO(); torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = _compress(quant_buf.getvalue(), h.compressor)
    bytes_total = len(quant_blob) + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f: f.write(quant_blob)
        log(f"Total submission size: {bytes_total} bytes")
    return bytes_total, len(quant_blob)

def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16(); restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f: quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob, h.compressor)), map_location="cpu")
    eval_model.load_state_dict(dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu), strict=True)
    return eval_model

# ----------------------------------------
# Evaluation  (unchanged)
# ----------------------------------------

def _loss_bpb(ls, tc, bc):
    vl = (ls/tc).item()
    return vl, vl/math.log(2.0)*(tc.item()/bc.item())

def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_seqs = (h.val_batch_tokens//(h.world_size*h.grad_accum_steps))//seq_len
    total_seqs = (val_data.val_tokens.numel()-1)//seq_len
    seq_start = (total_seqs*h.rank)//h.world_size; seq_end = (total_seqs*(h.rank+1))//h.world_size
    ls=torch.zeros((),device=device,dtype=torch.float64); tc=torch.zeros((),device=device,dtype=torch.float64); bc=torch.zeros((),device=device,dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss+local_batch_seqs, seq_end)
            local = val_data.val_tokens[bss*seq_len:(bse*seq_len+1)].to(device=device,dtype=torch.int64,non_blocking=True)
            x=local[:-1].reshape(-1,seq_len); y=local[1:].reshape(-1,seq_len)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): batch_loss=model(x,y).detach()
            ls+=batch_loss.to(torch.float64)*float(y.numel()); tc+=float(y.numel())
            tgt=y.reshape(-1); prev=x.reshape(-1)
            tb=val_data.base_bytes_lut[tgt].to(dtype=torch.int16)
            tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(dtype=torch.int16)
            bc+=tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls,op=dist.ReduceOp.SUM); dist.all_reduce(tc,op=dist.ReduceOp.SUM); dist.all_reduce(bc,op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(ls, tc, bc)

def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len=h.eval_seq_len; context_size=seq_len-h.eval_stride; total_tokens=val_data.val_tokens.numel()-1
    window_starts=[ws for ws in range(0,total_tokens,h.eval_stride) if ws+context_size<total_tokens]
    total_windows=len(window_starts)
    my_windows=window_starts[(total_windows*h.rank)//h.world_size:(total_windows*(h.rank+1))//h.world_size]
    ls=torch.zeros((),device=device,dtype=torch.float64); tc=torch.zeros((),device=device,dtype=torch.float64); bc=torch.zeros((),device=device,dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0,len(my_windows),batch_seqs):
            batch_ws=my_windows[bi:bi+batch_seqs]; bsz=len(batch_ws)
            x_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device); y_batch=torch.zeros(bsz,seq_len,dtype=torch.int64,device=device); wlens=[]
            for i,ws in enumerate(batch_ws):
                we=min(ws+seq_len,total_tokens); wlen=we-ws; wlens.append(wlen)
                chunk=val_data.val_tokens[ws:we+1].to(dtype=torch.int64,device=device)
                x_batch[i,:wlen]=chunk[:-1]; y_batch[i,:wlen]=chunk[1:]
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): logits=logits_fn(x_batch)
            nll=F.cross_entropy(logits.reshape(-1,logits.size(-1)).float(),y_batch.reshape(-1),reduction="none").reshape(bsz,seq_len)
            for i,ws in enumerate(batch_ws):
                s=0 if ws==0 else context_size; wlen=wlens[i]
                ls+=nll[i,s:wlen].to(torch.float64).sum(); tc+=float(wlen-s)
                tgt=y_batch[i,s:wlen]; prev=x_batch[i,s:wlen]
                tb=val_data.base_bytes_lut[tgt].to(torch.float64)
                tb+=(val_data.has_leading_space_lut[tgt]&~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                bc+=tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(ls,op=dist.ReduceOp.SUM); dist.all_reduce(tc,op=dist.ReduceOp.SUM); dist.all_reduce(bc,op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(ls, tc, bc)

def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize(); t0=time.perf_counter()
    val_loss,val_bpb=fn(*args,**kwargs); torch.cuda.synchronize()
    log(f"{label} val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f} eval_time:{1000*(time.perf_counter()-t0):.0f}ms")
    return val_loss, val_bpb

# ----------------------------------------
# Training
# ----------------------------------------

def train_model(h: Hyperparameters, device: torch.device, val_data: ValidationData):
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)
    model = DDP(compiled_model, device_ids=[h.local_rank], broadcast_buffers=False) if h.distributed else compiled_model

    n_unique = sum(p.numel() for p in base_model.parameters())
    n_eff = h.num_stem_layers + h.num_shared_loops + h.num_tail_layers
    log(f"model_params(unique):{n_unique} effective_depth:{n_eff}")
    log(f"weight_tying: stem={h.num_stem_layers} shared_loops={h.num_shared_loops} tail={h.num_tail_layers}")
    log(f"depth_rope: dims={h.depth_rope_dims} base={h.depth_rope_base} depth_range=0..{h.num_shared_loops-1}")

    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)

    max_wallclock_ms = (1000.0*h.max_wallclock_seconds - h.gptq_reserve_seconds*1000.0) if h.max_wallclock_seconds > 0 else None

    def training_frac(step, elapsed_ms):
        return step/max(h.iterations,1) if max_wallclock_ms is None else elapsed_ms/max(max_wallclock_ms,1e-9)

    def lr_mul(frac):
        if h.warmdown_frac<=0: return 1.0
        if frac>=1.0-h.warmdown_frac: return max((1.0-frac)/h.warmdown_frac,h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all(); train_loss=torch.zeros((),device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed: model.require_backward_grad_sync = micro_step==h.grad_accum_steps-1
            x,y=train_loader.next_batch(h.train_batch_tokens,h.grad_accum_steps)
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16): loss=model(x,y)
            train_loss+=loss.detach(); (loss/h.grad_accum_steps).backward()
        train_loss/=h.grad_accum_steps
        frac=min(step/h.muon_momentum_warmup_steps,1.0) if h.muon_momentum_warmup_steps>0 else 1.0
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"]=(1-frac)*h.muon_momentum_warmup_start+frac*h.muon_momentum
        for opt in optimizers:
            for group in opt.param_groups: group["lr"]=group["base_lr"]*lr_scale
        if h.grad_clip_norm>0: torch.nn.utils.clip_grad_norm_(base_model.parameters(),h.grad_clip_norm)
        optimizers.step()
        return train_loss

    if h.warmup_steps>0:
        init_model={n:t.detach().cpu().clone() for n,t in base_model.state_dict().items()}
        init_opt=[copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(h.warmup_steps):
            step_fn(ws,1.0)
            if ws<=5 or (ws+1)%10==0 or ws+1==h.warmup_steps: log(f"warmup_step: {ws+1}/{h.warmup_steps}")
        base_model.load_state_dict(init_model,strict=True)
        for opt,st in zip(optimizers,init_opt): opt.load_state_dict(st)
        optimizers.zero_grad_all()
        if h.distributed: model.require_backward_grad_sync=True
        train_loader=ShuffledSequenceLoader(h,device)

    ema_state={n:t.detach().float().clone() for n,t in base_model.state_dict().items()}
    training_time_ms=0.0; stop_after_step=None
    torch.cuda.synchronize(); t0=time.perf_counter(); step=0

    while True:
        last_step=step==h.iterations or (stop_after_step is not None and step>=stop_after_step)
        if last_step or (h.val_loss_every>0 and step%h.val_loss_every==0):
            torch.cuda.synchronize(); training_time_ms+=1000.0*(time.perf_counter()-t0)
            val_loss,val_bpb=eval_val(h,device,val_data,model)
            log(f"{step}/{h.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
            torch.cuda.synchronize(); t0=time.perf_counter()
        if last_step: break

        elapsed_ms=training_time_ms+1000.0*(time.perf_counter()-t0)
        train_loss=step_fn(step, lr_mul(training_frac(step, elapsed_ms)))
        with torch.no_grad():
            for name,t in base_model.state_dict().items():
                ema_state[name].mul_(h.ema_decay).add_(t.detach().float(),alpha=1.0-h.ema_decay)

        step+=1
        approx_ms=training_time_ms+1000.0*(time.perf_counter()-t0)
        if h.train_log_every>0 and (step<=5 or step%h.train_log_every==0):
            log(f"{step}/{h.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms/60000:.1f}m tok/s:{step*h.train_batch_tokens/(approx_ms/1000):.0f}")

        reached_cap=max_wallclock_ms is not None and approx_ms>=max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            t_cap=torch.tensor(int(reached_cap),device=device)
            dist.all_reduce(t_cap,op=dist.ReduceOp.MAX); reached_cap=bool(t_cap.item())
        if stop_after_step is None and reached_cap: stop_after_step=step

    log(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    log("ema:applying EMA weights")
    base_model.load_state_dict({n:t.to(base_model.state_dict()[n].dtype) for n,t in ema_state.items()},strict=True)
    return base_model, compiled_model


def train_and_eval(h, device):
    random.seed(h.seed); np.random.seed(h.seed)
    torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)
    val_data=ValidationData(h,device)
    log(f"train_shards:{len(list(Path(h.datasets_dir).glob('fineweb_train_*.bin')))} val_tokens:{val_data.val_tokens.numel()-1}")
    base_model,compiled_model=train_model(h,device,val_data)
    torch._dynamo.reset()
    timed_eval("pre-quant",eval_val,h,device,val_data,compiled_model)
    serialize(h,base_model,Path(__file__).read_text(encoding="utf-8"))
    if h.distributed: dist.barrier()
    eval_model=deserialize(h,device)
    compiled_eval=torch.compile(eval_model,dynamic=False,fullgraph=True)
    timed_eval("quantized",eval_val,h,device,val_data,compiled_eval)
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding",eval_val_sliding,h,device,val_data,eval_model)


def main():
    world_size=int(os.environ.get("WORLD_SIZE","1")); local_rank=int(os.environ.get("LOCAL_RANK","0"))
    distributed="RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if 8%world_size!=0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    device=torch.device("cuda",local_rank); torch.cuda.set_device(device)
    if distributed: dist.init_process_group(backend="nccl",device_id=device); dist.barrier()
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp,enable_flash_sdp,enable_math_sdp,enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp=False
    h=Hyperparameters(); set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs",exist_ok=True)
        log("Hyperparameters:")
        for k,v in sorted(vars(type(h)).items()):
            if not k.startswith("_"): log(f"  {k}: {v}")
    train_and_eval(h,device)
    if distributed: dist.destroy_process_group()

if __name__=="__main__":
    main()
