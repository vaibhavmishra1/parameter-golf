"""Weight-Tied Recursive Transformer for parameter-golf.

KEY IDEA: Instead of N unique transformer blocks, use a small set of unique
blocks where one central "shared" block is applied repeatedly (weight tying).

Default config (all tunable via env vars):
  stem  : NUM_STEM_LAYERS (2) unique blocks — front of network
  shared: 1 block applied NUM_SHARED_LOOPS (7) times  — recursive core
  tail  : NUM_TAIL_LAYERS (2) unique blocks — back of network
  -----------------------------------------------
  Effective depth  : 2 + 7 + 2 = 11  (same as baseline)
  Unique block count: 2 + 1 + 2 = 5   (baseline has 11)

Parameter savings: ~55% reduction in block-parameter bytes.
Freed budget can be reinvested into:
  - Wider model (MODEL_DIM 512 → 576/640)
  - Larger MLP multiplier (MLP_MULT 4 → 5/6)
  - More shared loops for deeper effective compute

Why it works: The shared block receives a DIFFERENT residual stream state at
every recursion pass, so it effectively computes different transformations even
though the weights are identical. The U-Net skip connections additionally
signal "encoder vs decoder" phase to the shared block.

Based on: 2026-04-07_TMA_MegaKernel_TripleLoop_ParallelResiduals
Removes: num_loops / loop_start / loop_end / enable_looping_at (not needed)
Adds   : num_stem_layers / num_shared_loops / num_tail_layers
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

    # === WEIGHT-TYING ARCHITECTURE PARAMS ===
    # Increase MODEL_DIM (e.g. 576 or 640) to reinvest the saved parameter budget.
    num_stem_layers  = int(os.environ.get('NUM_STEM_LAYERS',  2))
    num_shared_loops = int(os.environ.get('NUM_SHARED_LOOPS', 7))
    num_tail_layers  = int(os.environ.get('NUM_TAIL_LAYERS',  2))
    model_dim        = int(os.environ.get('MODEL_DIM', 512))   # try 576 or 640

    embedding_dim    = int(os.environ.get('EMBEDDING_DIM', 512))
    num_kv_heads     = int(os.environ.get('NUM_KV_HEADS', 4))
    num_heads        = int(os.environ.get('NUM_HEADS', 8))
    mlp_mult         = float(os.environ.get('MLP_MULT', 4.0))
    skip_gates_enabled = bool(int(os.environ.get('SKIP_GATES_ENABLED', '1')))
    tie_embeddings   = bool(int(os.environ.get('TIE_EMBEDDINGS', '1')))
    logit_softcap    = float(os.environ.get('LOGIT_SOFTCAP', 30.0))
    rope_base        = float(os.environ.get('ROPE_BASE', 10000.0))
    rope_dims        = int(os.environ.get('ROPE_DIMS', 16))
    rope_train_seq_len = int(os.environ.get('ROPE_TRAIN_SEQ_LEN', 2048))
    ln_scale         = bool(int(os.environ.get('LN_SCALE', '1')))
    qk_gain_init     = float(os.environ.get('QK_GAIN_INIT', 4.0))
    xsa_last_n       = int(os.environ.get('XSA_LAST_N', 2))  # XSA on last N tail blocks

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

def set_logging_hparams(h: Hyperparameters) -> None:
    global _logger_hparams
    _logger_hparams = h

def log(msg, console: bool = True) -> None:
    if _logger_hparams is None:
        print(msg)
        return
    if _logger_hparams.is_main_process:
        if console:
            print(msg)
        if _logger_hparams.logfile is not None:
            with open(_logger_hparams.logfile, "a", encoding="utf-8") as f:
                print(msg, file=f)

# ----------------------------------------
# Data Loading  (unchanged from SOTA)
# ----------------------------------------

class ValidationData:
    def __init__(self, h: Hyperparameters, device: torch.device):
        self.sp = spm.SentencePieceProcessor(model_file=h.tokenizer_path)
        if int(self.sp.vocab_size()) != h.vocab_size:
            raise ValueError(f"VOCAB_SIZE={h.vocab_size} != tokenizer {int(self.sp.vocab_size())}")
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
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("\u2581"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )

def load_validation_tokens(pattern, seq_len):
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found: {pattern}")
    tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    return tokens[:usable + 1]

def load_data_shard(file):
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Bad shard header: {file}")
    num_tokens = int(header[2])
    if file.stat().st_size != header_bytes + num_tokens * token_bytes:
        raise ValueError(f"Shard size mismatch: {file}")
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))

_SHARD_HEADER_BYTES = 256 * np.dtype("<i4").itemsize
_SHARD_NTOKENS_CACHE: dict[str, int] = {}
_MMAP_CACHE: dict[str, np.memmap] = {}

def _read_num_tokens(file):
    key = str(file)
    if key in _SHARD_NTOKENS_CACHE:
        return _SHARD_NTOKENS_CACHE[key]
    header = np.fromfile(file, dtype="<i4", count=256)
    n = int(header[2])
    _SHARD_NTOKENS_CACHE[key] = n
    return n

def _get_shard_memmap(file):
    key = str(file)
    if key in _MMAP_CACHE:
        return _MMAP_CACHE[key]
    n = _read_num_tokens(file)
    mm = np.memmap(file, mode="r", dtype="<u2", offset=_SHARD_HEADER_BYTES, shape=(n,))
    _MMAP_CACHE[key] = mm
    return mm

class ShuffledSequenceLoader:
    def __init__(self, h, device):
        self.world_size = h.world_size
        self.seq_len = h.train_seq_len
        self.device = device
        all_files = [Path(p) for p in sorted(glob.glob(h.train_files))]
        if not all_files:
            raise FileNotFoundError(f"No files: {h.train_files}")
        self.files = all_files[h.rank::h.world_size]
        self.rng = np.random.Generator(np.random.PCG64(h.rank))
        self.num_tokens = [_read_num_tokens(f) for f in self.files]
        self.start_inds: list[list[int]] = [[] for _ in self.files]
        for si in range(len(self.files)):
            self._reset_shard(si)

    def _reset_shard(self, si):
        max_phase = min(self.seq_len - 1, max(0, self.num_tokens[si] - self.seq_len - 1))
        phase = int(self.rng.integers(max_phase + 1)) if max_phase > 0 else 0
        num_sequences = (self.num_tokens[si] - 1 - phase) // self.seq_len
        sequence_order = self.rng.permutation(num_sequences)
        self.start_inds[si] = (phase + sequence_order * self.seq_len).tolist()

    def next_batch(self, global_tokens, grad_accum_steps):
        device_tokens = global_tokens // (self.world_size * grad_accum_steps)
        device_batch_size = device_tokens // self.seq_len
        remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
        x = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        y = torch.empty((device_batch_size, self.seq_len), dtype=torch.int64)
        for bi in range(device_batch_size):
            total = remaining.sum()
            if total <= 0:
                for si in range(len(self.files)):
                    self._reset_shard(si)
                remaining = np.array([len(s) for s in self.start_inds], dtype=np.float64)
                total = remaining.sum()
            probs = remaining / total
            si = int(self.rng.choice(len(self.files), p=probs))
            start_ind = self.start_inds[si].pop()
            remaining[si] -= 1
            mm = _get_shard_memmap(self.files[si])
            window = torch.as_tensor(np.array(mm[start_ind:start_ind + self.seq_len + 1], dtype=np.int64))
            x[bi] = window[:-1]
            y[bi] = window[1:]
        return x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)

# ----------------------------------------
# Fused MLP Kernels (unchanged from SOTA)
# ----------------------------------------

if HAS_TRITON:
    from triton.tools.tensor_descriptor import TensorDescriptor

    @triton.jit
    def _fused_leaky_relu_sq_tma_kernel(
        a_desc, b_desc, c_desc, aux_desc,
        M, N, K,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr, NUM_SMS: tl.constexpr,
    ):
        dtype = tl.bfloat16
        start_pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        k_tiles = tl.cdiv(K, BLOCK_K)
        num_tiles = num_pid_m * num_pid_n
        for tile_id in tl.range(start_pid, num_tiles, NUM_SMS, flatten=True):
            pid_m = tile_id // num_pid_n
            pid_n = tile_id % num_pid_n
            offs_am = pid_m * BLOCK_M
            offs_bn = pid_n * BLOCK_N
            accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for ki in range(k_tiles):
                offs_k = ki * BLOCK_K
                a = a_desc.load([offs_am, offs_k])
                b = b_desc.load([offs_bn, offs_k])
                accumulator = tl.dot(a, b.T, accumulator)
            acc = tl.reshape(accumulator, (BLOCK_M, 2, BLOCK_N // 2))
            acc = tl.permute(acc, (0, 2, 1))
            acc0, acc1 = tl.split(acc)
            c0 = acc0.to(dtype)
            c0_ag = tl.where(c0 > 0, 2.0 * c0, 0.5 * c0)
            c_desc.store([offs_am, offs_bn], c0_ag)
            c0_post = 0.5 * c0_ag * c0
            aux_desc.store([offs_am, offs_bn], c0_post)
            c1 = acc1.to(dtype)
            c1_ag = tl.where(c1 > 0, 2.0 * c1, 0.5 * c1)
            c_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_ag)
            c1_post = 0.5 * c1_ag * c1
            aux_desc.store([offs_am, offs_bn + BLOCK_N // 2], c1_post)

    def _triton_fused_leaky_relu_sq(x_flat, fc_weight):
        M, K = x_flat.shape
        N, K2 = fc_weight.shape
        assert K == K2
        act_grad = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        post = torch.empty((M, N), device=x_flat.device, dtype=x_flat.dtype)
        NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 256, 64
        a_desc = TensorDescriptor.from_tensor(x_flat, [BLOCK_M, BLOCK_K])
        b_desc = TensorDescriptor.from_tensor(fc_weight, [BLOCK_N, BLOCK_K])
        c_desc = TensorDescriptor.from_tensor(act_grad, [BLOCK_M, BLOCK_N // 2])
        aux_desc = TensorDescriptor.from_tensor(post, [BLOCK_M, BLOCK_N // 2])
        def grid(META):
            return (min(NUM_SMS, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)
        _fused_leaky_relu_sq_tma_kernel[grid](
            a_desc, b_desc, c_desc, aux_desc, M, N, K,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            GROUP_SIZE_M=1, NUM_SMS=NUM_SMS, num_stages=4, num_warps=8,
        )
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
            dW_proj = go.T @ post
            dpre = (go @ proj_w) * act_grad
            dW_fc = dpre.T @ x_flat
            dx = dpre @ fc_w
            return dx.reshape(ctx.orig_shape), dW_fc, dW_proj

# ----------------------------------------
# Model Architecture  (KEY CHANGES HERE)
# ----------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, eps=None):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), eps=self.eps)

class CastedLinear(nn.Linear):
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class Rotary(nn.Module):
    def __init__(self, dim, base=10000.0, train_seq_len=1024, rope_dims=0):
        super().__init__()
        self.dim = dim
        self.base = base
        self.train_seq_len = train_seq_len
        self.rope_dims = rope_dims if rope_dims > 0 else dim
        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if self._cos_cached is None or self._seq_len_cached != seq_len or self._cos_cached.device != device:
            rd = self.rope_dims
            if seq_len > self.train_seq_len:
                scale = seq_len / self.train_seq_len
                new_base = self.base * (scale ** (rd / (rd - 2)))
                inv_freq = 1.0 / (new_base ** (torch.arange(0, rd, 2, dtype=torch.float32, device=device) / rd))
            else:
                inv_freq = self.inv_freq.to(device)
            t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = freqs.cos()[None, :, None, :]
            self._sin_cached = freqs.sin()[None, :, None, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)

def apply_rotary_emb(x, cos, sin, rope_dims=0):
    if rope_dims > 0 and rope_dims < x.size(-1):
        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]
        half = rope_dims // 2
        x1, x2 = x_rope[..., :half], x_rope[..., half:]
        x_rope = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)
        return torch.cat((x_rope, x_pass), dim=-1)
    half = x.size(-1) // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        kv_dim = num_kv_heads * self.head_dim
        self.c_q   = CastedLinear(dim, dim, bias=False)
        self.c_k   = CastedLinear(dim, kv_dim, bias=False)
        self.c_v   = CastedLinear(dim, kv_dim, bias=False)
        self.proj  = CastedLinear(dim, dim, bias=False)
        self.proj._zero_init = True
        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))
        self.rope_dims = 0
        self.rotary = Rotary(self.head_dim, base=rope_base, train_seq_len=train_seq_len)
        self.use_xsa = False

    def _xsa_efficient(self, y, v):
        B, T, H, D = y.shape
        Hkv = v.size(-2)
        group = H // Hkv
        y_g = y.reshape(B, T, Hkv, group, D)
        vn = F.normalize(v, dim=-1).unsqueeze(-2)
        proj = (y_g * vn).sum(dim=-1, keepdim=True) * vn
        return (y_g - proj).reshape(B, T, H, D)

    def forward(self, x):
        bsz, seqlen, dim = x.shape
        q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(seqlen, x.device, q.dtype)
        q = apply_rotary_emb(q, cos, sin, self.rope_dims)
        k = apply_rotary_emb(k, cos, sin, self.rope_dims)
        q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
        y = flash_attn_3_func(q, k, v, causal=True)
        if self.use_xsa:
            y = self._xsa_efficient(y, v)
        y = y.reshape(bsz, seqlen, dim)
        return self.proj(y)

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
                 train_seq_len, layer_idx=0, ln_scale=False, parallel=False):
        super().__init__()
        self.parallel = parallel
        self.attn_norm = RMSNorm()
        self.mlp_norm  = RMSNorm()
        self.attn = CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init, train_seq_len)
        self.mlp  = MLP(dim, mlp_mult)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale  = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix  = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.ln_scale_factor = 1.0 / math.sqrt(layer_idx + 1) if ln_scale else 1.0

    def forward(self, x, x0):
        mix = self.resid_mix.to(dtype=x.dtype)
        x_in = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        normed = self.attn_norm(x_in) * self.ln_scale_factor
        attn_out = self.attn(normed)
        if self.parallel:
            mlp_out = self.mlp(self.mlp_norm(x_in) * self.ln_scale_factor)
            return x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out \
                        + self.mlp_scale.to(x_in.dtype)[None, None, :] * mlp_out
        x_out = x_in + self.attn_scale.to(x_in.dtype)[None, None, :] * attn_out
        return x_out + self.mlp_scale.to(x_out.dtype)[None, None, :] * self.mlp(
            self.mlp_norm(x_out) * self.ln_scale_factor)


# === NEW: Weight-Tied GPT ===
class GPT(nn.Module):
    """
    Weight-tied transformer: stem_blocks → shared_block (×N) → tail_blocks.
    The shared_block is a single nn.Module applied num_shared_loops times.
    The U-Net skip connections span the encoder/decoder split to provide
    "depth phase" information to the shared block.
    """
    def __init__(self, h: Hyperparameters):
        super().__init__()
        assert h.logit_softcap > 0.0
        self.tie_embeddings = h.tie_embeddings
        self.logit_softcap  = h.logit_softcap

        self.tok_emb = nn.Embedding(h.vocab_size, h.embedding_dim)
        if h.embedding_dim != h.model_dim:
            self.embed_proj = CastedLinear(h.embedding_dim, h.model_dim, bias=False)
            self.head_proj  = CastedLinear(h.model_dim, h.embedding_dim, bias=False)
        else:
            self.embed_proj = None
            self.head_proj  = None

        def make_block(layer_idx, parallel=False):
            b = Block(h.model_dim, h.num_heads, h.num_kv_heads, h.mlp_mult,
                      h.rope_base, h.qk_gain_init, h.train_seq_len,
                      layer_idx=layer_idx, ln_scale=h.ln_scale, parallel=parallel)
            if h.rope_dims > 0:
                b.attn.rope_dims = h.rope_dims
                b.attn.rotary = Rotary(b.attn.head_dim, base=h.rope_base,
                                       train_seq_len=h.train_seq_len, rope_dims=h.rope_dims)
            return b

        # Stem blocks — unique, position-sensitive front layers
        self.stem_blocks = nn.ModuleList([make_block(i) for i in range(h.num_stem_layers)])

        # Shared block — applied num_shared_loops times (weight tying)
        # Using layer_idx = midpoint so ln_scale is reasonable for middle of network
        mid_idx = h.num_stem_layers + h.num_shared_loops // 2
        self.shared_block = make_block(mid_idx)

        # Tail blocks — unique, position-sensitive end layers (parallel residual + XSA)
        total_tail = h.num_tail_layers
        tail_parallel_start = max(0, total_tail - 2)
        self.tail_blocks = nn.ModuleList([
            make_block(h.num_stem_layers + h.num_shared_loops + i,
                       parallel=(i >= tail_parallel_start))
            for i in range(total_tail)
        ])

        # Apply XSA to last xsa_last_n tail blocks
        for i in range(max(0, total_tail - h.xsa_last_n), total_tail):
            self.tail_blocks[i].attn.use_xsa = True

        self.final_norm = RMSNorm()
        self.lm_head = None if h.tie_embeddings else CastedLinear(h.embedding_dim, h.vocab_size, bias=False)

        # Compute encoder / decoder split for skip connections
        # Enc: stem + first half of shared loops
        # Dec: second half of shared loops + tail
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
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=h.tied_embed_init_std)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, '_zero_init', False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)

    def _inject_skip(self, x, skip_idx, skips):
        """Apply skip connection with optional learned gate."""
        scaled = self.skip_weights[skip_idx].to(x.dtype)[None, None, :] * skips.pop()
        if self.skip_gates is not None:
            g = torch.sigmoid(self.skip_gates[skip_idx].to(x.dtype))[None, None, :]
            return torch.lerp(scaled, x, g)
        return x + scaled

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        if self.embed_proj is not None:
            x = self.embed_proj(x)
        x0 = x
        skips: list[Tensor] = []

        # === ENCODER PHASE ===
        # Stem blocks (unique): always encoder
        for block in self.stem_blocks:
            x = block(x, x0)
            skips.append(x)

        # Shared block — first half of loops: encoder
        for _ in range(self.num_shared_enc):
            x = self.shared_block(x, x0)
            skips.append(x)

        # === DECODER PHASE ===
        skip_idx = 0

        # Shared block — second half of loops: decoder (pops skips)
        for _ in range(self.num_shared_dec):
            if skip_idx < self.num_skip_weights:
                x = self._inject_skip(x, skip_idx, skips)
                skip_idx += 1
            x = self.shared_block(x, x0)

        # Tail blocks (unique): decoder
        for block in self.tail_blocks:
            if skip_idx < self.num_skip_weights:
                x = self._inject_skip(x, skip_idx, skips)
                skip_idx += 1
            x = block(x, x0)

        x = self.final_norm(x)
        if self.head_proj is not None:
            x = self.head_proj(x)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)

    def forward(self, input_ids, target_ids):
        logits = self.forward_logits(input_ids)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                               target_ids.reshape(-1), reduction="mean")


def classify_param(name: str) -> str:
    if "tok_emb" in name or "lm_head" in name:
        return "embed"
    if ".mlp." in name:
        return "mlp"
    if ".attn." in name:
        return "attn"
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
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, backend_steps, nesterov=True,
                 weight_decay=0.0, row_normalize=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, backend_steps=backend_steps,
                                      nesterov=nesterov, weight_decay=weight_decay,
                                      row_normalize=row_normalize))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        distributed = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if distributed else 1
        rank = dist.get_rank() if distributed else 0
        for group in self.param_groups:
            params = group["params"]
            if not params:
                continue
            lr, momentum = group["lr"], group["momentum"]
            backend_steps, nesterov = group["backend_steps"], group["nesterov"]
            total_params = sum(int(p.numel()) for p in params)
            updates_flat = torch.zeros(total_params, device=params[0].device, dtype=torch.bfloat16)
            curr = 0
            for i, p in enumerate(params):
                if i % world_size == rank and p.grad is not None:
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if nesterov:
                        g = g.add(buf, alpha=momentum)
                    if group.get("row_normalize", False):
                        row_norms = g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7)
                        g = g / row_norms.to(g.dtype)
                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr: curr + p.numel()] = g.reshape(-1)
                curr += p.numel()
            if distributed:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)
            wd = group.get("weight_decay", 0.0)
            curr = 0
            for p in params:
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                g = updates_flat[curr: curr + p.numel()].view_as(p).to(p.dtype)
                p.add_(g, alpha=-lr)
                curr += p.numel()
        return loss


class Optimizers():
    def __init__(self, h: Hyperparameters, base_model: GPT):
        # Collect all block-level parameters from the three module groups
        all_block_named_params: list[tuple[str, nn.Parameter]] = []
        for pfx, mod in [("stem_blocks", base_model.stem_blocks),
                         ("shared_block", base_model.shared_block),
                         ("tail_blocks",  base_model.tail_blocks)]:
            all_block_named_params += [(f"{pfx}.{n}", p) for n, p in mod.named_parameters()]

        matrix_params = [
            p for name, p in all_block_named_params
            if p.ndim == 2 and not any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        scalar_params = [
            p for name, p in all_block_named_params
            if p.ndim < 2 or any(pat in name for pat in CONTROL_TENSOR_NAME_PATTERNS)
        ]
        if base_model.skip_weights.numel() > 0:
            scalar_params.append(base_model.skip_weights)
        if base_model.skip_gates is not None:
            scalar_params.append(base_model.skip_gates)

        token_lr = h.tied_embed_lr if h.tie_embeddings else h.embed_lr
        self.optimizer_tok = torch.optim.AdamW(
            [{"params": [base_model.tok_emb.weight], "lr": token_lr, "base_lr": token_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.embed_wd, fused=True,
        )
        self.optimizer_muon = Muon(
            matrix_params, lr=h.matrix_lr, momentum=h.muon_momentum,
            backend_steps=h.muon_backend_steps, weight_decay=h.muon_wd,
            row_normalize=h.muon_row_normalize,
        )
        for group in self.optimizer_muon.param_groups:
            group["base_lr"] = h.matrix_lr
        self.optimizer_scalar = torch.optim.AdamW(
            [{"params": scalar_params, "lr": h.scalar_lr, "base_lr": h.scalar_lr}],
            betas=(h.beta1, h.beta2), eps=h.adam_eps, weight_decay=h.adam_wd, fused=True,
        )
        self.optimizers = [self.optimizer_tok, self.optimizer_muon, self.optimizer_scalar]
        if base_model.lm_head is not None:
            self.optimizer_head = torch.optim.Adam(
                [{"params": [base_model.lm_head.weight], "lr": h.head_lr, "base_lr": h.head_lr}],
                betas=(h.beta1, h.beta2), eps=h.adam_eps, fused=True,
            )
            self.optimizers.insert(1, self.optimizer_head)

    def __iter__(self): return iter(self.optimizers)
    def zero_grad_all(self):
        for opt in self.optimizers: opt.zero_grad(set_to_none=True)
    def step(self):
        for opt in self.optimizers: opt.step()
        self.zero_grad_all()

# ----------------------------------------
# Quantization  (unchanged from SOTA)
# ----------------------------------------

def restore_fp32_params(model):
    for module in model.modules():
        if isinstance(module, CastedLinear):
            module.float()
    for name, param in model.named_parameters():
        if (param.ndim < 2 or any(p in name for p in CONTROL_TENSOR_NAME_PATTERNS)) and param.dtype != torch.float32:
            param.data = param.data.float()

def collect_hessians(model, train_loader, h, device, n_calibration_batches=64):
    hessians = {}
    hooks = []
    def make_hook(name):
        def hook_fn(module, inp, out):
            x = inp[0].detach().float()
            if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
            if name not in hessians:
                hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
            hessians[name].addmm_(x.T, x)
        return hook_fn
    for name, module in model.named_modules():
        if isinstance(module, CastedLinear) and module.weight.numel() > 65536:
            cat = classify_param(name + ".weight")
            if cat in ("mlp", "attn"):
                hooks.append(module.register_forward_hook(make_hook(name + ".weight")))
    if model.tie_embeddings:
        hook_mod = model.head_proj if model.head_proj is not None else model.final_norm
        def make_output_hook(name):
            def hook_fn(module, inp, out):
                x = out.detach().float()
                if x.ndim == 3: x = x.reshape(-1, x.shape[-1])
                if name not in hessians:
                    hessians[name] = torch.zeros(x.shape[1], x.shape[1], dtype=torch.float32, device=device)
                hessians[name].addmm_(x.T, x)
            return hook_fn
        hooks.append(hook_mod.register_forward_hook(make_output_hook("tok_emb.weight")))
    model.eval()
    with torch.no_grad():
        for _ in range(n_calibration_batches):
            x, _ = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            model.forward_logits(x)
    for hook in hooks: hook.remove()
    for name in hessians: hessians[name] = hessians[name].cpu() / n_calibration_batches
    return hessians

def gptq_quantize_weight(w, H, clip_sigmas=3.0, clip_range=63, block_size=128):
    W_orig = w.float().clone()
    rows, cols = W_orig.shape
    H = H.float().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H.diagonal().add_(0.01 * H.diag().mean())
    perm = torch.argsort(H.diag(), descending=True)
    invperm = torch.argsort(perm)
    W_perm = W_orig[:, perm].clone()
    W_perm[:, dead[perm]] = 0
    H = H[perm][:, perm]
    Hinv = torch.cholesky_inverse(torch.linalg.cholesky(H))
    Hinv = torch.linalg.cholesky(Hinv, upper=True)
    row_std = W_orig.std(dim=1)
    s = (clip_sigmas * row_std / clip_range).clamp_min(1e-10).to(torch.float16)
    sf = s.float()
    Q = torch.zeros(rows, cols, dtype=torch.int8)
    W_work = W_perm.clone()
    for i1 in range(0, cols, block_size):
        i2 = min(i1 + block_size, cols)
        W_block = W_work[:, i1:i2].clone()
        Hinv_block = Hinv[i1:i2, i1:i2]
        Err = torch.zeros(rows, i2 - i1)
        for j in range(i2 - i1):
            w_col = W_block[:, j]
            d = Hinv_block[j, j]
            q_col = torch.clamp(torch.round(w_col / sf), -clip_range, clip_range)
            Q[:, i1 + j] = q_col.to(torch.int8)
            err = (w_col - q_col.float() * sf) / d
            Err[:, j] = err
            W_block[:, j:] -= err.unsqueeze(1) * Hinv_block[j, j:].unsqueeze(0)
        if i2 < cols:
            W_work[:, i2:] -= Err @ Hinv[i1:i2, i2:]
    return Q[:, invperm], s

def gptq_mixed_quantize(state_dict, hessians, h):
    result, meta = {}, {}
    for name, tensor in state_dict.items():
        t = tensor.detach().cpu().contiguous()
        if not t.is_floating_point() or t.numel() <= 65536:
            result[name] = t.to(torch.float16) if t.is_floating_point() else t
            meta[name] = "passthrough (float16)"
            continue
        cs = h.embed_clip_sigmas if "tok_emb" in name else h.matrix_clip_sigmas
        bits = h.embed_bits if "tok_emb" in name else h.matrix_bits
        q, s = gptq_quantize_weight(t, hessians[name], clip_sigmas=cs, clip_range=2**(bits-1)-1)
        result[name + ".q"] = q
        result[name + ".scale"] = s
        meta[name] = f"gptq (int{bits})"
    categories = collections.defaultdict(set)
    for name, cat in meta.items():
        short = re.sub(r'\.\d+$', '', re.sub(r'(stem_blocks|shared_block|tail_blocks)\.\d*', r'\1', name))
        categories[cat].add(short)
    log("Quantized weights:")
    for cat in sorted(categories):
        log(f"  {cat}: {', '.join(sorted(categories[cat]))}")
    return result, meta

def dequantize_mixed(result, meta, template_sd):
    out = {}
    for name, orig in template_sd.items():
        info = meta.get(name)
        if info is None: continue
        orig_dtype = orig.dtype
        if "passthrough" in info:
            t = result[name]
            if t.dtype == torch.float16 and orig_dtype in (torch.float32, torch.bfloat16):
                t = t.to(orig_dtype)
            out[name] = t
            continue
        q, s = result[name + ".q"], result[name + ".scale"]
        if s.ndim > 0:
            out[name] = (q.float() * s.float().view(q.shape[0], *([1]*(q.ndim-1)))).to(orig_dtype)
        else:
            out[name] = (q.float() * float(s.item())).to(orig_dtype)
    return out

_BSHF_MAGIC = b"BSHF"

def _byte_shuffle(data, stride=2):
    if stride <= 1 or len(data) < stride: return data
    src = np.frombuffer(data, dtype=np.uint8)
    n = len(src)
    out = np.empty(n, dtype=np.uint8)
    dest_off = 0
    for pos in range(stride):
        chunk = src[pos::stride]
        out[dest_off:dest_off + len(chunk)] = chunk
        dest_off += len(chunk)
    return _BSHF_MAGIC + bytes([stride]) + out.tobytes()

def _byte_unshuffle(data):
    if len(data) < 5 or data[:4] != _BSHF_MAGIC: return data
    stride = data[4]
    if stride < 2: return data[5:]
    payload = np.frombuffer(data, dtype=np.uint8, offset=5)
    n = len(payload)
    out = np.empty(n, dtype=np.uint8)
    src_off = 0
    for pos in range(stride):
        chunk_len = n // stride + (1 if pos < n % stride else 0)
        out[pos::stride][:chunk_len] = payload[src_off:src_off + chunk_len]
        src_off += chunk_len
    return out.tobytes()

def _compress(data, compressor):
    data = _byte_shuffle(data)
    if compressor == "lzma": return lzma.compress(data, preset=6)
    elif compressor == "brotli":
        import brotli
        return brotli.compress(data, quality=11)
    raise ValueError(f"Unknown compressor: {compressor!r}")

def _decompress(data, compressor):
    if compressor == "lzma": raw = lzma.decompress(data)
    elif compressor == "brotli":
        import brotli
        raw = brotli.decompress(data)
    else: raise ValueError(f"Unknown compressor: {compressor!r}")
    return _byte_unshuffle(raw)

def serialize(h, base_model, code):
    code_bytes = len(code.encode("utf-8"))
    if h.is_main_process:
        torch.save(base_model.state_dict(), h.model_path)
        log(f"Serialized model: {os.path.getsize(h.model_path)} bytes")
    sd_cpu = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    log("GPTQ:collecting Hessians...")
    t0 = time.perf_counter()
    calib_loader = ShuffledSequenceLoader(h, torch.device("cuda", h.local_rank))
    hessians = collect_hessians(base_model, calib_loader, h,
                                 torch.device("cuda", h.local_rank), h.gptq_calibration_batches)
    log(f"GPTQ:collected {len(hessians)} Hessians in {time.perf_counter()-t0:.1f}s")
    quant_result, quant_meta = gptq_mixed_quantize(sd_cpu, hessians, h)
    quant_buf = io.BytesIO()
    torch.save({"w": quant_result, "m": quant_meta}, quant_buf)
    quant_blob = _compress(quant_buf.getvalue(), h.compressor)
    bytes_total = len(quant_blob) + code_bytes
    if h.is_main_process:
        with open(h.quantized_model_path, "wb") as f: f.write(quant_blob)
        log(f"Serialized model quantized+{h.compressor}: {len(quant_blob)} bytes")
        log(f"Total submission size: {bytes_total} bytes")
    return bytes_total, len(quant_blob)

def deserialize(h, device):
    eval_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(eval_model)
    sd_cpu = {k: v.detach().cpu() for k, v in eval_model.state_dict().items()}
    with open(h.quantized_model_path, "rb") as f: quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(_decompress(quant_blob, h.compressor)), map_location="cpu")
    eval_model.load_state_dict(dequantize_mixed(quant_state["w"], quant_state["m"], sd_cpu), strict=True)
    return eval_model

# ----------------------------------------
# Evaluation  (unchanged from SOTA)
# ----------------------------------------

def _loss_bpb(loss_sum, token_count, byte_count):
    val_loss = (loss_sum / token_count).item()
    val_bpb  = val_loss / math.log(2.0) * (token_count.item() / byte_count.item())
    return val_loss, val_bpb

def eval_val(h, device, val_data, model):
    seq_len = h.eval_seq_len
    local_batch_tokens = h.val_batch_tokens // (h.world_size * h.grad_accum_steps)
    local_batch_seqs = local_batch_tokens // seq_len
    total_seqs = (val_data.val_tokens.numel() - 1) // seq_len
    seq_start = (total_seqs * h.rank) // h.world_size
    seq_end   = (total_seqs * (h.rank + 1)) // h.world_size
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    model.eval()
    with torch.inference_mode():
        for bss in range(seq_start, seq_end, local_batch_seqs):
            bse = min(bss + local_batch_seqs, seq_end)
            local = val_data.val_tokens[bss*seq_len:(bse*seq_len+1)].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                batch_loss = model(x, y).detach()
            loss_sum += batch_loss.to(torch.float64) * float(y.numel())
            token_count += float(y.numel())
            tgt = y.reshape(-1); prev = x.reshape(-1)
            tb = val_data.base_bytes_lut[tgt].to(dtype=torch.int16)
            tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(dtype=torch.int16)
            byte_count += tb.to(torch.float64).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum,    op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count,  op=dist.ReduceOp.SUM)
    model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)

def eval_val_sliding(h, device, val_data, base_model, batch_seqs=32):
    base_model.eval()
    logits_fn = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)
    seq_len = h.eval_seq_len
    context_size = seq_len - h.eval_stride
    total_tokens = val_data.val_tokens.numel() - 1
    window_starts = [ws for ws in range(0, total_tokens, h.eval_stride) if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = (total_windows * h.rank) // h.world_size
    my_e = (total_windows * (h.rank + 1)) // h.world_size
    my_windows = window_starts[my_s:my_e]
    loss_sum    = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count  = torch.zeros((), device=device, dtype=torch.float64)
    with torch.inference_mode():
        for bi in range(0, len(my_windows), batch_seqs):
            batch_ws = my_windows[bi:bi+batch_seqs]
            bsz = len(batch_ws)
            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws; wlens.append(wlen)
                chunk = val_data.val_tokens[ws:we+1].to(dtype=torch.int64, device=device)
                x_batch[i, :wlen] = chunk[:-1]; y_batch[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x_batch)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y_batch.reshape(-1), reduction="none").reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                s = 0 if ws == 0 else context_size
                wlen = wlens[i]
                loss_sum += nll[i, s:wlen].to(torch.float64).sum()
                token_count += float(wlen - s)
                tgt = y_batch[i, s:wlen]; prev = x_batch[i, s:wlen]
                tb = val_data.base_bytes_lut[tgt].to(torch.float64)
                tb += (val_data.has_leading_space_lut[tgt] & ~val_data.is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    base_model.train()
    return _loss_bpb(loss_sum, token_count, byte_count)

def timed_eval(label, fn, *args, **kwargs):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    val_loss, val_bpb = fn(*args, **kwargs)
    torch.cuda.synchronize()
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
    n_eff = (h.num_stem_layers + h.num_shared_loops + h.num_tail_layers)
    log(f"model_params (unique):{n_unique}  effective_depth:{n_eff}")
    log(f"weight_tying: stem={h.num_stem_layers} shared_loops={h.num_shared_loops} tail={h.num_tail_layers}")

    optimizers = Optimizers(h, base_model)
    train_loader = ShuffledSequenceLoader(h, device)

    max_wallclock_ms = 1000.0 * h.max_wallclock_seconds if h.max_wallclock_seconds > 0 else None
    if max_wallclock_ms is not None:
        max_wallclock_ms -= h.gptq_reserve_seconds * 1000.0

    def training_frac(step, elapsed_ms):
        if max_wallclock_ms is None: return step / max(h.iterations, 1)
        return elapsed_ms / max(max_wallclock_ms, 1e-9)

    def lr_mul(frac):
        if h.warmdown_frac <= 0: return 1.0
        if frac >= 1.0 - h.warmdown_frac: return max((1.0 - frac) / h.warmdown_frac, h.min_lr)
        return 1.0

    def step_fn(step, lr_scale):
        optimizers.zero_grad_all()
        train_loss = torch.zeros((), device=device)
        for micro_step in range(h.grad_accum_steps):
            if h.distributed:
                model.require_backward_grad_sync = micro_step == h.grad_accum_steps - 1
            x, y = train_loader.next_batch(h.train_batch_tokens, h.grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            train_loss += loss.detach()
            (loss / h.grad_accum_steps).backward()
        train_loss /= h.grad_accum_steps
        frac = min(step / h.muon_momentum_warmup_steps, 1.0) if h.muon_momentum_warmup_steps > 0 else 1.0
        muon_mom = (1 - frac) * h.muon_momentum_warmup_start + frac * h.muon_momentum
        for group in optimizers.optimizer_muon.param_groups:
            group["momentum"] = muon_mom
        for opt in optimizers:
            for group in opt.param_groups: group["lr"] = group["base_lr"] * lr_scale
        if h.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), h.grad_clip_norm)
        optimizers.step()
        return train_loss

    # Warmup
    if h.warmup_steps > 0:
        init_model = {n: t.detach().cpu().clone() for n, t in base_model.state_dict().items()}
        init_opt   = [copy.deepcopy(opt.state_dict()) for opt in optimizers]
        model.train()
        for ws in range(h.warmup_steps):
            step_fn(ws, 1.0)
            if ws <= 5 or (ws+1) % 10 == 0 or ws+1 == h.warmup_steps:
                log(f"warmup_step: {ws+1}/{h.warmup_steps}")
        base_model.load_state_dict(init_model, strict=True)
        for opt, st in zip(optimizers, init_opt): opt.load_state_dict(st)
        optimizers.zero_grad_all()
        if h.distributed: model.require_backward_grad_sync = True
        train_loader = ShuffledSequenceLoader(h, device)

    # EMA
    ema_state = {n: t.detach().float().clone() for n, t in base_model.state_dict().items()}

    training_time_ms = 0.0
    stop_after_step = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step = 0

    while True:
        last_step = step == h.iterations or (stop_after_step is not None and step >= stop_after_step)
        if last_step or (h.val_loss_every > 0 and step % h.val_loss_every == 0):
            torch.cuda.synchronize()
            training_time_ms += 1000.0 * (time.perf_counter() - t0)
            val_loss, val_bpb = eval_val(h, device, val_data, model)
            log(f"{step}/{h.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f}")
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        if last_step:
            if stop_after_step is not None and step < h.iterations:
                log(f"stopping_early: wallclock_cap step:{step}/{h.iterations}")
            break

        elapsed_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        train_loss = step_fn(step, lr_mul(training_frac(step, elapsed_ms)))

        with torch.no_grad():
            for name, t in base_model.state_dict().items():
                ema_state[name].mul_(h.ema_decay).add_(t.detach().float(), alpha=1.0 - h.ema_decay)

        step += 1
        approx_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)
        if h.train_log_every > 0 and (step <= 5 or step % h.train_log_every == 0):
            tok_s = step * h.train_batch_tokens / (approx_ms / 1000.0)
            log(f"{step}/{h.iterations} train_loss:{train_loss.item():.4f} "
                f"train_time:{approx_ms/60000:.1f}m tok/s:{tok_s:.0f}")

        reached_cap = max_wallclock_ms is not None and approx_ms >= max_wallclock_ms
        if h.distributed and max_wallclock_ms is not None:
            t = torch.tensor(int(reached_cap), device=device)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            reached_cap = bool(t.item())
        if stop_after_step is None and reached_cap:
            stop_after_step = step

    log(f"peak memory: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    log("ema:applying EMA weights")
    base_model.load_state_dict({n: t.to(base_model.state_dict()[n].dtype) for n, t in ema_state.items()}, strict=True)
    return base_model, compiled_model


def train_and_eval(h: Hyperparameters, device: torch.device):
    random.seed(h.seed); np.random.seed(h.seed)
    torch.manual_seed(h.seed); torch.cuda.manual_seed_all(h.seed)
    val_data = ValidationData(h, device)
    log(f"train_shards:{len(list(Path(h.datasets_dir).glob('fineweb_train_*.bin')))}")
    log(f"val_tokens:{val_data.val_tokens.numel()-1}")
    base_model, compiled_model = train_model(h, device, val_data)
    torch._dynamo.reset()
    timed_eval("pre-quant", eval_val, h, device, val_data, compiled_model)
    serialize(h, base_model, Path(__file__).read_text(encoding="utf-8"))
    if h.distributed: dist.barrier()
    eval_model = deserialize(h, device)
    compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
    timed_eval("quantized", eval_val, h, device, val_data, compiled_eval)
    if h.sliding_window_enabled:
        timed_eval("quantized_sliding", eval_val_sliding, h, device, val_data, eval_model)


def main():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    if 8 % world_size != 0: raise ValueError(f"WORLD_SIZE={world_size} must divide 8")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False
    h = Hyperparameters()
    set_logging_hparams(h)
    if h.is_main_process:
        os.makedirs("logs", exist_ok=True)
        log("Hyperparameters:")
        for k, v in sorted(vars(type(h)).items()):
            if not k.startswith("_"): log(f"  {k}: {v}")
    train_and_eval(h, device)
    if distributed: dist.destroy_process_group()

if __name__ == "__main__":
    main()
