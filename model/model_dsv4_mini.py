from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
from functools import lru_cache
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    """
    Pure PyTorch implementation of Sinkhorn splitting for Hyper-Connections.
    """
    pre_logits = mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[..., hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
    comb_logits = mixes[..., 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
    
    pre = torch.sigmoid(pre_logits) + hc_eps
    post = 2 * torch.sigmoid(post_logits)
    
    comb = comb_logits.view(*mixes.shape[:-1], hc_mult, hc_mult)
    comb = F.softmax(comb, dim=-1) + hc_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(hc_sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_eps)
    
    return pre, post, comb


# -----------------------------------------------------------------------------
# Config and Shared Components
# -----------------------------------------------------------------------------

class DeepSeekV4MiniConfig(PretrainedConfig):
    model_type = "dsv4_mini"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=8,
        num_attention_heads=8,
        head_dim=128,
        rope_head_dim=64,
        q_lora_rank=256,
        o_lora_rank=256,
        o_groups=4,
        moe_inter_dim=768,
        num_routed_experts=16,
        num_shared_experts=1,
        num_activated_experts=1,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=0.0,
        norm_eps=1e-6,
        window_size=512,
        compress_ratios=(0, 0, 4, 64, 4, 64, 4, 0),
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        hc_mult=0,
        hc_sinkhorn_iters=10,
        hc_eps=1e-6,
        n_hash_layers=0,
        n_mtp_layers=0,
        max_seq_len=2000,
        # MoE load-balancing auxiliary loss coefficient (0 to disable)
        router_aux_loss_coef=1e-3,
        # YaRN inference-time RoPE extrapolation (disabled by default)
        inference_rope_scaling=False,
        rope_factor=16.0,
        original_seq_len=2048,
        beta_fast=32,
        beta_slow=1,
        # HCA training chunk size. None uses `window_size`, avoiding full-sequence attention masks.
        # Set explicitly (e.g. 1024/2048) to tune activation/throughput tradeoffs.
        attn_chunk_size=None,
        ce_chunk_size=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.rope_head_dim = rope_head_dim
        self.q_lora_rank = q_lora_rank
        self.o_lora_rank = o_lora_rank
        self.o_groups = o_groups
        self.moe_inter_dim = moe_inter_dim
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.num_activated_experts = num_activated_experts
        self.score_func = score_func
        self.route_scale = route_scale
        self.swiglu_limit = swiglu_limit
        self.norm_eps = norm_eps
        self.window_size = window_size
        if isinstance(compress_ratios, (list, tuple)):
            self.compress_ratios = list(compress_ratios)
        else:
            self.compress_ratios = [0] * num_hidden_layers
        self.rope_theta = rope_theta
        self.compress_rope_theta = compress_rope_theta
        self.hc_mult = hc_mult
        self.hc_sinkhorn_iters = hc_sinkhorn_iters
        self.hc_eps = hc_eps
        self.n_hash_layers = n_hash_layers
        self.n_mtp_layers = n_mtp_layers
        self.max_seq_len = max_seq_len
        self.router_aux_loss_coef = router_aux_loss_coef
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_factor = rope_factor
        self.original_seq_len = original_seq_len
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.attn_chunk_size = attn_chunk_size
        self.ce_chunk_size = ce_chunk_size
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        weight = self.weight if self.weight.dtype == x.dtype else self.weight.to(dtype=x.dtype)
        return F.rms_norm(x, (x.size(-1),), weight, self.eps)


@lru_cache(2)
def precompute_freqs_cis(dim, seqlen, base, original_seq_len=0, factor=1.0, beta_fast=32, beta_slow=1) -> torch.Tensor:
    """Precompute complex RoPE frequencies. When original_seq_len > 0, applies YaRN scaling
    to extend RoPE beyond the trained context, with a smooth linear ramp between
    beta_fast (high-freq, kept as-is) and beta_slow (low-freq, interpolated by 1/factor)."""

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(low, high, dim):
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0 and factor > 1.0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)                       # [seqlen, dim/2]
    # Real-valued RoPE table: stack cos and sin along a new dim.
    # Shape: [seqlen, dim/2, 2]  where [..., 0]=cos, [..., 1]=sin.
    # Kept in fp32 for numerical stability across YaRN scaling.
    freqs_cis = torch.stack([freqs.cos(), freqs.sin()], dim=-1)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """
    Real-valued RoPE with adjacent-pair rotation (mathematically equivalent to the
    complex-valued formulation). Runs entirely in the input dtype (typically bf16);
    the RoPE arithmetic is a handful of multiplies/adds and is numerically robust
    at bf16 in practice.

    Pure functional: does NOT modify the input. Returns a new tensor of the same
    shape/dtype containing the rotated values. Callers must reassemble the full
    tensor if `x` was a rope-only slice.

    x:         [..., seqlen, D]       or  [..., seqlen, H, D]   (D == rope_head_dim, even)
    freqs_cis: [seqlen, D/2, 2]                       where [..., 0]=cos, [..., 1]=sin
               (caller must already slice by start_pos:start_pos+seqlen)
    inverse:   if True, apply the inverse rotation (uses -sin).
    """
    seqlen, d_half = freqs_cis.size(0), freqs_cis.size(1)

    # Decode fast path: seqlen==1 avoids the [1, 1, ...] shape juggling and lets
    # broadcasting handle everything with 4 element-wise kernels instead of 7+.
    if seqlen == 1:
        cos = freqs_cis[0, :, 0].to(x.dtype)                       # [D/2]
        sin = freqs_cis[0, :, 1].to(x.dtype)                       # [D/2]
        if inverse:
            sin = -sin
        x_pair = x.unflatten(-1, (d_half, 2))                      # [..., D/2, 2]
        x_even = x_pair[..., 0]
        x_odd  = x_pair[..., 1]
        new_even = x_even * cos - x_odd  * sin
        new_odd  = x_even * sin + x_odd  * cos
        return torch.stack([new_even, new_odd], dim=-1).flatten(-2)

    cos = freqs_cis[..., 0].to(x.dtype)         # [seqlen, D/2]
    sin = freqs_cis[..., 1].to(x.dtype)         # [seqlen, D/2]
    if inverse:
        sin = -sin

    #   x.ndim == 3 (Compressor path): x is [bsz, seqlen, D]      → cos/sin: [1, seqlen, D/2]
    #   x.ndim == 4 (Attention path):  x is [bsz, seqlen, H, D]   → cos/sin: [1, seqlen, 1, D/2]
    if x.ndim == 3:
        cos = cos.view(1, seqlen, d_half)
        sin = sin.view(1, seqlen, d_half)
    else:
        cos = cos.view(1, seqlen, 1, d_half)
        sin = sin.view(1, seqlen, 1, d_half)

    x_pair = x.unflatten(-1, (d_half, 2))
    x_even = x_pair[..., 0]
    x_odd  = x_pair[..., 1]

    new_even = x_even * cos - x_odd  * sin
    new_odd  = x_even * sin + x_odd  * cos

    return torch.stack([new_even, new_odd], dim=-1).flatten(-2)


def apply_rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE to the trailing `rd` dims of `x` (out-of-place, autograd-safe).
    Returns a new tensor with the non-rope prefix and rotated tail concatenated."""
    if rd == x.size(-1):
        return apply_rotary_emb(x, freqs_cis, inverse)
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)], dim=-1)


# -----------------------------------------------------------------------------
# DeepSeekV4 Mini Modules
# -----------------------------------------------------------------------------

class Compressor(nn.Module):
    def __init__(self, config: DeepSeekV4MiniConfig, compress_ratio: int = 4, head_dim: int = 128):
        super().__init__()
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, coff * self.head_dim, dtype=torch.float32))
        nn.init.zeros_(self.ape)
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False)
        self.norm = RMSNorm(self.head_dim, config.norm_eps)

        self.register_buffer("kv_cache", torch.zeros(1, config.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.register_buffer("kv_state", torch.zeros(1, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((1, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)

        yarn_orig = config.original_seq_len if config.inference_rope_scaling else 0
        yarn_factor = config.rope_factor if config.inference_rope_scaling else 1.0
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, config.max_seq_len, config.compress_rope_theta,
                                          yarn_orig, yarn_factor, config.beta_fast, config.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()

        if not self.training and self.kv_state.size(0) < bsz:
            self.kv_state = torch.zeros(bsz, self.kv_state.size(1), self.kv_state.size(2), dtype=torch.float32, device=x.device)
            self.score_state = torch.full((bsz, self.score_state.size(1), self.score_state.size(2)), float("-inf"), dtype=torch.float32, device=x.device)
            self.kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_cache.size(2), dtype=self.kv_cache.dtype, device=x.device)

        ratio, overlap, d, rd = self.compress_ratio, self.overlap, self.head_dim, self.rope_head_dim
        dtype = x.dtype
        # mm in x's dtype (typically bf16) for tensor-core speed; cast weights if needed.
        w_kv, w_gate = self.wkv.weight, self.wgate.weight
        if w_kv.dtype != x.dtype:
            w_kv = w_kv.to(x.dtype)
            w_gate = w_gate.to(x.dtype)
        kv = F.linear(x, w_kv)
        score = F.linear(x, w_gate)
        # Upcast to fp32 for softmax / state accumulation / ape addition.
        kv = kv.float()
        score = score.float()
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if not self.training:
                if overlap and cutoff >= ratio:
                    self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff].detach()
                    self.score_state[:bsz, :ratio] = (score[:, cutoff-ratio : cutoff] + self.ape).detach()
                if remainder > 0:
                    self.kv_state[:bsz, offset : offset+remainder] = kv[:, cutoff:].detach()
                    self.score_state[:bsz, offset : offset+remainder] = (score[:, cutoff:] + self.ape[:remainder]).detach()
            if remainder > 0:
                kv = kv[:, :cutoff]
                score = score[:, :cutoff]
            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape
            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))
            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            score += self.ape[start_pos % ratio]
            if overlap:
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1).detach()
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1).detach()
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:].detach()
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:].detach()
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1).detach()
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1).detach()
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        
        kv = apply_rope_tail(kv, freqs_cis, rd)
        
        if not self.training:
            if start_pos == 0:
                self.kv_cache[:bsz, :seqlen // ratio] = kv.detach()
            else:
                self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1).detach()
        return kv


class Attention(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0
        self.eps = config.norm_eps
        self.attn_chunk_size = config.attn_chunk_size

        self.q_proj = nn.Linear(self.dim, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        self.comp_gate = nn.Linear(self.dim, self.n_heads, bias=True)
        self.comp_gate._is_hca_comp_gate = True
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.dim, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)

        kv_cache_size = self.window_size
        self.register_buffer("kv_cache", torch.zeros(1, kv_cache_size, self.head_dim), persistent=False)
        
        # Pure sliding-window layers keep the high-resolution base RoPE. Only HCA
        # layers use the long-range compression theta and inference-time YaRN.
        if self.compress_ratio:
            rope_theta = config.compress_rope_theta
            yarn_orig = config.original_seq_len if config.inference_rope_scaling else 0
            yarn_factor = config.rope_factor if config.inference_rope_scaling else 1.0
        else:
            rope_theta = config.rope_theta
            yarn_orig = 0
            yarn_factor = 1.0
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, config.max_seq_len, rope_theta,
                                          yarn_orig, yarn_factor, config.beta_fast, config.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def _sdpa_single_kv(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """SDPA for multi-head Q over one shared latent K/V bank.

        q:  [B, S, H, D]
        kv: [B, T, D]
        returns [B, S, H, D]
        """
        bsz, _, n_heads, _ = q.shape
        q = q.transpose(1, 2)
        k = kv[:, None].expand(bsz, n_heads, -1, -1)
        v = k
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.softmax_scale,
        )
        return out.transpose(1, 2)

    def _ordered_window_cache(self, bsz: int, start_pos: int) -> torch.Tensor:
        win = self.window_size
        cache = self.kv_cache[:bsz]
        if start_pos < win:
            return cache[:, : start_pos + 1]
        return cache

    def _raw_sliding_sdpa(self, q: torch.Tensor, kv: torch.Tensor, start_pos: int) -> torch.Tensor:
        bsz, seqlen, _, _ = q.shape
        win = self.window_size

        if start_pos > 0:
            return self._sdpa_single_kv(q, self._ordered_window_cache(bsz, start_pos))

        chunk = self.attn_chunk_size or win
        out = torch.empty_like(q)
        device = q.device
        for s0 in range(0, seqlen, chunk):
            s1 = min(s0 + chunk, seqlen)
            k0 = max(0, s0 - win + 1)
            kv_c = kv[:, k0:s1]
            q_pos = torch.arange(s0, s1, device=device).view(-1, 1)
            k_pos = torch.arange(k0, s1, device=device).view(1, -1)
            mask = (k_pos <= q_pos) & (k_pos >= q_pos - win + 1)
            out[:, s0:s1] = self._sdpa_single_kv(q[:, s0:s1], kv_c, mask)
        return out

    def _compressed_prefix_len(self, pos: int) -> int:
        if not self.compress_ratio:
            return 0
        # A compressed block ending at e is visible only after it is outside the raw window:
        # e <= pos - window_size.  Blocks are written at index end // ratio.
        return max(0, (pos - self.window_size + 1) // self.compress_ratio)

    def _far_compressed_sdpa(self, q: torch.Tensor, comp_kv: torch.Tensor | None, start_pos: int) -> torch.Tensor:
        out = torch.zeros_like(q)
        if comp_kv is None or comp_kv.size(1) == 0:
            return out

        if start_pos > 0:
            prefix_len = min(self._compressed_prefix_len(start_pos), comp_kv.size(1))
            if prefix_len > 0:
                return self._sdpa_single_kv(q, comp_kv[:, :prefix_len])
            return out

        seqlen = q.size(1)
        chunk = self.attn_chunk_size or self.window_size
        for s0 in range(0, seqlen, chunk):
            s1 = min(s0 + chunk, seqlen)
            prefix_len = min(self._compressed_prefix_len(s0), comp_kv.size(1))
            if prefix_len > 0:
                out[:, s0:s1] = self._sdpa_single_kv(q[:, s0:s1], comp_kv[:, :prefix_len])
        return out

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()

        if not self.training and self.kv_cache.size(0) < bsz:
            self.kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_cache.size(2), dtype=self.kv_cache.dtype, device=x.device)

        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        if freqs_cis.size(0) < seqlen:
            raise RuntimeError(
                f"Context length exceeded: start_pos={start_pos}, seqlen={seqlen}, "
                f"but freqs_cis was precomputed for max_seq_len={self.freqs_cis.size(0)}. "
                f"Increase --max_seq_len or enable --inference_rope_scaling."
            )
        win = self.window_size
        rd = self.rope_head_dim

        q = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
        q = F.rms_norm(q, (self.head_dim,), eps=self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv = apply_rope_tail(kv, freqs_cis, rd)

        if start_pos == 0:
            if not self.training:
                if seqlen <= win:
                    self.kv_cache[:bsz, :seqlen] = kv.detach()
                else:
                    cutoff = seqlen % win
                    kv_win = kv[:, -win:].detach()
                    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv_win.split([win - cutoff, cutoff], dim=1)
            comp_kv = None
            if self.compress_ratio:
                comp_kv = self.compressor(x, start_pos)
            o = self._raw_sliding_sdpa(q, kv, start_pos)
            if self.compress_ratio:
                o_comp = self._far_compressed_sdpa(q, comp_kv, start_pos)
                gate = torch.sigmoid(self.comp_gate(x)).unsqueeze(-1).to(o.dtype)
                o = o + gate * o_comp
        else:
            if not self.training:
                self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1).detach()
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = self._raw_sliding_sdpa(q, kv, start_pos)
            if self.compress_ratio:
                o_comp = self._far_compressed_sdpa(q, self.compressor.kv_cache[:bsz], start_pos)
                gate = torch.sigmoid(self.comp_gate(x)).unsqueeze(-1).to(o.dtype)
                o = o + gate * o_comp
            
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)
        return self.o_proj(o.reshape(bsz, seqlen, -1))


class Gate(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__()
        self.dim = config.hidden_size
        self.topk = config.num_activated_experts
        self.score_func = config.score_func
        self.route_scale = config.route_scale
        self.hash = layer_id < config.n_hash_layers
        self.weight = nn.Parameter(torch.empty(config.num_routed_experts, config.hidden_size))
        nn.init.normal_(self.weight, std=0.02)
        if self.hash:
            self.tid2eid = nn.Parameter(torch.randint(0, config.num_routed_experts, (config.vocab_size, config.num_activated_experts), dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(config.num_routed_experts, dtype=torch.float32))
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, input_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = F.linear(x.float(), self.weight.float())
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1)
        elif self.score_func == "sigmoid":
            scores = scores.sigmoid()
        else:
            scores = F.softplus(scores).sqrt()
            
        original_scores = scores
        if self.bias is not None:
            scores = scores + self.bias
            
        if self.hash:
            indices = self.tid2eid[input_ids]
        else:
            indices = scores.topk(self.topk, dim=-1)[1]
            
        weights = original_scores.gather(1, indices)
        if self.score_func != "softmax":
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)
        weights = weights * self.route_scale
        return weights, indices, original_scores


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit=0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x)
        up = self.w3(x)
        if self.swiglu_limit > 0:
            up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            gate = torch.clamp(gate, max=self.swiglu_limit)
        x = F.silu(gate) * up
        if weights is not None:
            x = weights * x
        return self.w2(x.to(dtype))


class MoE(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_routed_experts = config.num_routed_experts
        self.n_activated_experts = config.num_activated_experts
        self.n_shared_experts = config.num_shared_experts
        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.0)

        # Dense fast path: single routed expert, top-1, no shared expert — collapses
        # to a plain FFN with zero routing overhead (no Gate, no dispatch, no aux loss).
        self.dense_mode = (
            self.n_routed_experts == 1
            and self.n_activated_experts == 1
            and self.n_shared_experts == 0
        )

        if self.dense_mode:
            self.expert = Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)
            return

        self.gate = Gate(layer_id, config)
        self.experts = nn.ModuleList([Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit) for _ in range(self.n_routed_experts)])

        if self.n_shared_experts > 0:
            self.shared_experts = Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)
        else:
            self.shared_experts = None

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        if self.dense_mode:
            return self.expert(x), x.new_zeros(())

        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices, original_scores = self.gate(x_flat, input_ids.flatten())
        
        # Load-balancing aux loss (only during training): encourages uniform expert usage.
        # load[i] = fraction of tokens routed to expert i; scores_mean[i] = mean gate score for expert i.
        # Loss is minimized when both distributions are uniform (dot product minimized).
        if self.training and self.router_aux_loss_coef > 0 and not self.gate.hash:
            load = F.one_hot(indices.view(-1), self.n_routed_experts).float().mean(dim=0)
            scores_mean = original_scores.mean(dim=0)
            aux_loss = (load * scores_mean).sum() * self.n_routed_experts * self.router_aux_loss_coef
        else:
            aux_loss = x.new_zeros(())
        
        # Fast path for single-token decode: skip the routed-expert loop entirely.
        # x_flat has shape [N, dim] where N = bsz * seqlen. In decode, N == bsz (usually 1).
        # With num_activated_experts=1, we can just do one direct expert call per token.
        if not self.training and x_flat.size(0) <= 4 and self.n_activated_experts == 1:
            y_bf = torch.zeros_like(x_flat)
            # indices: [N, 1], weights: [N, 1]
            for token_i in range(x_flat.size(0)):
                exp_id = int(indices[token_i, 0].item())
                y_bf[token_i] = self.experts[exp_id](x_flat[token_i:token_i+1], weights[token_i:token_i+1, 0:1]).squeeze(0)
            if self.shared_experts is not None:
                y_bf = y_bf + self.shared_experts(x_flat)
            return y_bf.view(shape), aux_loss
        
        y = torch.zeros_like(x_flat, dtype=torch.float32)

        for i in range(self.n_routed_experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            y[idx] += self.experts[i](x_flat[idx], weights[idx, top, None])

        if self.shared_experts is not None:
            y += self.shared_experts(x_flat)
        return y.type_as(x).view(shape), aux_loss


class Block(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__()
        self.layer_id = layer_id
        self.norm_eps = config.norm_eps
        self.attn = Attention(layer_id, config)
        self.ffn = MoE(layer_id, config)
        self.attn_norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, self.norm_eps)

        self.hc_mult = config.hc_mult
        self.hc_sinkhorn_iters = config.hc_sinkhorn_iters
        self.hc_eps = config.hc_eps

        if self.hc_mult > 0:
            mix_hc = (2 + self.hc_mult) * self.hc_mult
            hc_dim = self.hc_mult * config.hidden_size

            self.hc_attn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            nn.init.normal_(self.hc_attn_fn, std=0.02)
            self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
            nn.init.normal_(self.hc_ffn_fn, std=0.02)
            self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            nn.init.zeros_(self.hc_attn_base)
            self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
            nn.init.zeros_(self.hc_ffn_base)
            self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            nn.init.ones_(self.hc_attn_scale)
            self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            nn.init.ones_(self.hc_ffn_scale)

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)
        # mm: 若 hc_fn 与 x_flat 同 dtype (推理: 皆 bf16), 走 tensor core; 否则 cast.
        if hc_fn.dtype != x_flat.dtype:
            hc_fn = hc_fn.to(x_flat.dtype)
        mixes = F.linear(x_flat, hc_fn)
        # rsqrt / sinkhorn 需要 fp32
        x_flat_f = x_flat.float()
        rsqrt = torch.rsqrt(x_flat_f.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = mixes.float() * rsqrt
        # Fewer Sinkhorn iterations at inference — the matrix converges quickly and
        # extra iterations just add kernel-launch overhead in decode-bound cases.
        iters = self.hc_sinkhorn_iters if self.training else min(self.hc_sinkhorn_iters, 1)
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale.float(), hc_base.float(), self.hc_mult, iters, self.hc_eps)
        # y[b,s,d] = Σ_m pre[b,s,m] * x[b,s,m,d]  — replaces broadcast-then-sum which
        # materialized a [B,S,M,D] intermediate.
        y = torch.einsum("bsm,bsmd->bsd", pre, x_flat_f.view(shape))
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        # Original:
        #   y = post.unsqueeze(-1) * x.unsqueeze(-2)                             # [B,S,M,D]
        #     + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)    # sums over "input" hc dim
        # The second term materialized a [B,S,M,M,D] intermediate. Both terms rewritten as einsum:
        #   term1[b,s,m,d] = post[b,s,m] * x[b,s,d]                     ("bsm,bsd->bsmd")
        #   term2[b,s,m,d] = Σ_n comb[b,s,n,m] * residual[b,s,n,d]     ("bsnm,bsnd->bsmd")
        # einsum requires matching dtypes; post/comb are fp32 (from Sinkhorn), x/residual bf16 → cast.
        dtype = x.dtype
        y = torch.einsum("bsm,bsd->bsmd", post.to(dtype), x) + torch.einsum("bsnm,bsnd->bsmd", comb.to(dtype), residual)
        return y

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if self.hc_mult == 0:
            # Standard pre-norm residual (Llama/GPT style) when HC is disabled.
            # x has shape [B, S, D] here (no hyper-connection M-dim).
            h = x + self.attn(self.attn_norm(x), start_pos)
            ffn_out, aux_loss = self.ffn(self.ffn_norm(h), input_ids)
            return h + ffn_out, aux_loss

        residual = x
        x_attn, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x_attn = self.attn_norm(x_attn)
        x_attn = self.attn(x_attn, start_pos)
        x = self.hc_post(x_attn, residual, post, comb)

        residual = x
        x_ffn, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x_ffn = self.ffn_norm(x_ffn)
        x_ffn, aux_loss = self.ffn(x_ffn, input_ids)
        x = self.hc_post(x_ffn, residual, post, comb)
        return x, aux_loss


class ParallelHead(nn.Module):
    def __init__(self, vocab_size: int, dim: int, norm_eps: float = 1e-6, hc_eps: float = 1e-6):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.norm_eps = norm_eps
        self.hc_eps = hc_eps
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.bfloat16))
        nn.init.normal_(self.weight, std=0.02)

    def get_logits(self, x):
        return F.linear(x[:, -1].to(self.weight.dtype), self.weight).float()

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        if hc_fn is None:
            # HC disabled: x is already [B, S, D]; skip the hc_head mixing step.
            logits = F.linear(norm(x).to(self.weight.dtype), self.weight).float()
            return logits
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        # bf16 mm for speed; upcast to fp32 for numerically-stable loss/log_softmax downstream
        logits = F.linear(norm(x).to(self.weight.dtype), self.weight).float()
        return logits

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2)
        if hc_fn.dtype != x_flat.dtype:
            hc_fn = hc_fn.to(x_flat.dtype)
        mixes = F.linear(x_flat, hc_fn)
        x_flat_f = x_flat.float()
        rsqrt = torch.rsqrt(x_flat_f.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = mixes.float() * rsqrt
        pre = torch.sigmoid(mixes * hc_scale[0].float() + hc_base.float()) + self.hc_eps
        # y[b,s,d] = Σ_m pre[b,s,m] * x[b,s,m,d]  — same optimization as hc_pre.
        y = torch.einsum("bsm,bsmd->bsd", pre, x_flat_f.view(shape))
        return y.to(dtype)


class MTPBlock(Block):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__(layer_id, config)
        self.e_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.h_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.enorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, config.norm_eps)
        self.norm = RMSNorm(config.hidden_size, config.norm_eps)
        self.hc_mult = config.hc_mult
        if self.hc_mult > 0:
            hc_dim = self.hc_mult * config.hidden_size

            self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
            nn.init.normal_(self.hc_head_fn, std=0.02)
            self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
            nn.init.zeros_(self.hc_head_base)
            self.hc_head_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            nn.init.ones_(self.hc_head_scale)
        self.embed = None
        self.head = None

    def forward(self, x: torch.Tensor, start_pos: int, input_ids: torch.Tensor) -> torch.Tensor:
        e = self.embed(input_ids)
        e = self.enorm(e)
        x = self.hnorm(x)
        if self.hc_mult > 0:
            x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        else:
            x = self.e_proj(e) + self.h_proj(x)
        x, aux_loss = super().forward(x, start_pos, input_ids)
        hc_fn = self.hc_head_fn if self.hc_mult > 0 else None
        hc_scale = self.hc_head_scale if self.hc_mult > 0 else None
        hc_base = self.hc_head_base if self.hc_mult > 0 else None
        logits = self.head(x, hc_fn, hc_scale, hc_base, self.norm)
        return logits, aux_loss


# -----------------------------------------------------------------------------
# DeepSeekV4MiniForCausalLM (HuggingFace Compatible)
# -----------------------------------------------------------------------------

@dataclass
class MoeCausalLMOutputWithPast:
    loss: Optional[torch.FloatTensor] = None
    aux_loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[int] = None
    hidden_states: Optional[torch.FloatTensor] = None


class DeepSeekV4MiniForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = DeepSeekV4MiniConfig
    
    def __init__(self, config: DeepSeekV4MiniConfig):
        super().__init__(config)
        self.max_seq_len = config.max_seq_len
        self.norm_eps = config.norm_eps
        self.hc_eps = config.hc_eps
        
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        
        self.layers = nn.ModuleList([Block(layer_id, config) for layer_id in range(config.num_hidden_layers)])
        
        self.norm = RMSNorm(config.hidden_size, self.norm_eps)
        self.head = ParallelHead(config.vocab_size, config.hidden_size, self.norm_eps, self.hc_eps)
        
        self.mtp = nn.ModuleList()
        for layer_id in range(config.n_mtp_layers):
            mtp_block = MTPBlock(config.num_hidden_layers + layer_id, config)
            mtp_block.embed = self.embed
            mtp_block.head = self.head
            self.mtp.append(mtp_block)
            
        self.hc_mult = config.hc_mult
        if self.hc_mult > 0:
            hc_dim = self.hc_mult * config.hidden_size

            self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
            nn.init.normal_(self.hc_head_fn, std=0.02)
            self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
            nn.init.zeros_(self.hc_head_base)
            self.hc_head_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
            nn.init.ones_(self.hc_head_scale)
        else:
            self.hc_head_fn = None
            self.hc_head_base = None
            self.hc_head_scale = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if getattr(module, "_is_hca_comp_gate", False):
            torch.nn.init.zeros_(module.weight)
            torch.nn.init.constant_(module.bias, -2.0)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Parameter):
            # Only init 1D/2D general parameters. Specific ones like scaling should be initialized properly.
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    # Parameter/buffer name suffixes/patterns that must remain fp32 even after `to_inference_dtype()`
    # (softmax / rsqrt / Sinkhorn / state accumulation — all numerically sensitive).
    # These are FULL leaf-module.attribute keys — matching is done on module-name boundaries,
    # so `gate.weight` will NOT match `wgate.weight`.
    _FP32_INFERENCE_LEAF_KEYS = (
        # HC scale/base params (Block + MTPBlock + ParallelHead + top-level)
        'hc_attn_scale', 'hc_attn_base',
        'hc_ffn_scale', 'hc_ffn_base',
        'hc_head_scale', 'hc_head_base',
        # MoE
        'gate.weight', 'gate.bias',
        # Compressor APE + states
        'ape',
        'kv_state', 'score_state',
    )
    _FP32_NORM_MODULES = ()

    def to_inference_dtype(self, dtype=torch.bfloat16):
        """Cast the model to `dtype` (default bf16) for fast inference, but preserve
        the fp32 dtype of parameters/buffers that need numerical stability. Idempotent."""
        self.to(dtype)

        def should_keep_fp32(name: str) -> bool:
            parts = name.split('.')
            # RMSNorm weight: last two parts are (<one of _FP32_NORM_MODULES>, 'weight')
            if len(parts) >= 2 and parts[-1] == 'weight' and parts[-2] in self._FP32_NORM_MODULES:
                return True
            # Leaf-key match: check trailing 1- or 2-component suffix against keys.
            for key in self._FP32_INFERENCE_LEAF_KEYS:
                key_parts = key.split('.')
                if len(parts) >= len(key_parts) and parts[-len(key_parts):] == key_parts:
                    return True
            return False

        for name, param in self.named_parameters():
            if should_keep_fp32(name):
                param.data = param.data.float()
        for name, buf in self.named_buffers():
            if should_keep_fp32(name):
                buf.data = buf.data.float()
        return self

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[int] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        start_pos = past_key_values if past_key_values is not None else 0
            
        h = self.embed(input_ids)
        if self.hc_mult > 0:
            h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).clone()

        aux_loss = h.new_zeros(())
        for layer in self.layers:
            if self.training:
                h, layer_aux = checkpoint(layer, h, start_pos, input_ids, use_reentrant=True)
            else:
                h, layer_aux = layer(h, start_pos, input_ids)
            aux_loss = aux_loss + layer_aux

        loss = None
        if labels is not None:
            weight = self.head.weight
            V = self.config.vocab_size
            S = h.size(1)
            labels_dev = labels.to(h.device)
            ce_chunk = getattr(self.config, 'ce_chunk_size', None)

            if ce_chunk is None or ce_chunk >= S - 1:
                # Original one-shot CE — materialize full [B, S, V] fp32 logits.
                logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
                shift_logits = logits[..., :-1, :].contiguous().view(-1, V)
                shift_labels = labels_dev[..., 1:].contiguous().view(-1)
                loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
            else:
                # Chunked CE along seqlen — avoids materializing the [B, S, V] fp32
                # logits tensor. Each chunk independently applies (hc_head + norm +
                # linear + CE) and only its own [B, chunk, V] logits ever exist.
                # Mathematically identical to the un-chunked version since CE is a
                # per-token sum with `reduction='sum'`, divided by total valid count.
                total_loss = h.new_zeros((), dtype=torch.float32)
                total_count = h.new_zeros((), dtype=torch.long)
                # Predict label at position p+1 from hidden at position p; valid p in [0, S-2].
                for s0 in range(0, S - 1, ce_chunk):
                    s1 = min(s0 + ce_chunk, S - 1)
                    h_chunk = h[:, s0:s1]
                    if self.hc_mult > 0:
                        x_chunk = self.head.hc_head(h_chunk, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
                    else:
                        x_chunk = h_chunk
                    x_chunk = self.norm(x_chunk).to(weight.dtype)
                    logits_chunk = F.linear(x_chunk, weight).float()          # [B, chunk, V]
                    labels_chunk = labels_dev[:, s0 + 1:s1 + 1]
                    total_loss = total_loss + F.cross_entropy(
                        logits_chunk.reshape(-1, V),
                        labels_chunk.reshape(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    total_count = total_count + (labels_chunk != -100).sum()
                loss = total_loss / total_count.clamp(min=1)
                # Return only the last-position logits — a tiny [B, 1, V] slice —
                # to preserve HF API compatibility for callers that peek at .logits.
                last_h = h[:, -1:]
                if self.hc_mult > 0:
                    last_x = self.head.hc_head(last_h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
                else:
                    last_x = last_h
                logits = F.linear(self.norm(last_x).to(weight.dtype), weight).float()
        else:
            logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=start_pos + input_ids.shape[1],
            hidden_states=h,
        )

    @torch.inference_mode()
    def generate(self, inputs, attention_mask=None, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=0, eos_token_id=None, streamer=None, repetition_penalty=1.0, frequency_penalty=0.0, **kwargs):
        input_ids = inputs
        bsz, seqlen = input_ids.shape
        start_pos = 0
        
        # Clear KV cache state before generation
        for m in self.modules():
            if isinstance(m, Compressor):
                m.kv_state.fill_(0)
                m.score_state.fill_(float("-inf"))
            if hasattr(m, "kv_cache"):
                m.kv_cache.fill_(0)
        
        generated = input_ids.clone()
        next_token = None
        
        for i in range(max_new_tokens):
            if i == 0:
                out = self(input_ids, past_key_values=0)
                next_token_logits = out.logits[:, -1, :]
            else:
                out = self(next_token.unsqueeze(1), past_key_values=start_pos)
                next_token_logits = out.logits[:, -1, :]
            
            start_pos += (input_ids.shape[1] if i == 0 else 1)
            
            # 重复惩罚 + 频率惩罚 (对采样和贪心两种模式都生效)
            if repetition_penalty != 1.0 or frequency_penalty != 0.0:
                for batch_idx in range(bsz):
                    seen_ids = generated[batch_idx]
                    if repetition_penalty != 1.0:
                        uniq = torch.unique(seen_ids)
                        score = next_token_logits[batch_idx, uniq]
                        next_token_logits[batch_idx, uniq] = torch.where(
                            score > 0, score / repetition_penalty, score * repetition_penalty)
                    if frequency_penalty != 0.0:
                        # logits[t] -= frequency_penalty * count(t)
                        counts = torch.bincount(seen_ids, minlength=next_token_logits.shape[-1]).to(next_token_logits.dtype)
                        next_token_logits[batch_idx] -= frequency_penalty * counts
            
            if temperature > 0 and do_sample:
                next_token_logits = next_token_logits / temperature
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                for batch_idx in range(bsz):
                    indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                    next_token_logits[batch_idx, indices_to_remove] = -float('Inf')
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1)
                
            if streamer is not None:
                streamer.put(next_token)
                
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=-1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
                
        if streamer is not None:
            streamer.end()
            
        return generated
