from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
from functools import lru_cache
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PretrainedConfig, PreTrainedModel, GenerationMixin


# -----------------------------------------------------------------------------
# PyTorch Replacements for DeepSeek Custom Kernels
# -----------------------------------------------------------------------------

def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """
    Pure PyTorch implementation of DeepSeek's sparse attention kernel.
    q: [bsz, seqlen, n_heads, head_dim]
    kv: [bsz, cache_len, head_dim]
    topk_idxs: [bsz, seqlen, topk]
    attn_sink: [n_heads]
    """
    bsz, seqlen, n_heads, head_dim = q.shape
    topk = topk_idxs.shape[2]
    cache_len = kv.size(1)

    # Gather KV via flat index_select — avoids the [bsz, seqlen, cache_len, head_dim]
    # intermediate that expand+gather would materialize.
    valid_mask = topk_idxs >= 0                                          # [bsz, seqlen, topk]
    idx_clamped = topk_idxs.clamp(min=0).long()                          # [bsz, seqlen, topk]
    batch_offset = (torch.arange(bsz, device=kv.device) * cache_len).view(bsz, 1, 1)
    flat_idx = (idx_clamped + batch_offset).view(-1)                     # [bsz*seqlen*topk]
    kv_selected = kv.reshape(-1, head_dim).index_select(0, flat_idx).view(bsz, seqlen, topk, head_dim)
    kv_selected = kv_selected * valid_mask.unsqueeze(-1).type_as(kv_selected)

    # Q * K^T via einsum: [bsz, seqlen, n_heads, D] × [bsz, seqlen, topk, D] → [bsz, n_heads, seqlen, topk]
    # (K/V are shared across heads at each token position.)
    scores = torch.einsum("bshd,bstd->bhst", q, kv_selected) * softmax_scale

    valid_mask_attn = valid_mask.view(bsz, 1, seqlen, topk)              # [bsz, 1, seqlen, topk]
    scores = scores.masked_fill(~valid_mask_attn, float('-inf'))

    # Sink (attn_sink is fp32; upcast scores briefly for stable softmax, then cast back)
    sink_score = attn_sink.view(1, n_heads, 1, 1).expand(bsz, -1, seqlen, 1)  # [bsz, n_heads, seqlen, 1]
    scores = torch.cat([scores.float(), sink_score.float()], dim=-1)      # [bsz, n_heads, seqlen, topk+1]

    probs = F.softmax(scores, dim=-1).to(kv.dtype)

    # V: probs [B, H, S, T+1] × v_all [B, S, T+1, D] → out [B, H, S, D]
    v_sink = torch.zeros(bsz, seqlen, 1, head_dim, device=kv.device, dtype=kv.dtype)
    v_all = torch.cat([kv_selected, v_sink], dim=2)                       # [bsz, seqlen, topk+1, head_dim]
    out = torch.einsum("bhst,bstd->bhsd", probs, v_all)                   # [bsz, n_heads, seqlen, head_dim]
    return out.transpose(1, 2)                                            # [bsz, seqlen, n_heads, head_dim]


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
        window_size=128,
        compress_ratios=(0, 0, 4, 128, 4, 128, 4, 0),
        rope_theta=10000.0,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
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
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.index_topk = index_topk
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
        super().__init__(**kwargs)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


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
    complex-valued formulation but ONNX/RKNN-friendly — no complex ops).

    Pure functional: does NOT modify the input. Returns a new tensor of the same
    shape/dtype containing the rotated values. Callers must reassemble the full
    tensor if `x` was a rope-only slice.

    x:         [..., seqlen, D]       or  [..., seqlen, H, D]   (D == rope_head_dim, even)
    freqs_cis: [seqlen, D/2, 2]                       where [..., 0]=cos, [..., 1]=sin
               (caller must already slice by start_pos:start_pos+seqlen)
    inverse:   if True, apply the inverse rotation (uses -sin).
    """
    orig_dtype = x.dtype
    seqlen, d_half = freqs_cis.size(0), freqs_cis.size(1)

    cos = freqs_cis[..., 0]                     # [seqlen, D/2]
    sin = freqs_cis[..., 1]                     # [seqlen, D/2]
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

    x_pair = x.float().unflatten(-1, (d_half, 2))
    x_even = x_pair[..., 0]
    x_odd  = x_pair[..., 1]

    new_even = x_even * cos - x_odd  * sin
    new_odd  = x_even * sin + x_odd  * cos

    return torch.stack([new_even, new_odd], dim=-1).flatten(-2).to(orig_dtype)


def apply_rope_tail(x: torch.Tensor, freqs_cis: torch.Tensor, rd: int, inverse: bool = False) -> torch.Tensor:
    """Apply RoPE to the trailing `rd` dims of `x` (out-of-place, autograd-safe).
    Returns a new tensor with the non-rope prefix and rotated tail concatenated."""
    if rd == x.size(-1):
        return apply_rotary_emb(x, freqs_cis, inverse)
    return torch.cat([x[..., :-rd], apply_rotary_emb(x[..., -rd:], freqs_cis, inverse)], dim=-1)


@lru_cache(2)
def get_window_topk_idxs(window_size: int, bsz: int, seqlen: int, start_pos: int):
    if start_pos >= window_size - 1:
        start_pos %= window_size
        matrix = torch.cat([torch.arange(start_pos + 1, window_size),  torch.arange(0, start_pos + 1)], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


@lru_cache(2)
def get_compress_topk_idxs(ratio: int, bsz: int, seqlen: int, start_pos: int, offset: int):
    if start_pos > 0:
        matrix = torch.arange(0, (start_pos + 1) // ratio) + offset
    else:
        matrix = torch.arange(seqlen // ratio).repeat(seqlen, 1)
        mask = matrix >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
        matrix = torch.where(mask, -1, matrix + offset)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


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
        
        self.kv_cache = None
        self.register_buffer("kv_state", torch.zeros(1, coff * compress_ratio, coff * self.head_dim, dtype=torch.float32), persistent=False)
        self.register_buffer("score_state", torch.full((1, coff * compress_ratio, coff * self.head_dim), float("-inf"), dtype=torch.float32), persistent=False)
        self.freqs_cis = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        
        if self.kv_state.size(0) < bsz:
            self.kv_state = torch.zeros(bsz, self.kv_state.size(1), self.kv_state.size(2), dtype=torch.float32, device=x.device)
            self.score_state = torch.full((bsz, self.score_state.size(1), self.score_state.size(2)), float("-inf"), dtype=torch.float32, device=x.device)

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


class Indexer(torch.nn.Module):
    def __init__(self, config: DeepSeekV4MiniConfig, compress_ratio: int = 4):
        super().__init__()
        self.dim = config.hidden_size
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5
        self.compress_ratio = compress_ratio

        self.compressor = Compressor(config, compress_ratio, self.head_dim)
        self.register_buffer("kv_cache", torch.zeros(1, config.max_seq_len // compress_ratio, self.head_dim), persistent=False)
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        
        if self.kv_cache.size(0) < bsz:
            self.kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_cache.size(2), dtype=self.kv_cache.dtype, device=x.device)
            if self.compressor.kv_cache is not None:
                self.compressor.kv_cache = self.kv_cache

        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q = apply_rope_tail(q, freqs_cis, rd)
        
        kv_compress = self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        
        if self.training and start_pos == 0 and kv_compress is not None:
            # Take kv directly from Compressor return to keep gradients flowing.
            index_kv = kv_compress
        else:
            index_kv = self.kv_cache[:bsz, :end_pos // ratio]
        index_score = torch.einsum("bshd,btd->bsht", q, index_kv)
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        
        if start_pos == 0:
            mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            index_score += torch.where(mask.to(index_score.device), float("-inf"), 0)
        
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        
        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1).to(topk_idxs.device) // ratio
            topk_idxs = torch.where(mask, -1, topk_idxs + offset)
        else:
            topk_idxs += offset
        return topk_idxs


class Attention(nn.Module):
    def __init__(self, layer_id: int, config: DeepSeekV4MiniConfig):
        super().__init__()
        self.layer_id = layer_id
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratios[layer_id] if layer_id < len(config.compress_ratios) else 0
        self.eps = config.norm_eps

        self.attn_sink = nn.Parameter(torch.empty(self.n_heads, dtype=torch.float32))
        nn.init.zeros_(self.attn_sink)
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)
        
        self.wo_a = nn.Linear(self.n_heads * self.head_dim // self.n_groups, self.n_groups * self.o_lora_rank, bias=False)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        if self.compress_ratio:
            self.compressor = Compressor(config, self.compress_ratio, self.head_dim)
            if self.compress_ratio == 4:
                self.indexer = Indexer(config, self.compress_ratio)
            else:
                self.indexer = None

        kv_cache_size = self.window_size + (config.max_seq_len // self.compress_ratio if self.compress_ratio else 0)
        self.register_buffer("kv_cache", torch.zeros(1, kv_cache_size, self.head_dim), persistent=False)
        
        yarn_orig = config.original_seq_len if config.inference_rope_scaling else 0
        yarn_factor = config.rope_factor if config.inference_rope_scaling else 1.0
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, config.max_seq_len, config.rope_theta,
                                          yarn_orig, yarn_factor, config.beta_fast, config.beta_slow)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        
        if self.kv_cache.size(0) < bsz:
            self.kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_cache.size(2), dtype=self.kv_cache.dtype, device=x.device)
            if self.compress_ratio and self.compressor.kv_cache is not None:
                self.compressor.kv_cache = self.kv_cache[:, self.window_size:]

        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen].to(x.device)
        if freqs_cis.size(0) < seqlen:
            raise RuntimeError(
                f"Context length exceeded: start_pos={start_pos}, seqlen={seqlen}, "
                f"but freqs_cis was precomputed for max_seq_len={self.freqs_cis.size(0)}. "
                f"Increase --max_seq_len or enable --inference_rope_scaling."
            )
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        
        if self.compress_ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis.to(x.device)
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis.to(x.device)
        
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        q = apply_rope_tail(q, freqs_cis, rd)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        kv = apply_rope_tail(kv, freqs_cis, rd)
        
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos).to(x.device)
        
        if self.compress_ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
            else:
                compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset).to(x.device)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        
        topk_idxs = topk_idxs.int()

        if start_pos == 0:
            if not self.training:
                if seqlen <= win:
                    self.kv_cache[:bsz, :seqlen] = kv.detach()
                else:
                    cutoff = seqlen % win
                    kv_win = kv[:, -win:].detach()
                    self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv_win.split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            if not self.training:
                self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1).detach()
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
            
        o = apply_rope_tail(o, freqs_cis, rd, inverse=True)

        o = o.reshape(bsz, seqlen, self.n_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a)
        x = self.wo_b(o.flatten(2))
        return x


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
        self.router_aux_loss_coef = getattr(config, 'router_aux_loss_coef', 0.0)
        
        self.gate = Gate(layer_id, config)
        self.experts = nn.ModuleList([Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit) for _ in range(self.n_routed_experts)])
        
        self.shared_experts = Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)
        # buffer for the load-balancing loss; read by top-level model and reset each forward.
        self.aux_loss = None

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices, original_scores = self.gate(x_flat, input_ids.flatten())
        
        # Load-balancing aux loss (only during training): encourages uniform expert usage.
        # load[i] = fraction of tokens routed to expert i; scores_mean[i] = mean gate score for expert i.
        # Loss is minimized when both distributions are uniform (dot product minimized).
        if self.training and self.router_aux_loss_coef > 0 and not self.gate.hash:
            load = F.one_hot(indices.view(-1), self.n_routed_experts).float().mean(dim=0)
            scores_mean = original_scores.mean(dim=0)
            self.aux_loss = (load * scores_mean).sum() * self.n_routed_experts * self.router_aux_loss_coef
        else:
            self.aux_loss = None
        
        # Fast path for single-token decode: skip the routed-expert loop entirely.
        # x_flat has shape [N, dim] where N = bsz * seqlen. In decode, N == bsz (usually 1).
        # With num_activated_experts=1, we can just do one direct expert call per token.
        if not self.training and x_flat.size(0) <= 4 and self.n_activated_experts == 1:
            y_bf = torch.zeros_like(x_flat)
            # indices: [N, 1], weights: [N, 1]
            for token_i in range(x_flat.size(0)):
                exp_id = int(indices[token_i, 0].item())
                y_bf[token_i] = self.experts[exp_id](x_flat[token_i:token_i+1], weights[token_i:token_i+1, 0:1]).squeeze(0)
            y_bf = y_bf + self.shared_experts(x_flat)
            return y_bf.view(shape)
        
        y = torch.zeros_like(x_flat, dtype=torch.float32)

        for i in range(self.n_routed_experts):
            idx, top = torch.where(indices == i)
            if idx.numel() == 0:
                continue
            y[idx] += self.experts[i](x_flat[idx], weights[idx, top, None])
            
        y += self.shared_experts(x_flat)
        return y.type_as(x).view(shape)


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
        iters = self.hc_sinkhorn_iters if self.training else min(self.hc_sinkhorn_iters, 5)
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
        residual = x
        x_attn, post, comb = self.hc_pre(x, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base)
        x_attn = self.attn_norm(x_attn)
        x_attn = self.attn(x_attn, start_pos)
        x = self.hc_post(x_attn, residual, post, comb)

        residual = x
        x_ffn, post, comb = self.hc_pre(x, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base)
        x_ffn = self.ffn_norm(x_ffn)
        x_ffn = self.ffn(x_ffn, input_ids)
        x = self.hc_post(x_ffn, residual, post, comb)
        return x


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
        x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
        x = super().forward(x, start_pos, input_ids)
        logits = self.head(x, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        return logits


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
        hc_dim = self.hc_mult * config.hidden_size
        
        self.hc_head_fn = nn.Parameter(torch.empty(self.hc_mult, hc_dim, dtype=torch.float32))
        nn.init.normal_(self.hc_head_fn, std=0.02)
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        nn.init.zeros_(self.hc_head_base)
        self.hc_head_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        nn.init.ones_(self.hc_head_scale)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
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
        # Attention sink
        'attn_sink',
        # Compressor APE + states
        'ape',
        'kv_state', 'score_state',
    )
    # Leaf module names whose `.weight` must stay fp32 (RMSNorm family).
    _FP32_NORM_MODULES = ('norm', 'attn_norm', 'ffn_norm', 'q_norm', 'kv_norm', 'enorm', 'hnorm')

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
        h = h.unsqueeze(2).expand(-1, -1, self.hc_mult, -1).clone()
        
        for layer in self.layers:
            if self.training:
                h = checkpoint(layer, h, start_pos, input_ids, use_reentrant=True)
            else:
                h = layer(h, start_pos, input_ids)
            
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        
        # Sum load-balancing aux loss across all MoE layers (only set during training).
        aux_loss = torch.tensor(0.0, device=h.device)
        for layer in self.layers:
            ffn = getattr(layer, 'ffn', None)
            if ffn is not None and getattr(ffn, 'aux_loss', None) is not None:
                aux_loss = aux_loss + ffn.aux_loss
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=start_pos + input_ids.shape[1],
            hidden_states=h,
        )

    @torch.inference_mode()
    def generate(self, inputs, attention_mask=None, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=0, eos_token_id=None, streamer=None, **kwargs):
        input_ids = inputs
        bsz, seqlen = input_ids.shape
        start_pos = 0
        
        # Clear KV cache state before generation
        for m in self.modules():
            if isinstance(m, Compressor):
                m.kv_state.fill_(0)
                m.score_state.fill_(float("-inf"))
            if hasattr(m, "kv_cache") and m.kv_cache is not None:
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
