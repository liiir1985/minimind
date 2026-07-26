import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal, List
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F
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
    
    # Gather KV
    topk_idxs_exp = topk_idxs.unsqueeze(-1).expand(-1, -1, -1, head_dim).long()
    valid_mask = topk_idxs_exp >= 0
    idx_clamped = topk_idxs_exp.clamp(min=0)
    
    kv_expanded = kv.unsqueeze(1).expand(-1, seqlen, -1, -1)
    kv_selected = torch.gather(kv_expanded, 2, idx_clamped) # [bsz, seqlen, topk, head_dim]
    kv_selected = kv_selected * valid_mask.type_as(kv_selected)
    
    # Q * K^T
    q_trans = q.transpose(1, 2).unsqueeze(3) # [bsz, n_heads, seqlen, 1, head_dim]
    k_trans = kv_selected.unsqueeze(1) # [bsz, 1, seqlen, topk, head_dim]
    
    scores = (q_trans @ k_trans.transpose(-2, -1)) * softmax_scale # [bsz, n_heads, seqlen, 1, topk]
    
    valid_mask_attn = (topk_idxs >= 0).view(bsz, 1, seqlen, 1, topk)
    scores = scores.masked_fill(~valid_mask_attn, float('-inf'))
    
    # Sink
    sink_score = attn_sink.view(1, n_heads, 1, 1, 1).expand(bsz, -1, seqlen, 1, 1)
    scores = torch.cat([scores, sink_score], dim=-1)
    
    probs = F.softmax(scores, dim=-1)
    
    # V
    v_sink = torch.zeros(bsz, 1, seqlen, 1, head_dim, device=k_trans.device, dtype=k_trans.dtype)
    v_trans = torch.cat([k_trans, v_sink], dim=3)
    
    out = (probs @ v_trans).squeeze(3) # [bsz, n_heads, seqlen, head_dim]
    return out.transpose(1, 2)


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, hc_sinkhorn_iters, hc_eps):
    """
    Pure PyTorch implementation of Sinkhorn splitting for Hyper-Connections.
    """
    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    post_logits = mixes[:, hc_mult:2*hc_mult] * hc_scale[1] + hc_base[hc_mult:2*hc_mult]
    comb_logits = mixes[:, 2*hc_mult:] * hc_scale[2] + hc_base[2*hc_mult:]
    
    pre = torch.sigmoid(pre_logits) + hc_eps
    post = torch.sigmoid(post_logits) + hc_eps
    
    comb = comb_logits.view(-1, hc_mult, hc_mult)
    log_P = comb
    for _ in range(hc_sinkhorn_iters):
        log_P = log_P - torch.logsumexp(log_P, dim=-1, keepdim=True)
        log_P = log_P - torch.logsumexp(log_P, dim=-2, keepdim=True)
    comb = torch.exp(log_P)
    
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
        num_routed_experts=32,
        num_shared_experts=1,
        num_activated_experts=1,
        score_func="sqrtsoftplus",
        route_scale=1.0,
        swiglu_limit=0.0,
        norm_eps=1e-6,
        window_size=128,
        compress_ratios=(0,0,0,0,0,0,0,0),
        rope_theta=10000.0,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        n_hash_layers=0,
        n_mtp_layers=1,
        max_seq_len=2000,
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
def precompute_freqs_cis(dim, seqlen, base) -> torch.Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    y = x.clone() if not x.is_contiguous() else x
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_c.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_c.size(1), x_c.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_c.size(1), 1, x_c.size(-1))
    x_out = torch.view_as_real(x_c * freqs_cis).flatten(-2)
    y.copy_(x_out)
    return y


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
        x = x.float()
        kv = self.wkv(x)
        score = self.wgate(x)
        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            offset = ratio if overlap else 0
            if overlap and cutoff >= ratio:
                self.kv_state[:bsz, :ratio] = kv[:, cutoff-ratio : cutoff]
                self.score_state[:bsz, :ratio] = score[:, cutoff-ratio : cutoff] + self.ape
            if remainder > 0:
                kv, self.kv_state[:bsz, offset : offset+remainder] = kv.split([cutoff, remainder], dim=1)
                self.score_state[:bsz, offset : offset+remainder] = score[:, cutoff:] + self.ape[:remainder]
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
                self.kv_state[:bsz, ratio + start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, ratio + start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv_state = torch.cat([self.kv_state[:bsz, :ratio, :d], self.kv_state[:bsz, ratio:, d:]], dim=1)
                    score_state = torch.cat([self.score_state[:bsz, :ratio, :d], self.score_state[:bsz, ratio:, d:]], dim=1)
                    kv = (kv_state * score_state.softmax(dim=1)).sum(dim=1, keepdim=True)
                    self.kv_state[:bsz, :ratio] = self.kv_state[:bsz, ratio:]
                    self.score_state[:bsz, :ratio] = self.score_state[:bsz, ratio:]
            else:
                self.kv_state[:bsz, start_pos % ratio] = kv.squeeze(1)
                self.score_state[:bsz, start_pos % ratio] = score.squeeze(1)
                if should_compress:
                    kv = (self.kv_state[:bsz] * self.score_state[:bsz].softmax(dim=1)).sum(dim=1, keepdim=True)
        if not should_compress:
            return
        kv = self.norm(kv.to(dtype))
        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        
        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv
        else:
            self.kv_cache[:bsz, start_pos // ratio] = kv.squeeze(1)
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
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
        
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
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
        
        freqs_cis = precompute_freqs_cis(self.rope_head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

    def forward(self, x: torch.Tensor, start_pos: int):
        bsz, seqlen, _ = x.size()
        
        if self.kv_cache.size(0) < bsz:
            self.kv_cache = torch.zeros(bsz, self.kv_cache.size(1), self.kv_cache.size(2), dtype=self.kv_cache.dtype, device=x.device)
            if self.compress_ratio and self.compressor.kv_cache is not None:
                self.compressor.kv_cache = self.kv_cache[:, self.window_size:]

        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen].to(x.device)
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
        apply_rotary_emb(q[..., -rd:], freqs_cis)

        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        
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
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff: win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split([win - cutoff, cutoff], dim=1)
            if self.compress_ratio:
                kv_compress = self.compressor(x, start_pos)
                if kv_compress is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if self.compress_ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
            
        apply_rotary_emb(o[..., -rd:], freqs_cis, True)

        o = o.view(bsz, seqlen, self.n_groups, -1)
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
        if self.hash:
            self.tid2eid = nn.Parameter(torch.empty(config.vocab_size, config.num_activated_experts, dtype=torch.int32), requires_grad=False)
            self.bias = None
        else:
            self.bias = nn.Parameter(torch.empty(config.num_routed_experts, dtype=torch.float32))

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
        return weights, indices


class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int, swiglu_limit=0):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)
        self.swiglu_limit = swiglu_limit

    def forward(self, x: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        dtype = x.dtype
        gate = self.w1(x).float()
        up = self.w3(x).float()
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
        
        self.gate = Gate(layer_id, config)
        self.experts = nn.ModuleList([Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit) for _ in range(self.n_routed_experts)])
        
        self.shared_experts = Expert(config.hidden_size, config.moe_inter_dim, swiglu_limit=config.swiglu_limit)

    def forward(self, x: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x_flat = x.view(-1, self.dim)
        weights, indices = self.gate(x_flat, input_ids.flatten())
        
        y = torch.zeros_like(x_flat, dtype=torch.float32)
        
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x_flat[idx], weights[idx, top, None])
            
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
        self.hc_ffn_fn = nn.Parameter(torch.empty(mix_hc, hc_dim, dtype=torch.float32))
        self.hc_attn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_ffn_base = nn.Parameter(torch.empty(mix_hc, dtype=torch.float32))
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

    def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x_flat = x.flatten(2).float()
        rsqrt = torch.rsqrt(x_flat.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x_flat, hc_fn) * rsqrt
        pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
        y = torch.sum(pre.unsqueeze(-1) * x_flat.view(shape), dim=2)
        return y.to(dtype), post, comb

    def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
        y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
        return y.type_as(x)

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
        self.weight = nn.Parameter(torch.empty(vocab_size, dim, dtype=torch.float32))

    def get_logits(self, x):
        return F.linear(x[:, -1].float(), self.weight)

    def forward(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor, norm: RMSNorm):
        x = self.hc_head(x, hc_fn, hc_scale, hc_base)
        logits = F.linear(norm(x).float(), self.weight)
        return logits

    def hc_head(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
        shape, dtype = x.size(), x.dtype
        x = x.flatten(2).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
        mixes = F.linear(x, hc_fn) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
        y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
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
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
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
        self.hc_head_base = nn.Parameter(torch.empty(self.hc_mult, dtype=torch.float32))
        self.hc_head_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

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
            h = layer(h, start_pos, input_ids)
            
        logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
        
        # MTP loss can be computed here if labels are provided in a specific format,
        # but for compatibility with standard training loops, we return 0 aux_loss.
        aux_loss = torch.tensor(0.0, device=h.device)
        
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
