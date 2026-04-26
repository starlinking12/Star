"""
Recurrent block with ACT halting, LTI injection, and LoRA adaptation.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict

from open_mythos.act import ACTHalting
from open_mythos.lti import LTIInjection
from open_mythos.lora import LoRAAdapter
from open_mythos.attention import GQAttention, MLAttention
from open_mythos.moe import MoEFFN
from open_mythos.norm import RMSNorm


class TransformerBlock(nn.Module):
    """Standard transformer block with pre-norm and MoE FFN."""
    
    def __init__(self, cfg, use_moe: bool = True):
        super().__init__()
        self.cfg = cfg
        self.use_moe = use_moe
        
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        
        if cfg.attn_type == "gqa":
            self.attn = GQAttention(cfg)
        else:
            self.attn = MLAttention(cfg)
        
        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            class StandardFFN(nn.Module):
                def __init__(self, dim, hidden_dim):
                    super().__init__()
                    self.w1 = nn.Linear(dim, hidden_dim)
                    self.w2 = nn.Linear(hidden_dim, dim)
                    self.w3 = nn.Linear(dim, hidden_dim)
                def forward(self, x):
                    return self.w2(F.silu(self.w1(x)) * self.w3(x))
            self.ffn = StandardFFN(cfg.dim, cfg.expert_dim)
        
        self.training_mode = "train"
    
    def forward(self, x, freqs_cis, mask=None, kv_cache=None, cache_key=None, start_pos=0):
        attn_out = self.attn(self.norm1(x), freqs_cis, mask, kv_cache, cache_key, start_pos)
        x = x + attn_out
        
        if self.use_moe and self.training_mode == "train":
            ffn_out = self.ffn.forward_with_load_balancing(self.norm2(x))
        else:
            ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        return x


class RecurrentBlock(nn.Module):
    """Recurrent block that processes hidden state through multiple loops."""
    
    def __init__(self, cfg, use_moe: bool = True):
        super().__init__()
        self.cfg = cfg
        self.max_loops = cfg.max_loop_iters
        self.act_threshold = cfg.act_threshold
        
        self.injection = LTIInjection(cfg.dim)
        self.transformer = TransformerBlock(cfg, use_moe=use_moe)
        self.lora = LoRAAdapter(cfg.dim, rank=cfg.lora_rank, max_loops=cfg.max_loop_iters)
        self.act = ACTHalting(cfg.dim, act_threshold=cfg.act_threshold, max_loops=cfg.max_loop_iters)
        self.register_buffer("loop_embedding", torch.randn(cfg.max_loop_iters, cfg.dim) * 0.02)
    
    def forward(self, h, e, freqs_cis, n_loops=None, kv_cache=None, start_pos=0):
        if n_loops is None:
            n_loops = self.max_loops
        
        batch, seq_len, dim = h.shape
        halting_state = None
        
        for t in range(n_loops):
            h = self.injection(h, e)
            if t < len(self.loop_embedding):
                loop_emb = self.loop_embedding[t].view(1, 1, -1).expand(batch, seq_len, -1)
                h = h + loop_emb
            h = self.lora(h, loop_t=t)
            h = self.transformer(h, freqs_cis, kv_cache=kv_cache, cache_key=f"layer_{t}", start_pos=start_pos)
            h_out, halting_state = self.act(h, t, halting_state, return_state=True)
            h = h_out
            if halting_state[2].all():
                break
        
        return h