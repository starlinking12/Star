"""
Full OpenMythos model with prelude, recurrent core, and coda.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from open_mythos.norm import RMSNorm
from open_mythos.recurrent import RecurrentBlock, TransformerBlock
from open_mythos.rope import precompute_rope_freqs


class MythosConfig:
    """Configuration class for OpenMythos model."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        dim: int = 2048,
        n_heads: int = 16,
        n_kv_heads: int = 8,
        max_seq_len: int = 8192,
        max_loop_iters: int = 16,
        prelude_layers: int = 4,
        coda_layers: int = 2,
        attn_type: str = "mla",
        n_experts: int = 32,
        n_shared_experts: int = 2,
        n_experts_per_tok: int = 4,
        expert_dim: int = 1024,
        act_threshold: float = 0.99,
        lora_rank: int = 8,
        kv_lora_rank: int = 128,
        q_lora_rank: int = 256,
        qk_rope_head_dim: int = 32,
        qk_nope_head_dim: int = 64,
        v_head_dim: int = 128,
        dropout: float = 0.0,
        rope_theta: float = 500000.0,
    ):
        self.vocab_size = vocab_size
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.max_seq_len = max_seq_len
        self.max_loop_iters = max_loop_iters
        self.prelude_layers = prelude_layers
        self.coda_layers = coda_layers
        self.attn_type = attn_type
        self.n_experts = n_experts
        self.n_shared_experts = n_shared_experts
        self.n_experts_per_tok = n_experts_per_tok
        self.expert_dim = expert_dim
        self.act_threshold = act_threshold
        self.lora_rank = lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.dropout = dropout
        self.rope_theta = rope_theta


class OpenMythos(nn.Module):
    """OpenMythos: Recurrent-Depth Transformer."""
    
    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.prelude = nn.ModuleList([TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)])
        self.recurrent = RecurrentBlock(cfg, use_moe=True)
        self.coda = nn.ModuleList([TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)])
        self.norm = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.head.weight = self.embed.weight
        
        head_dim = cfg.dim // cfg.n_heads
        rope_dim = cfg.qk_rope_head_dim if cfg.attn_type == "mla" else head_dim
        self.register_buffer("freqs_cis", precompute_rope_freqs(rope_dim, cfg.max_seq_len, cfg.rope_theta))
        self.register_buffer("causal_mask", torch.triu(torch.full((1, 1, cfg.max_seq_len, cfg.max_seq_len), float("-inf")), diagonal=1))
    
    def forward(self, input_ids, n_loops=None, kv_cache=None, start_pos=0, return_hidden=False):
        batch, seq_len = input_ids.shape
        device = input_ids.device
        
        h = self.embed(input_ids)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        mask = self.causal_mask[:, :, :seq_len, :seq_len].to(device) if seq_len > 1 else None
        
        for i, block in enumerate(self.prelude):
            h = block(h, freqs_cis, mask, kv_cache, f"prelude_{i}", start_pos)
        
        e = h.clone()
        h = self.recurrent(h, e, freqs_cis, n_loops, kv_cache, start_pos)
        
        for i, block in enumerate(self.coda):
            h = block(h, freqs_cis, mask, kv_cache, f"coda_{i}", start_pos)
        
        h = self.norm(h)
        logits = self.head(h)
        
        return (logits, h) if return_hidden else logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=256, temperature=0.8, top_k=50, top_p=0.95, n_loops=None, eos_token_id=2):
        device = input_ids.device
        generated = input_ids
        kv_cache = {}
        start_pos = 0
        
        for _ in range(max_new_tokens):
            logits = self(generated[:, -1:], n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            next_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][:, -1:]
                next_logits[indices_to_remove] = float("-inf")
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)
            start_pos += 1
            
            if (next_token == eos_token_id).all():
                break
        
        return generated