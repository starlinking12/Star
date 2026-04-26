"""
Mixture of Experts with load balancing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class Expert(nn.Module):
    """SwiGLU FFN expert."""
    
    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, expert_dim, bias=False)
        self.w2 = nn.Linear(expert_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, expert_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with aux-loss-free load balancing."""
    
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.n_experts = cfg.n_experts
        self.n_shared_experts = cfg.n_shared_experts
        self.n_experts_per_tok = cfg.n_experts_per_tok
        self.expert_dim = cfg.expert_dim
        
        self.shared_experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_shared_experts)])
        self.routed_experts = nn.ModuleList([Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)])
        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(cfg.n_experts))
        self.bias_gamma = getattr(cfg, 'moe_bias_gamma', 1e-3)
    
    def update_router_bias(self, expert_load: torch.Tensor):
        if not self.training:
            return
        total_tokens = expert_load.sum()
        if total_tokens == 0:
            return
        expert_proportion = expert_load / total_tokens
        target_proportion = 1.0 / self.n_experts
        delta = (expert_proportion - target_proportion) * self.bias_gamma
        with torch.no_grad():
            self.router_bias -= delta
            self.router_bias.clamp_(-5.0, 5.0)
    
    def _get_expert_load(self, topk_indices: torch.Tensor) -> torch.Tensor:
        expert_load = torch.zeros(self.n_experts, device=topk_indices.device)
        flat_indices = topk_indices.view(-1)
        unique, counts = torch.unique(flat_indices, return_counts=True)
        expert_load[unique] = counts.float()
        return expert_load
    
    def _get_topk_indices_and_weights(self, router_logits):
        topk_weights, topk_indices = torch.topk(router_logits, self.n_experts_per_tok, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        return topk_indices, topk_weights
    
    def forward_with_load_balancing(self, x):
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        
        router_logits = self.router(x_flat) + self.router_bias
        topk_indices, topk_weights = self._get_topk_indices_and_weights(router_logits)
        expert_load = self._get_expert_load(topk_indices)
        
        y_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            token_indices = mask.nonzero(as_tuple=True)[0]
            weights_for_tokens = torch.zeros(len(token_indices), device=x_flat.device)
            for j, tidx in enumerate(token_indices):
                pos = (topk_indices[tidx] == i).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    weights_for_tokens[j] = topk_weights[tidx, pos[0]]
            expert_output = expert(expert_input)
            y_flat[mask] += expert_output * weights_for_tokens.unsqueeze(-1)
        
        for expert in self.shared_experts:
            y_flat += expert(x_flat)
        
        self.update_router_bias(expert_load)
        return y_flat.view(batch, seq_len, dim)
    
    def forward(self, x):
        batch, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)
        router_logits = self.router(x_flat) + self.router_bias
        topk_indices, topk_weights = self._get_topk_indices_and_weights(router_logits)
        
        y_flat = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i).any(dim=-1)
            if not mask.any():
                continue
            expert_input = x_flat[mask]
            token_indices = mask.nonzero(as_tuple=True)[0]
            weights_for_tokens = torch.zeros(len(token_indices), device=x_flat.device)
            for j, tidx in enumerate(token_indices):
                pos = (topk_indices[tidx] == i).nonzero(as_tuple=True)[0]
                if len(pos) > 0:
                    weights_for_tokens[j] = topk_weights[tidx, pos[0]]
            expert_output = expert(expert_input)
            y_flat[mask] += expert_output * weights_for_tokens.unsqueeze(-1)
        
        for expert in self.shared_experts:
            y_flat += expert(x_flat)
        
        return y_flat.view(batch, seq_len, dim)