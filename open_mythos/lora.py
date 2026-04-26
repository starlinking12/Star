"""
LoRA adapters for per-loop depth adaptation.
"""

import math
import torch
import torch.nn as nn


class LoRAAdapter(nn.Module):
    """Low-Rank Adaptation that varies per loop iteration."""
    
    def __init__(self, dim: int, rank: int = 8, max_loops: int = 64):
        super().__init__()
        self.dim = dim
        self.rank = rank
        self.max_loops = max_loops
        
        self.A = nn.Parameter(torch.zeros(max_loops, dim, rank))
        self.B = nn.Parameter(torch.zeros(max_loops, rank, dim))
        self.scale = nn.Parameter(torch.ones(1) * 0.01)
        
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)
    
    def forward(self, x: torch.Tensor, loop_t: int = 0) -> torch.Tensor:
        if loop_t >= self.max_loops:
            loop_t = self.max_loops - 1
        
        lora = torch.einsum("bsd,d r, r d->bsd", x, self.A[loop_t], self.B[loop_t])
        return x + lora * self.scale