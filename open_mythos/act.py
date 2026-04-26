"""
Adaptive Computation Time halting with gradient isolation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class ACTHalting(nn.Module):
    """
    ACT halting where:
    - Halted tokens: output = remainder * h (final, gradient stopped)
    - Non-halted tokens: output = h (continues evolving)
    """
    
    def __init__(self, dim: int, act_threshold: float = 0.99, max_loops: int = 64):
        super().__init__()
        self.act_threshold = act_threshold
        self.max_loops = max_loops
        self.halt_prob = nn.Linear(dim, 1)
    
    def forward(
        self,
        h: torch.Tensor,
        loop_t: int,
        halting_state: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        return_state: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        batch, seq_len, dim = h.shape
        device = h.device
        
        if halting_state is None:
            cumulative_p = torch.zeros(batch, seq_len, device=device)
            remaining = torch.ones(batch, seq_len, device=device)
            halted_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.bool)
        else:
            cumulative_p, remaining, halted_mask = halting_state
        
        p_t = torch.sigmoid(self.halt_prob(h)).squeeze(-1)
        p_remain = p_t * remaining
        
        would_exceed = (cumulative_p + p_remain) >= self.act_threshold
        is_last_step = (loop_t == self.max_loops - 1)
        halt_now = (would_exceed | is_last_step) & ~halted_mask
        
        remainder = (1.0 - cumulative_p).clamp(min=0.0, max=1.0)
        
        h_out = torch.where(halt_now.unsqueeze(-1), h * remainder.unsqueeze(-1), h)
        h_out = torch.where(halted_mask.unsqueeze(-1), torch.zeros_like(h), h_out)
        
        new_cumulative_p = (cumulative_p + p_remain).clamp(max=1.0)
        new_remaining = remaining * (1.0 - p_t)
        new_halted_mask = halted_mask | halt_now
        
        new_halting_state = (new_cumulative_p.detach(), new_remaining.detach(), new_halted_mask.detach())
        
        return h_out, new_halting_state