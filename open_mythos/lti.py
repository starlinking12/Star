"""
LTI Injection with corrected spectral radius computation.
"""

import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """LTI-stable injection: h_new = A * h + B * e"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.log_dt = nn.Parameter(torch.full((1,), 0.0))
        self.log_A = nn.Parameter(torch.full((dim,), -1.0))
        self.B = nn.Parameter(torch.randn(dim, dim) * 0.01)
    
    def get_A_diag(self) -> torch.Tensor:
        log_dt_clamped = self.log_dt.clamp(-2.0, 2.0)
        log_A_clamped = self.log_A.clamp(-6.0, 0.0)
        A_diag = torch.exp(-torch.exp(log_dt_clamped) * torch.exp(log_A_clamped))
        return torch.clamp(A_diag, max=0.999)
    
    def get_spectral_radius(self) -> float:
        return self.get_A_diag().max().item()
    
    def forward(self, h: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        A_diag = self.get_A_diag()
        A_h = A_diag.unsqueeze(0).unsqueeze(0) * h
        B_e = torch.einsum('ij,bnj->bni', self.B, e)
        return A_h + B_e


class SpectralRadiusMonitor:
    """Monitor and adjust spectral radius during training."""
    
    def __init__(self, lti_injection, min_radius=0.05, max_radius=0.95, target=0.5, log_interval=100):
        self.lti = lti_injection
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.target = target
        self.log_interval = log_interval
        self.step_count = 0
        self.handle = self.lti.register_forward_hook(self._hook)
    
    def _hook(self, module, input, output):
        self.step_count += 1
        radius = self.lti.get_spectral_radius()
        if self.step_count % self.log_interval == 0:
            if radius < self.min_radius:
                print(f"[Spectral] WARNING: ρ(A)={radius:.4f} < {self.min_radius}")
            elif radius > self.max_radius:
                print(f"[Spectral] WARNING: ρ(A)={radius:.4f} > {self.max_radius}")
            else:
                print(f"[Spectral] ρ(A)={radius:.4f}")
        return output
    
    def close(self):
        self.handle.remove()