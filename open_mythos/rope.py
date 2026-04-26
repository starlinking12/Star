"""
Rotary Position Embeddings (RoPE).
"""

import torch
import math


def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500000.0) -> torch.Tensor:
    """Precompute RoPE frequencies for all positions."""
    assert dim % 2 == 0, "dim must be even for RoPE"
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    t = torch.arange(max_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(angles), angles)


def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to input tensor."""
    original_shape = x.shape
    x = x.float()
    dim = x.shape[-1]
    assert dim % 2 == 0, f"Last dimension {dim} must be even"
    x_complex = torch.view_as_complex(x.reshape(*x.shape[:-1], dim // 2, 2))
    freqs_cis = freqs_cis.view(1, freqs_cis.shape[0], 1, freqs_cis.shape[1])
    x_rotated = x_complex * freqs_cis
    x_out = torch.view_as_real(x_rotated).reshape(*original_shape)
    return x_out.to(x.dtype)