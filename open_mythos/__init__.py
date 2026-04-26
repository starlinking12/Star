"""
OpenMythos: Recurrent-Depth Transformer for front-tier AI.
"""

from open_mythos.model import OpenMythos, MythosConfig
from open_mythos.norm import RMSNorm
from open_mythos.act import ACTHalting
from open_mythos.lti import LTIInjection
from open_mythos.moe import MoEFFN, Expert
from open_mythos.attention import GQAttention, MLAttention
from open_mythos.recurrent import RecurrentBlock, TransformerBlock
from open_mythos.rope import precompute_rope_freqs, apply_rope
from open_mythos.lora import LoRAAdapter

__all__ = [
    "OpenMythos",
    "MythosConfig",
    "RMSNorm",
    "ACTHalting",
    "LTIInjection",
    "MoEFFN",
    "Expert",
    "GQAttention",
    "MLAttention",
    "RecurrentBlock",
    "TransformerBlock",
    "precompute_rope_freqs",
    "apply_rope",
    "LoRAAdapter",
]

__version__ = "1.0.0"