"""
3B Parameter Mythos Model Configuration.
"""

from open_mythos import MythosConfig


def mythos_3b() -> MythosConfig:
    """Returns configuration for 3B parameter model."""
    return MythosConfig(
        # Architecture
        vocab_size=50272,
        dim=3200,
        n_heads=25,
        n_kv_heads=8,
        max_seq_len=2048,
        max_loop_iters=16,
        prelude_layers=4,
        coda_layers=2,
        attn_type="mla",
        
        # MoE (Mixture of Experts)
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=6,
        expert_dim=1408,
        
        # ACT Halting
        act_threshold=0.99,
        
        # LoRA
        lora_rank=16,
        
        # MLA (Multi-Latent Attention)
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        
        # Regularization
        dropout=0.0,
        rope_theta=500000.0,
    )


def mythos_1b() -> MythosConfig:
    """Returns configuration for 1B parameter model."""
    return MythosConfig(
        vocab_size=50272,
        dim=2048,
        n_heads=16,
        n_kv_heads=8,
        max_seq_len=2048,
        max_loop_iters=16,
        prelude_layers=4,
        coda_layers=2,
        attn_type="mla",
        n_experts=32,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=1024,
        act_threshold=0.99,
        lora_rank=8,
        kv_lora_rank=256,
        q_lora_rank=768,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
    )


def mythos_70b() -> MythosConfig:
    """Returns configuration for 70B parameter model (large)."""
    return MythosConfig(
        vocab_size=128000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=16384,
        max_loop_iters=32,
        prelude_layers=8,
        coda_layers=4,
        attn_type="mla",
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=4096,
        act_threshold=0.99,
        lora_rank=16,
        kv_lora_rank=1024,
        q_lora_rank=2048,
        qk_rope_head_dim=128,
        qk_nope_head_dim=256,
        v_head_dim=256,
    )