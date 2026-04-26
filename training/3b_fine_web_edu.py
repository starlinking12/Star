"""
3B parameter OpenMythos training on FineWeb-Edu.
Streams dataset directly from HuggingFace - no download needed.
Uses FSDP for multi-GPU training.
"""

import os
import sys
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
    apply_activation_checkpointing,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig


# ============================================================================
# Streaming FineWeb-Edu Dataset
# ============================================================================

class FineWebEduStreamingDataset(torch.utils.data.IterableDataset):
    """
    Streams FineWeb-Edu shards from HuggingFace.
    Sharded across ranks and workers for disjoint coverage.
    """
    def __init__(self, tokenizer, seq_len: int = 2048, split: str = "train", 
                 rank: int = 0, world_size: int = 1, max_samples: int = None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.split = split
        self.rank = rank
        self.world_size = world_size
        self.max_samples = max_samples
        self.buffer = []

    def __iter__(self):
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        total_shards = self.world_size * num_workers
        shard_idx = self.rank * num_workers + worker_id

        # Stream directly from HuggingFace - no disk download
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split=self.split,
            streaming=True,
        ).shard(num_shards=total_shards, index=shard_idx)

        for sample in ds:
            # Simple tokenization - replace with proper tokenizer
            text = sample["text"][:10000]
            tokens = [ord(c) % 50272 for c in text]
            self.buffer.extend(tokens)
            
            while len(self.buffer) >= self.seq_len + 1:
                chunk = self.buffer[:self.seq_len + 1]
                self.buffer = self.buffer[self.seq_len + 1:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels": torch.tensor(chunk[1:], dtype=torch.long),
                }
                if self.max_samples and len(self.buffer) > self.max_samples:
                    return


# ============================================================================
# 3B Model Configuration
# ============================================================================

def get_3b_config():
    """Returns the 3B parameter Mythos configuration."""
    return MythosConfig(
        vocab_size=50272,
        dim=3200,           # 3B parameters
        n_heads=25,         # 25 query heads
        n_kv_heads=8,       # GQA: 8 KV heads
        max_seq_len=2048,
        max_loop_iters=16,
        prelude_layers=4,
        coda_layers=2,
        attn_type="mla",
        n_experts=64,       # 64 routed experts
        n_shared_experts=2,
        n_experts_per_tok=6,
        expert_dim=1408,
        act_threshold=0.99,
        lora_rank=16,
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        dropout=0.0,
    )


# ============================================================================
# Learning Rate Scheduler
# ============================================================================

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.1):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ============================================================================
# Main Training Function
# ============================================================================

def train():
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Starting 3B Mythos training on {world_size} GPUs")
        print(f"Device: {device}")

    # Model configuration
    cfg = get_3b_config()
    
    # Create model
    model = OpenMythos(cfg)
    model.to(device)
    
    # Wrap with FSDP for multi-GPU training
    auto_wrap_policy = transformer_auto_wrap_policy
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=torch.distributed.fsdp.MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        device_id=device,
    )
    
    # Enable activation checkpointing (saves memory)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=lambda m: checkpoint_wrapper(
            m, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
    )
    
    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )
    
    # Dataset (streaming, no download)
    dataset = FineWebEduStreamingDataset(
        tokenizer=None,
        seq_len=cfg.max_seq_len,
        rank=rank,
        world_size=world_size,
    )
    
    # Use DistributedSampler for proper sharding
    from torch.utils.data.distributed import DistributedSampler
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
    )
    
    # Learning rate scheduler
    total_steps = 100000  # ~30B tokens / (2*2048) per step
    warmup_steps = 2000
    scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
    
    # Training loop
    model.train()
    step = 0
    running_loss = 0.0
    
    if rank == 0:
        print(f"Total steps: {total_steps}, Warmup: {warmup_steps}")
        print("Starting training loop...")
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward pass
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        # Logging
        running_loss += loss.item()
        step += 1
        
        if rank == 0 and step % 10 == 0:
            avg_loss = running_loss / 10
            lr = scheduler.get_last_lr()[0]
            print(f"Step {step:6d} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")
            running_loss = 0.0
        
        # Save checkpoint
        if rank == 0 and step % 1000 == 0:
            checkpoint = {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "config": cfg.__dict__,
            }
            torch.save(checkpoint, f"checkpoint_step_{step}.pt")
            print(f"Saved checkpoint at step {step}")
        
        # Early stop for testing (remove for real training)
        if step >= total_steps:
            break
    
    if rank == 0:
        torch.save(model.state_dict(), "final_model_3b.pt")
        print("Training complete!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    train()