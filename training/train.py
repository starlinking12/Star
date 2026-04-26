"""
Main training loop for OpenMythos.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig
from training.dataset import CodeDataset, SimpleTokenizer


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = self._create_model()
        self.model.to(self.device)
        self.optimizer = self._create_optimizer()
        self.scaler = GradScaler() if args.mixed_precision else None
        
        self.train_loader, self.val_loader, self.tokenizer = self._create_dataloaders()
        self.step = 0
    
    def _create_model(self):
        cfg = MythosConfig(
            vocab_size=self.args.vocab_size,
            dim=self.args.dim,
            n_heads=self.args.n_heads,
            n_kv_heads=self.args.n_kv_heads,
            max_seq_len=self.args.max_seq_len,
            max_loop_iters=self.args.max_loops,
            prelude_layers=self.args.prelude_layers,
            coda_layers=self.args.coda_layers,
            attn_type=self.args.attn_type,
            n_experts=self.args.n_experts,
            n_shared_experts=self.args.n_shared_experts,
            n_experts_per_tok=self.args.n_experts_per_tok,
            expert_dim=self.args.expert_dim,
            act_threshold=self.args.act_threshold,
            lora_rank=self.args.lora_rank,
        )
        return OpenMythos(cfg)
    
    def _create_optimizer(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
    
    def _create_dataloaders(self):
        tokenizer = SimpleTokenizer(vocab_size=self.args.vocab_size)
        dataset = CodeDataset(self.args.train_data, tokenizer=tokenizer, max_seq_len=self.args.max_seq_len)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=self.args.num_workers)
        return loader, None, tokenizer
    
    def train(self):
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.args.num_epochs):
            epoch_loss = 0.0
            for batch in self.train_loader:
                self.step += 1
                batch = batch["input_ids"].to(self.device)
                
                if self.scaler:
                    with autocast():
                        logits = self.model(batch)
                        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
                    self.scaler.scale(loss).backward()
                    if self.args.grad_clip > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(batch)
                    loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), batch.view(-1))
                    loss.backward()
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                epoch_loss += loss.item()
                
                if self.step % self.args.log_interval == 0:
                    avg_loss = epoch_loss / self.step
                    self.logger.info(f"Step {self.step} | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f}")
            
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            self.logger.info(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--dim", type=int, default=2048)
    parser.add_argument("--n-heads", type=int, default=16)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--max-loops", type=int, default=16)
    parser.add_argument("--prelude-layers", type=int, default=4)
    parser.add_argument("--coda-layers", type=int, default=2)
    parser.add_argument("--attn-type", type=str, default="mla")
    parser.add_argument("--n-experts", type=int, default=32)
    parser.add_argument("--n-shared-experts", type=int, default=2)
    parser.add_argument("--n-experts-per-tok", type=int, default=4)
    parser.add_argument("--expert-dim", type=int, default=1024)
    parser.add_argument("--act-threshold", type=float, default=0.99)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    
    trainer = Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()