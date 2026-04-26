"""
Vulnerability fine-tuning with contrastive learning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig


class VulnerabilityDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_seq_len: int = 2048):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.examples = self._load(data_path)
    
    def _load(self, data_path: str) -> List[Dict]:
        examples = []
        path = Path(data_path)
        if path.is_file():
            with open(path, 'r') as f:
                if path.suffix == '.jsonl':
                    for line in f:
                        if line.strip():
                            examples.append(json.loads(line))
                else:
                    data = json.load(f)
                    examples.extend(data if isinstance(data, list) else [data])
        return examples
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        ex = self.examples[idx]
        code = ex.get("code", "")
        # Simple tokenization
        tokens = [ord(c) % 32000 for c in code[:self.max_seq_len]]
        input_ids = tokens + [0] * (self.max_seq_len - len(tokens))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "label": torch.tensor(1 if ex.get("vulnerable_lines") else 0, dtype=torch.long),
        }


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2, labels):
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        logits = torch.matmul(z1, z2.T) / self.temperature
        return F.cross_entropy(logits, labels)


def fine_tune(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MythosConfig()
    model = OpenMythos(cfg)
    
    if args.pretrained_path:
        checkpoint = torch.load(args.pretrained_path, map_location=device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.train()
    
    # Placeholder dataset
    dataset = VulnerabilityDataset(args.train_data, None, args.max_seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.num_epochs):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            logits, hidden = model(input_ids, return_hidden=True)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        
        print(f"Epoch {epoch} | Loss: {total_loss / len(loader):.4f}")
    
    torch.save(model.state_dict(), args.output_path)
    print(f"Model saved to {args.output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--pretrained-path", type=str)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    args = parser.parse_args()
    fine_tune(args)


if __name__ == "__main__":
    main()