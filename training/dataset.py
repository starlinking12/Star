"""
Code dataset for training.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    def __init__(self, vocab_size: int = 32000, max_seq_len: int = 8192):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<BOS>": 2, "<EOS>": 3}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>", 2: "<BOS>", 3: "<EOS>"}
        
        chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_ \t\n\r!\"#$%&'()*+,-./:;<=>?@[\\]^`{|}~"
        for i, c in enumerate(chars, start=len(self.word_to_idx)):
            if i >= vocab_size:
                break
            self.word_to_idx[c] = i
            self.idx_to_word[i] = c
    
    def encode(self, text: str, max_length: Optional[int] = None) -> List[int]:
        max_len = max_length or self.max_seq_len
        tokens = [self.word_to_idx.get(c, self.word_to_idx["<UNK>"]) for c in text[:max_len-2]]
        tokens = [self.word_to_idx["<BOS>"]] + tokens + [self.word_to_idx["<EOS>"]]
        return tokens[:max_len]
    
    def decode(self, tokens: List[int]) -> str:
        chars = []
        for t in tokens:
            if t in self.idx_to_word and self.idx_to_word[t] not in ["<PAD>", "<BOS>", "<EOS>"]:
                chars.append(self.idx_to_word[t])
        return ''.join(chars)
    
    def pad_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        if len(tokens) >= max_length:
            return tokens[:max_length]
        return tokens + [0] * (max_length - len(tokens))
    
    @property
    def pad_token_id(self) -> int:
        return 0


class CodeDataset(Dataset):
    EXTENSIONS = ['.c', '.cpp', '.h', '.py', '.js', '.ts', '.go', '.rs', '.java']
    
    def __init__(self, data_path: str, tokenizer: SimpleTokenizer, max_seq_len: int = 8192):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.files = self._collect_files(data_path)
    
    def _collect_files(self, data_path: str) -> List[Path]:
        path = Path(data_path)
        files = []
        if path.is_file():
            files.append(path)
        else:
            for ext in self.EXTENSIONS:
                files.extend(path.rglob(f"*{ext}"))
        return files
    
    def __len__(self) -> int:
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        try:
            with open(self.files[idx], 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception:
            return self.__getitem__((idx + 1) % len(self.files))
        
        tokens = self.tokenizer.encode(content, self.max_seq_len)
        tokens = self.tokenizer.pad_sequence(tokens, self.max_seq_len)
        
        input_tokens = tokens[:-1]
        target_tokens = tokens[1:]
        
        input_tokens = input_tokens + [0] * (self.max_seq_len - 1 - len(input_tokens))
        target_tokens = target_tokens + [0] * (self.max_seq_len - 1 - len(target_tokens))
        
        return {
            "input_ids": torch.tensor(input_tokens, dtype=torch.long),
            "labels": torch.tensor(target_tokens, dtype=torch.long),
        }