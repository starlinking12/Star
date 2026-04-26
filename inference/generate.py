"""
Text generation with OpenMythos.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.0
    do_sample: bool = True
    n_loops: int = 16


class Generator:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg = MythosConfig()
        self.model = OpenMythos(cfg)
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            if "model_state_dict" in checkpoint:
                checkpoint = checkpoint["model_state_dict"]
            self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def generate(self, input_ids: torch.Tensor, config: GenerationConfig) -> torch.Tensor:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                n_loops=config.n_loops,
            )
        
        return output_ids


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    generator = Generator(args.model_path, args.device)
    input_ids = torch.tensor([[ord(c) % 32000 for c in args.prompt[:100]]])
    output_ids = generator.generate(input_ids, GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    ))
    output_text = ''.join(chr(id % 128) for id in output_ids[0].tolist() if id < 128)
    print(output_text)


if __name__ == "__main__":
    main()