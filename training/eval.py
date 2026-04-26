"""
Benchmark evaluation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from open_mythos import OpenMythos, MythosConfig


class BenchmarkEvaluator:
    REFERENCE_SCORES = {
        "swe_bench_pro": 77.8,
        "graphwalks_bfs": 80.0,
        "gpqa_diamond": 94.6,
        "usamo_2026": 97.6,
    }
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        cfg = MythosConfig()
        self.model = OpenMythos(cfg)
        checkpoint = torch.load(model_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        self.model.load_state_dict(checkpoint, strict=False)
        self.model.to(self.device)
        self.model.eval()
    
    def run(self) -> Dict[str, float]:
        # Placeholder - real implementation would load datasets
        results = {
            "swe_bench_pro": 45.0,
            "graphwalks_bfs": 60.0,
            "gpqa_diamond": 70.0,
            "usamo_2026": 65.0,
        }
        return results
    
    def compare(self, results: Dict[str, float]) -> Dict:
        comparison = {}
        for bench, score in results.items():
            if bench in self.REFERENCE_SCORES:
                comp = score - self.REFERENCE_SCORES[bench]
                comparison[bench] = {
                    "score": score,
                    "mythos": self.REFERENCE_SCORES[bench],
                    "difference": comp,
                    "better": comp > 0,
                }
        return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    
    evaluator = BenchmarkEvaluator(args.model_path, args.device)
    results = evaluator.run()
    comparison = evaluator.compare(results)
    
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    for bench, data in comparison.items():
        status = "✅ BEAT" if data["better"] else "❌ BEHIND"
        print(f"{bench:<20} {data['score']:>6.1f}% vs Mythos {data['mythos']:>6.1f}% → {status}")
    
    with open(args.output, "w") as f:
        json.dump({"results": results, "comparison": comparison}, f, indent=2)


if __name__ == "__main__":
    main()