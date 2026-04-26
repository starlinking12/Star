"""
Run benchmarks and compare to Mythos.
"""

import argparse
import json
import subprocess
from datetime import datetime


REFERENCE_SCORES = {
    "swe_bench_pro": 77.8,
    "graphwalks_bfs": 80.0,
    "gpqa_diamond": 94.6,
    "usamo_2026": 97.6,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output", type=str, default="report.md")
    args = parser.parse_args()
    
    # Run evaluation
    result = subprocess.run(
        ["python", "training/eval.py", "--model-path", args.model_path, "--output", "results.json"],
        capture_output=True, text=True
    )
    
    with open("results.json", "r") as f:
        results = json.load(f)
    
    # Generate markdown report
    lines = [f"# Benchmark Report\n", f"Date: {datetime.now()}\n", "| Benchmark | Score | Mythos | Diff |", "|-----------|-------|--------|------|"]
    
    for bench, mythos_score in REFERENCE_SCORES.items():
        our_score = results.get(bench, 0)
        diff = our_score - mythos_score
        status = "✅" if diff > 0 else "❌" if diff < 0 else "="
        lines.append(f"| {bench} | {our_score:.1f}% | {mythos_score:.1f}% | {diff:+.1f}% {status} |")
    
    with open(args.output, "w") as f:
        f.write("\n".join(lines))
    
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()