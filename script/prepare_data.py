"""
Data preparation script.
"""

import argparse
import json
import random
from pathlib import Path


def prepare_demo_data(output_dir: str, num_samples: int = 1000):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    code_samples = [
        '#include <stdio.h>\nint main() { printf("hello"); return 0; }',
        'def hello():\n    print("world")',
        'function test() { return 42; }',
    ]
    
    with open(output_dir / "train.jsonl", "w") as f:
        for i in range(num_samples):
            sample = {
                "code": random.choice(code_samples),
                "language": "c",
                "vulnerable_lines": [] if i % 5 else [1],
                "vulnerability_type": "" if i % 5 else "buffer_overflow",
            }
            f.write(json.dumps(sample) + "\n")
    
    print(f"Created {num_samples} samples in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--num-samples", type=int, default=1000)
    args = parser.parse_args()
    prepare_demo_data(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main()