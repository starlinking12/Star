"""
Download pretrained weights.
"""

import argparse
import hashlib
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, output_path: str):
    response = requests.get(url, stream=True)
    total = int(response.headers.get('content-length', 0))
    with open(output_path, 'wb') as f:
        with tqdm(total=total, unit='B', unit_scale=True) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    print(f"Downloaded to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["small", "large", "xl"], default="small")
    parser.add_argument("--output-dir", type=str, default="./checkpoints")
    args = parser.parse_args()
    
    urls = {
        "small": "https://huggingface.co/openmythos/openmythos-small/resolve/main/model.pt",
        "large": "https://huggingface.co/openmythos/openmythos-large/resolve/main/model.pt",
        "xl": "https://huggingface.co/openmythos/openmythos-xl/resolve/main/model.pt",
    }
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"openmythos-{args.model}.pt"
    
    download_file(urls[args.model], str(output_path))


if __name__ == "__main__":
    main()