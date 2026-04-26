"""
Multi-node training launcher.
"""

import argparse
import subprocess


def launch_slurm(num_nodes: int, num_gpus: int, config: str):
    script = f"""#!/bin/bash
#SBATCH --job-name=mythos
#SBATCH --nodes={num_nodes}
#SBATCH --ntasks-per-node={num_gpus}
#SBATCH --gres=gpu:{num_gpus}
#SBATCH --time=72:00:00

srun python -m torch.distributed.run \\
    --nnodes={num_nodes} \\
    --nproc_per_node={num_gpus} \\
    training/train.py --config {config} --distributed
"""
    with open("slurm_job.sh", "w") as f:
        f.write(script)
    subprocess.run(["sbatch", "slurm_job.sh"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--backend", type=str, default="slurm")
    parser.add_argument("--nodes", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=8)
    args = parser.parse_args()
    
    if args.backend == "slurm":
        launch_slurm(args.nodes, args.gpus, args.config)
    else:
        print(f"Running locally with {args.gpus} GPUs")
        cmd = f"python -m torch.distributed.run --nproc_per_node={args.gpus} training/train.py --config {args.config} --distributed"
        subprocess.run(cmd, shell=True)


if __name__ == "__main__":
    main()