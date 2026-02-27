from __future__ import annotations

import argparse
import os
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="Muennighoff/babi")
    parser.add_argument("--config", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1, help="Micro-batch size per step")
    parser.add_argument("--grad-accum-steps", type=int, default=64, help="Gradient accumulation steps")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--target-acc", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="auto", choices=["cpu", "cuda", "mps", "auto"])
    parser.add_argument("--distributions", default="beta,gaussian,laplace,studentt")
    parser.add_argument("--chunks", default="on,off")
    parser.add_argument("--lambdas", default="0,0.1", help="Comma separated diversity lambdas")
    parser.add_argument("--gates", default="gate,add", help="Integration method: gate or add")
    parser.add_argument("--save-root", default="checkpoints_grid", help="Root directory for checkpoints")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    distributions = [item.strip() for item in args.distributions.split(",") if item.strip()]
    chunks = [item.strip() for item in args.chunks.split(",") if item.strip()]
    lambdas = [item.strip() for item in args.lambdas.split(",") if item.strip()]
    gates = [item.strip() for item in args.gates.split(",") if item.strip()]
    
    os.makedirs(args.save_root, exist_ok=True)
    
    for dist in distributions:
        for chunk in chunks:
            for lam in lambdas:
                for gate in gates:
                    tag = f"{dist}_{chunk}_lam{lam}_{gate}"
                    save_dir = os.path.join(args.save_root, f"dygca_{tag}")
                    
                    if os.path.exists(os.path.join(save_dir, "dygca.pt")):
                        print(f"=== Skip {tag} (already exists) ===", flush=True)
                        continue
                        
                    print(f"=== Training {tag} ===", flush=True)
                    cmd = [
                        sys.executable, "scripts/train_dygca.py",
                        "--base-model", args.base_model,
                        "--dataset", args.dataset,
                        "--config", args.config,
                        "--split", args.split,
                        "--batch-size", str(args.batch_size),
                        "--distribution", dist,
                        "--diversity-lambda", lam,
                        "--device", args.device,
                        "--save-dir", save_dir,
                        "--target-acc", str(args.target_acc),
                        "--grad-accum-steps", str(args.grad_accum_steps),
                        "--epochs", str(args.epochs),
                    ]
                    
                    if chunk == "off":
                        cmd.append("--no-chunk-attn")
                    if gate == "add":
                        cmd.append("--no-gate")
                        
                    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
