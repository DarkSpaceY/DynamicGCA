from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hf_cache import get_hf_hub_dir, list_cached_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opencompass-dir", default=os.environ.get("OPENCOMPASS_DIR"))
    parser.add_argument("--dataset", default="needlebench_v2_128k")
    parser.add_argument("--hf-model", default="")
    parser.add_argument("--hf-type", default="base", choices=["base", "chat"])
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-out-len", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_model_path(model_id: str) -> str:
    if Path(model_id).exists():
        return str(Path(model_id).resolve())
    return model_id


def main() -> int:
    args = parse_args()
    opencompass_dir = Path(args.opencompass_dir or SCRIPT_DIR.parent / "third_party" / "opencompass")
    run_py = opencompass_dir / "run.py"

    if not run_py.exists():
        print(f"OpenCompass not found at: {opencompass_dir}")
        print("Set OPENCOMPASS_DIR or clone https://github.com/open-compass/opencompass")
        return 1

    if not args.hf_model:
        cached = list_cached_models()
        if cached:
            args.hf_model = cached[0]
        else:
            print(f"No cached models found in {get_hf_hub_dir()}")
            return 1

    model_path = resolve_model_path(args.hf_model)

    cmd = [
        sys.executable,
        str(run_py),
        "--datasets",
        args.dataset,
        "--hf-type",
        args.hf_type,
        "--hf-path",
        model_path,
        "--tokenizer-path",
        model_path,
        "--tokenizer-kwargs",
        "padding_side='left' truncation='left' trust_remote_code=True",
        "--model-kwargs",
        "device_map='auto' trust_remote_code=True",
        "--max-seq-len",
        str(args.max_seq_len),
        "--max-out-len",
        str(args.max_out_len),
        "--batch-size",
        str(args.batch_size),
        "--hf-num-gpus",
        str(args.num_gpus),
    ]

    if args.debug:
        cmd.append("--debug")

    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=str(opencompass_dir))
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
