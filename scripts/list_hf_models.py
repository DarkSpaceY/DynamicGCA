from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from hf_cache import filter_models, get_hf_cache_dir, get_hf_hub_dir, list_cached_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--patterns", nargs="*", default=None)
    parser.add_argument("--top", type=int, default=0)
    return parser.parse_args()


def is_small_model(model_id: str) -> bool:
    name = model_id.lower()
    keywords = [
        "0.5b",
        "1.5b",
        "2b",
        "3b",
        "mini",
        "small",
        "distil",
        "gpt2",
        "phi-4-mini",
    ]
    return any(k in name for k in keywords)


def main() -> int:
    args = parse_args()
    cache_dir = get_hf_cache_dir()
    hub_dir = get_hf_hub_dir()
    models = list_cached_models()

    if not models:
        print(f"No cached models found in {hub_dir}")
        return 1

    selected = models if args.all else [m for m in models if is_small_model(m)]
    selected = filter_models(selected, args.patterns)
    if args.top and args.top > 0:
        selected = selected[: args.top]

    print(f"HF cache dir: {cache_dir}")
    print(f"HF hub dir: {hub_dir}")
    for model in selected:
        print(model)

    return 0


if __name__ == "__main__":
    sys.exit(main())
