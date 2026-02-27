from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="third_party/opencompass")
    parser.add_argument("--repo", default="https://github.com/open-compass/opencompass")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dest = Path(args.dest).resolve()
    if dest.exists():
        print(f"OpenCompass already exists: {dest}")
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(["git", "clone", args.repo, str(dest)])
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
