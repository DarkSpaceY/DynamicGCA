from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional


def get_hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser().resolve()
    return Path.home() / ".cache" / "huggingface"


def get_hf_hub_dir() -> Path:
    return get_hf_cache_dir() / "hub"


def list_model_dirs() -> List[Path]:
    hub_dir = get_hf_hub_dir()
    if not hub_dir.exists():
        return []
    return sorted([p for p in hub_dir.iterdir() if p.is_dir() and p.name.startswith("models--")])


def to_model_id(cache_dir_name: str) -> str:
    return cache_dir_name.replace("models--", "").replace("--", "/")


def list_cached_models() -> List[str]:
    return [to_model_id(p.name) for p in list_model_dirs()]


def filter_models(models: Iterable[str], patterns: Optional[List[str]] = None) -> List[str]:
    if not patterns:
        return sorted(set(models))
    lowered = [p.lower() for p in patterns]
    result = []
    for model in models:
        name = model.lower()
        if any(p in name for p in lowered):
            result.append(model)
    return sorted(set(result))
