import os
import torch
from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
from datasets import load_dataset

def is_model_cached(model_name: str) -> bool:
    """Check if the model is available in the local Hugging Face cache."""
    # If it's a local path that exists, return True
    if os.path.isdir(model_name):
        return True
    
    # Try to find a key file (like config.json) in the cache
    try:
        # Use try_to_load_from_cache to check for the presence of config.json
        filepath = try_to_load_from_cache(model_name, "config.json")
        if filepath is not None and filepath != _CACHED_NO_EXIST:
            return True
    except Exception as e:
        # Log error but return False to allow attempting a regular load
        # print(f"Error checking model cache for {model_name}: {e}")
        pass
    return False


def is_dataset_cached(dataset_name: str, config: str = None) -> bool:
    """Check if the dataset is available in the local cache."""
    if os.path.isdir(dataset_name):
        return True
    
    # Normalize empty config string to None
    if config == "":
        config = None
        
    try:
        # Attempt to load metadata/config with local_files_only
        # For some datasets, passing name=None or name="default" matters
        load_dataset(dataset_name, config, local_files_only=True)
        return True
    except Exception:
        return False
