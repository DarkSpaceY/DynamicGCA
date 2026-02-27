from datasets import load_dataset
import os

def check_training_sets():
    datasets_to_check = [
        ("pg19", "default"),
        ("hoskinson-center/proof-pile", "default"),
        ("kmfoda/booksum", "default")
    ]
    
    # Use proxy if provided in environment
    proxy = os.getenv("HTTP_PROXY")
    if proxy:
        print(f"Using proxy: {proxy}")

    for ds_name, config in datasets_to_check:
        try:
            print(f"\nChecking dataset: {ds_name}")
            # Just load the metadata/info
            ds = load_dataset(ds_name, config, split="train", streaming=True)
            # Try to get 1 sample
            for sample in ds.take(1):
                print(f"Successfully accessed {ds_name}")
                print(f"Sample keys: {list(sample.keys())}")
        except Exception as e:
            print(f"Error accessing {ds_name}: {e}")

if __name__ == "__main__":
    check_training_sets()
