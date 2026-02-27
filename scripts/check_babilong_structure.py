from datasets import load_dataset

def check():
    try:
        ds = load_dataset('RMT-team/babilong', '128k')
        print(f"Available tasks: {list(ds.keys())}")
        for task in ['qa1', 'qa2', 'qa5']:
            if task in ds:
                sample = ds[task][0]
                print(f"\nTask: {task}")
                print(f"All keys in sample: {list(sample.keys())}")
                # Print everything except the massive input
                for k, v in sample.items():
                    if k != 'input':
                        print(f"{k}: {v}")
                # Look for evidence of facts in the input if possible
                # (Often in bAbI format, facts are marked or have a specific structure)
                if 'input' in sample:
                    # Check first 500 chars and last 500 chars
                    print(f"Input snippet (start): {sample['input'][:500]}")
                    print(f"Input snippet (end): {sample['input'][-500:]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check()
