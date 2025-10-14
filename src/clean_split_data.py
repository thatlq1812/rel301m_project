import os
import json

def clean_dataset(data_path: str, out_file: str):
    seen = set()
    kept = 0
    removed = 0

    with open(data_path, "r") as fin, open(out_file, "w") as fout:
        for line in fin:
            obj = json.loads(line)
            key = (obj["fen"], obj["move"], obj["side_to_move"])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept += 1
            fout.write(json.dumps(obj) + "\n")

    print(f"Saved cleaned dataset: {out_file}")
    print(f"Kept: {kept}, Removed: {removed}, Total: {kept+removed}")
    return kept, removed

import random
from typing import List, Dict, Any

def split_jsonl_dataset(
    in_file: str,
    out_dir: str,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:

    with open(in_file, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = [json.loads(line) for line in f]

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(train_ratio * n)
    n_val   = int(val_ratio * n)

    splits = {
        "train.jsonl": data[:n_train],
        "val.jsonl":   data[n_train:n_train + n_val],
        "test.jsonl":  data[n_train + n_val:],
    }

    os.makedirs(out_dir, exist_ok=True)
    for name, subset in splits.items():
        path = os.path.join(out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            for obj in subset:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"Saved {name}: {len(subset)} samples")

if __name__ == "__main__":
    data_path = "data/working/move_dataset.jsonl"
    out_dir = "data/working"
    
    if not os.path.exists(data_path):
        print(f"Error: Input file {data_path} does not exist. Please run data processing first.")
        exit(1)
    
    clean_file = os.path.join(out_dir, "move_dataset_clean.jsonl")
    kept, removed = clean_dataset(data_path, clean_file)
    split_jsonl_dataset(clean_file, out_dir)