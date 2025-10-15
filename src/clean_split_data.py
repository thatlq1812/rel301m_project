"""
Dataset cleaning and splitting helpers.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple


def clean_dataset(data_path: str | Path, out_file: str | Path) -> Tuple[int, int]:
    """
    Remove duplicate (fen, move, side_to_move) triples from the dataset.

    Returns:
        Tuple[int, int]: number of kept samples, number of removed samples.
    """
    data_path = Path(data_path)
    out_file = Path(out_file)
    seen = set()
    kept = 0
    removed = 0

    with data_path.open("r", encoding="utf-8") as fin, out_file.open("w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            key = (obj["fen"], obj["move"], obj["side_to_move"])
            if key in seen:
                removed += 1
                continue
            seen.add(key)
            kept += 1
            fout.write(json.dumps(obj) + "\n")

    print(f"[clean_split] wrote cleaned dataset to {out_file} (kept={kept}, removed={removed})")
    return kept, removed


def split_jsonl_dataset(
    in_file: str | Path,
    out_dir: str | Path,
    seed: int = 42,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> None:
    """Shuffle and split a JSONL dataset into train/val/test partitions."""
    in_file = Path(in_file)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with in_file.open("r", encoding="utf-8") as fin:
        data: List[Dict[str, Any]] = [json.loads(line) for line in fin]

    random.seed(seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train.jsonl": data[:n_train],
        "val.jsonl": data[n_train : n_train + n_val],
        "test.jsonl": data[n_train + n_val :],
    }

    for name, subset in splits.items():
        path = out_dir / name
        with path.open("w", encoding="utf-8") as fout:
            for obj in subset:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
        print(f"[clean_split] {name}: {len(subset)} samples -> {path}")


if __name__ == "__main__":
    data_path = Path("data/working/move_dataset.jsonl")
    out_dir = Path("data/working")
    if not data_path.exists():
        raise SystemExit(f"Input file {data_path} does not exist. Run data_processing first.")
    clean_file = out_dir / "move_dataset_clean.jsonl"
    clean_dataset(data_path, clean_file)
    split_jsonl_dataset(clean_file, out_dir)
