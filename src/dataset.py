import json
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import fen_to_planes, legal_move_mask

class FenMoveDataset(Dataset):
    """
    Supervised dataset of (FEN, move) pairs produced from historical PGN data.
    Each sample returns encoded planes, the move index, and a legality mask so
    that models can ignore illegal actions during training.
    """

    def __init__(self, path: str, move2id: Dict[str, int]):
        with open(path, "r", encoding="utf-8") as file:
            self.samples = [json.loads(line) for line in file]
        self.move2id = move2id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int, np.ndarray]:
        obj = self.samples[idx]
        planes, board = fen_to_planes(obj["fen"])
        move_idx = self.move2id.get(obj["move"], -1)
        mask = legal_move_mask(board, self.move2id)
        return planes.astype(np.float32), move_idx, mask

def collate_fn(batch):
    xs, ys, masks = zip(*batch)
    return (
        torch.from_numpy(np.stack(xs)).float(),
        torch.tensor(ys, dtype=torch.long),
        torch.from_numpy(np.stack(masks)).float()
    )
