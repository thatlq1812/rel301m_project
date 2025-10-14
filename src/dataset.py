import os
import json
import numpy as np
import torch
import chess
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

class FenMoveDataset(Dataset):
    def __init__(self, path: str, move2id: dict):
        with open(path, "r", encoding="utf-8") as f:
            self.samples = [json.loads(line) for line in f]
        self.move2id = move2id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        obj = self.samples[idx]
        x, board = self.fen_to_planes(obj["fen"])
        y = self.move2id.get(obj["move"], -1)
        mask = self.legal_mask(board)
        return x.astype(np.float32), y, mask

    @staticmethod
    def fen_to_planes(fen: str):
        board = chess.Board(fen)
        planes = []
        piece_types = [chess.PAWN,chess.KNIGHT,chess.BISHOP,
                       chess.ROOK,chess.QUEEN,chess.KING]
        for color in [chess.WHITE, chess.BLACK]:
            for pt in piece_types:
                m = np.zeros((8,8), dtype=np.float32)
                for sq in board.pieces(pt, color):
                    r,c = divmod(63 - sq, 8)
                    m[r,c] = 1.0
                planes.append(m)
        # side to move
        planes.append(np.full((8,8), 1.0 if board.turn==chess.WHITE else 0.0, dtype=np.float32))
        # castling rights
        wk = np.full((8,8), board.has_kingside_castling_rights(chess.WHITE), dtype=np.float32)
        wq = np.full((8,8), board.has_queenside_castling_rights(chess.WHITE), dtype=np.float32)
        bk = np.full((8,8), board.has_kingside_castling_rights(chess.BLACK), dtype=np.float32)
        bq = np.full((8,8), board.has_queenside_castling_rights(chess.BLACK), dtype=np.float32)
        planes += [wk,wq,bk,bq]
        # en passant
        ep = np.zeros((8,8), dtype=np.float32)
        if board.ep_square is not None:
            r,c = divmod(63 - board.ep_square, 8)
            ep[r,c] = 1.0
        planes.append(ep)
        return np.stack(planes, axis=0), board

    def legal_mask(self, board: chess.Board):
        mask = np.zeros(len(self.move2id), dtype=np.float32)
        for m in board.legal_moves:
            uci = m.uci()
            if uci in self.move2id:
                mask[self.move2id[uci]] = 1.0
        return mask

def collate_fn(batch):
    xs, ys, masks = zip(*batch)
    return (
        torch.from_numpy(np.stack(xs)).float(),
        torch.tensor(ys, dtype=torch.long),
        torch.from_numpy(np.stack(masks)).float()
    )