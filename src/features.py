"""
Utility functions for converting chess boards into neural-network-ready tensors
and building legality masks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import chess

PIECE_TYPES = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


def board_to_planes(board: chess.Board) -> np.ndarray:
    """
    Encode the given chess board into the standard 18-plane AlphaZero-style
    representation. The planes capture piece placement, side to move, castling
    rights, and en-passant opportunities.
    """
    planes = []
    for color in (chess.WHITE, chess.BLACK):
        for piece_type in PIECE_TYPES:
            plane = np.zeros((8, 8), dtype=np.float32)
            for square in board.pieces(piece_type, color):
                # Flip the board so that white is always at the bottom.
                rank, file = divmod(63 - square, 8)
                plane[rank, file] = 1.0
            planes.append(plane)

    # Side to move.
    planes.append(np.full((8, 8), 1.0 if board.turn == chess.WHITE else 0.0, dtype=np.float32))

    # Castling rights.
    planes.append(np.full((8, 8), board.has_kingside_castling_rights(chess.WHITE), dtype=np.float32))
    planes.append(np.full((8, 8), board.has_queenside_castling_rights(chess.WHITE), dtype=np.float32))
    planes.append(np.full((8, 8), board.has_kingside_castling_rights(chess.BLACK), dtype=np.float32))
    planes.append(np.full((8, 8), board.has_queenside_castling_rights(chess.BLACK), dtype=np.float32))

    # En-passant target square.
    ep = np.zeros((8, 8), dtype=np.float32)
    if board.ep_square is not None:
        rank, file = divmod(63 - board.ep_square, 8)
        ep[rank, file] = 1.0
    planes.append(ep)

    return np.stack(planes, axis=0)


def fen_to_planes(fen: str) -> Tuple[np.ndarray, chess.Board]:
    """Convenience helper to convert a FEN string into planes and return the board."""
    board = chess.Board(fen)
    return board_to_planes(board), board


def legal_move_mask(board: chess.Board, move2id: Dict[str, int]) -> np.ndarray:
    """
    Build a mask over the move vocabulary where legal moves are 1.0 and illegal
    moves are 0.0 for the provided board state.
    """
    mask = np.zeros(len(move2id), dtype=np.float32)
    for move in board.legal_moves:
        uci = move.uci()
        idx = move2id.get(uci)
        if idx is not None:
            mask[idx] = 1.0
    return mask


def mask_logits(logits, legal_mask):
    """
    Mask logits in-place by setting illegal move logits to -inf. Accepts torch
    tensors and returns the masked tensor for convenience.
    """
    return logits.masked_fill(legal_mask == 0, float("-inf"))

