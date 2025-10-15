"""
AlphaZero-style move vocabulary construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import chess


def build_alphazero_4672() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Reproduce the 4,672-move encoding used in AlphaZero-style policy heads.

    Returns:
        A tuple of (move2id, id2move) dictionaries.
    """
    move2id: Dict[str, int] = {}
    id2move: Dict[int, str] = {}
    idx = 0

    directions = [
        (1, 0),
        (0, 1),
        (-1, 0),
        (0, -1),  # rook-like
        (1, 1),
        (-1, 1),
        (1, -1),
        (-1, -1),  # bishop-like
    ]
    knight_offsets = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
    underpromo_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]

    for square in chess.SQUARES:
        rank, file = divmod(square, 8)

        # Sliding moves (rook and bishop directions with up to 7 steps).
        for dr, dc in directions:
            for distance in range(1, 8):
                r, c = rank + dr * distance, file + dc * distance
                if 0 <= r < 8 and 0 <= c < 8:
                    to_sq = r * 8 + c
                    uci = chess.Move(square, to_sq).uci()
                else:
                    uci = f"null_{square}_{dr}_{dc}_{distance}"
                move2id[uci] = idx
                id2move[idx] = uci
                idx += 1

        # Knight moves.
        for dr, dc in knight_offsets:
            r, c = rank + dr, file + dc
            if 0 <= r < 8 and 0 <= c < 8:
                to_sq = r * 8 + c
                uci = chess.Move(square, to_sq).uci()
            else:
                uci = f"null_knight_{square}_{dr}_{dc}"
            move2id[uci] = idx
            id2move[idx] = uci
            idx += 1

        # Under-promotions (3 files * 3 pieces).
        for dc in (-1, 0, 1):
            for promo_piece in underpromo_pieces:
                if rank == 6:  # white pawn promotion
                    r, c = rank + 1, file + dc
                    if 0 <= c < 8:
                        to_sq = r * 8 + c
                        uci = chess.Move(square, to_sq, promotion=promo_piece).uci()
                    else:
                        uci = f"null_promo_w_{square}_{dc}_{promo_piece}"
                elif rank == 1:  # black pawn promotion
                    r, c = rank - 1, file + dc
                    if 0 <= c < 8:
                        to_sq = r * 8 + c
                        uci = chess.Move(square, to_sq, promotion=promo_piece).uci()
                    else:
                        uci = f"null_promo_b_{square}_{dc}_{promo_piece}"
                else:
                    uci = f"null_promo_{square}_{dc}_{promo_piece}"
                move2id[uci] = idx
                id2move[idx] = uci
                idx += 1

    print(f"[vocabulary] generated {len(move2id)} moves")
    return move2id, id2move


if __name__ == "__main__":
    move2id, _ = build_alphazero_4672()
    workdir = Path("data/working")
    workdir.mkdir(parents=True, exist_ok=True)
    vocab_path = workdir / "move_vocab_4672.json"
    with vocab_path.open("w", encoding="utf-8") as fout:
        json.dump(move2id, fout)
    print(f"Vocabulary written to {vocab_path}")
