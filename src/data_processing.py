"""
Utilities for transforming raw PGN data into (FEN, move) training examples.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import chess
import chess.pgn
from tqdm.auto import tqdm


def extract_moves_from_pgn(
    path_in: str | Path,
    path_out: str | Path,
    limit: Optional[int] = None,
    require_eval_annotations: bool = False,
) -> None:
    """
    Convert PGN games into a JSONL dataset of (FEN, move, side_to_move) tuples.

    Args:
        path_in: Source PGN file.
        path_out: Destination JSONL file.
        limit: Optional maximum number of games to process.
        require_eval_annotations: If True, only export games containing [%eval] comments.
    """
    path_in = Path(path_in)
    path_out = Path(path_out)
    path_out.parent.mkdir(parents=True, exist_ok=True)

    processed_games = 0
    total_positions = 0

    with path_in.open("r", encoding="utf-8", errors="ignore") as fin, path_out.open("w", encoding="utf-8") as fout:
        progress = tqdm(desc="processing games", unit="game")
        while True:
            game = chess.pgn.read_game(fin)
            if game is None:
                break
            if limit is not None and processed_games >= limit:
                break

            if require_eval_annotations:
                game_text = str(game)
                if "[%eval" not in game_text:
                    continue

            board = game.board()
            for move in game.mainline_moves():
                sample = {
                    "fen": board.fen(),
                    "move": move.uci(),
                    "side_to_move": 1 if board.turn == chess.WHITE else -1,
                }
                fout.write(json.dumps(sample) + "\n")
                board.push(move)
                total_positions += 1

            processed_games += 1
            progress.update()

        progress.close()

    print(f"[data_processing] processed {processed_games} games -> {total_positions} positions written to {path_out}")


if __name__ == "__main__":
    INPUT = Path("data/input/lichess_db_standard_rated_2014-08.pgn")
    OUTPUT = Path("data/working/move_dataset.jsonl")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    extract_moves_from_pgn(INPUT, OUTPUT, limit=None)
