import bz2
import io
import re
import json
import os
import chess
import chess.pgn

EVAL_RE = re.compile(r"\[%eval\s+([+#\-0-9\.]+)\]")

def extract_moves_from_pgn(path_in: str, path_out: str, limit=None):
    # First, load and filter games with eval like in the notebook
    print("Loading and filtering games with eval...")
    with open(path_in, "r", encoding="utf-8", errors="ignore") as f:
        data = f.read()
    data = str(data)  # Convert to string for easier functionality
    raw_games = data.split('[Event')  # Split the data into chess games using the '[Event' string
    print("Game at 0th index: %s" % raw_games[0])
    del raw_games[0]  # The first index isn't a game
    del data  # Remove string to save memory

    eval_games = []
    for game in raw_games:
        if game.find('eval') != -1:
            eval_games.append(game)

    print(f"Found {len(eval_games)} games with eval out of {len(raw_games)} total games")

    # Now process the filtered games
    n_games, n_rows = 0, 0
    with open(path_out, "w", encoding="utf-8") as fout:
        for game_str in eval_games:
            # Reconstruct full PGN string
            full_pgn = '[Event' + game_str
            game = chess.pgn.read_game(io.StringIO(full_pgn))
            if game is None:
                continue
            n_games += 1
            board = game.board()
            node = game
            ply = 0
            while node.variations:
                next_node = node.variation(0)
                move = next_node.move
                fen_before = board.fen()
                move_uci = move.uci()
                fout.write(json.dumps({
                    "fen": fen_before,
                    "move": move_uci,
                    "side_to_move": 1 if board.turn else -1
                }) + "\n")
                board.push(move)
                node = next_node
                ply += 1
                n_rows += 1
            if limit and n_games >= limit:
                break
    print(f"[OK] {n_games} games, {n_rows} moves → {path_out}")

if __name__ == "__main__":
    # Adjust paths as needed
    path_in = "data/input/lichess_db_standard_rated_2014-08.pgn"
    path_out = "data/working/move_dataset.jsonl"
    os.makedirs("data/working", exist_ok=True)
    extract_moves_from_pgn(path_in, path_out, limit=100000)