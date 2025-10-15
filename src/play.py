"""
Interactive chess gameplay helpers for battling the trained policy network.

This module provides a lightweight CLI-oriented interface so that humans can
play against the model using the existing feature encoders and checkpoints.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import chess

from .features import board_to_planes, legal_move_mask, mask_logits
from .models import PolicyNet, PolicyValueNet


class ChessRLAgent:
    """
    Wrapper around the trained policy network that turns chess.Board positions
    into move selections. Supports both PolicyNet (supervised) and
    PolicyValueNet (RL) checkpoints.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: Optional[torch.device] = None,
        temperature: float = 0.8,
        deterministic: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.deterministic = deterministic

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.move2id: Dict[str, int] = checkpoint["vocab"]
        self.id2move: Dict[int, str] = {idx: move for move, idx in self.move2id.items()}

        model_state = checkpoint["model_state"]
        vocab_size = len(self.move2id)

        if any(key.startswith("value_head") for key in model_state.keys()):
            # Reinforcement learning checkpoint with value head.
            self.model = PolicyValueNet(vocab_size)
            self.model.load_state_dict(model_state)
            self.use_value_head = True
        else:
            # Supervised policy-only checkpoint.
            self.model = PolicyNet(vocab_size)
            self.model.load_state_dict(model_state)
            self.use_value_head = False

        self.model.to(self.device)
        self.model.eval()

    def select_move(self, board: chess.Board) -> chess.Move:
        """
        Select a legal move for the provided board. Uses temperature sampling by
        default but can fall back to greedy selection.
        """
        planes = board_to_planes(board)
        inputs = torch.from_numpy(planes).unsqueeze(0).float().to(self.device)
        mask = torch.from_numpy(legal_move_mask(board, self.move2id)).unsqueeze(0).to(self.device)

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            if self.use_value_head:
                logits, _ = self.model(inputs)
            else:
                logits = self.model(inputs)
        masked_logits = mask_logits(logits, mask)

        legal_count = int(mask.sum().item())
        if legal_count == 0:
            raise RuntimeError("No legal moves available; the game should already be over.")

        if self.deterministic or self.temperature <= 1e-3:
            move_index = int(torch.argmax(masked_logits, dim=-1).item())
        else:
            scaled = masked_logits / self.temperature
            probs = torch.softmax(scaled, dim=-1)
            distribution = torch.distributions.Categorical(probs=probs[0])
            move_index = int(distribution.sample().item())

        move_uci = self.id2move.get(move_index)
        move = chess.Move.from_uci(move_uci) if move_uci else None
        if move not in board.legal_moves:
            # Fallback: choose the highest scoring legal move deterministically.
            legal_indices = torch.nonzero(mask[0]).squeeze(-1)
            legal_logits = masked_logits[0, legal_indices]
            best_idx = int(legal_indices[torch.argmax(legal_logits)].item())
            move_uci = self.id2move[best_idx]
            move = chess.Move.from_uci(move_uci)

        return move


def render_board(board: chess.Board, orientation: chess.Color = chess.WHITE) -> str:
    """
    Return an ASCII representation of the board using python-chess' unicode
    helper. Orientation can be set to chess.BLACK to view from the opponent's
    perspective.
    """
    return board.unicode(borders=True, invert_color=(orientation == chess.BLACK))


def prompt_human_move(board: chess.Board) -> chess.Move:
    """
    Prompt the user for a move, accepting UCI or SAN notation. Provides basic
    command handling for resignation and legal move listing.
    """
    while True:
        user_input = input("Your move (uci or san) ['help' for options]: ").strip()
        if not user_input:
            continue
        lowered = user_input.lower()
        if lowered in {"quit", "resign"}:
            raise KeyboardInterrupt("Player resigned.")
        if lowered in {"help", "?"}:
            print("Enter moves in UCI (e2e4) or SAN (e4) format. Type 'moves' to list legal moves.")
            continue
        if lowered in {"moves", "list"}:
            san_moves = [board.san(move) for move in board.legal_moves]
            print("Legal moves:", ", ".join(san_moves))
            continue
        try:
            # Try SAN first for convenience.
            move = board.parse_san(user_input)
        except ValueError:
            try:
                move = chess.Move.from_uci(lowered)
            except ValueError:
                print("Could not parse move. Try again.")
                continue
        if move not in board.legal_moves:
            print("Illegal move. Try again.")
            continue
        return move


def human_vs_agent(
    checkpoint: Path,
    human_color: chess.Color = chess.WHITE,
    temperature: float = 0.8,
    deterministic: bool = False,
) -> None:
    """
    Play a single game of chess between a human and the trained agent using the
    selected checkpoint. The board is rendered in ASCII after every move.
    """
    agent = ChessRLAgent(checkpoint, temperature=temperature, deterministic=deterministic)
    board = chess.Board()

    print("Starting new game. Human plays", "White" if human_color == chess.WHITE else "Black")
    print(render_board(board, orientation=human_color))

    try:
        while not board.is_game_over():
            if board.turn == human_color:
                move = prompt_human_move(board)
                board.push(move)
                print(f"You played {board.peek().uci()} ({board.san(board.peek())}).")
            else:
                move = agent.select_move(board)
                board.push(move)
                print(f"Agent played {move.uci()} ({board.san(move)}).")

            print(render_board(board, orientation=human_color))

        outcome = board.outcome(claim_draw=True)
        if outcome is None:
            print("Game ended without a result.")
        else:
            if outcome.winner is None:
                print("Game drawn:", outcome.termination.name)
            elif outcome.winner == human_color:
                print("Congratulations, you won!", outcome.termination.name)
            else:
                print("Agent wins:", outcome.termination.name)

    except KeyboardInterrupt:
        print("\nGame ended by resignation.")


def run_cli_game(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        print(f"Checkpoint {checkpoint} not found.", file=sys.stderr)
        sys.exit(1)

    human_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK
    human_vs_agent(
        checkpoint=checkpoint,
        human_color=human_color,
        temperature=args.temperature,
        deterministic=args.deterministic,
    )


def build_play_parser(subparsers):
    parser = subparsers.add_parser("play", help="Play an interactive game against the trained agent.")
    parser.add_argument("--checkpoint", type=str, default="models/policy_model.pt", help="Path to model checkpoint (.pt).")
    parser.add_argument("--color", choices=["white", "black"], default="white", help="Human player's color.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for the agent policy.")
    parser.add_argument("--deterministic", action="store_true", help="Force the agent to play the highest probability move.")
    parser.set_defaults(func=run_cli_game)
