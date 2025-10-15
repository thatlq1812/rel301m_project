"""CLI utility for generating move predictions from a trained policy network."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch

from .features import fen_to_planes, legal_move_mask, mask_logits
from .models import PolicyNet


@torch.no_grad()
def predict_move(
    fen: str,
    model: PolicyNet,
    move2id: Dict[str, int],
    device: torch.device,
    topk: int = 5,
) -> List[str]:
    """Return the top-k legal moves for the given FEN using the provided model."""
    model.eval()
    planes, board = fen_to_planes(fen)
    inputs = torch.from_numpy(planes).unsqueeze(0).float().to(device)
    legal_mask = torch.from_numpy(legal_move_mask(board, move2id)).unsqueeze(0).to(device)

    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
        logits = model(inputs)
        masked_logits = mask_logits(logits, legal_mask)

    probabilities = torch.softmax(masked_logits, dim=-1)
    legal_moves_available = int(legal_mask.sum().item())
    k = min(topk, legal_moves_available) if legal_moves_available > 0 else 0
    top_indices = torch.topk(probabilities, k=k, dim=-1).indices[0].tolist() if k > 0 else []

    id2move = {idx: move for move, idx in move2id.items()}
    return [id2move[i] for i in top_indices]


def load_checkpoint(path: Path, device: torch.device):
    """Load a saved policy checkpoint from disk."""
    return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description="Predict chess moves from a trained policy network.")
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    parser.add_argument(
        "--fen",
        type=str,
        default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        help="FEN position to evaluate.",
    )
    parser.add_argument("--topk", type=int, default=5, help="Number of candidate moves to return.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    move2id = checkpoint["vocab"]

    model = PolicyNet(len(move2id))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    moves = predict_move(args.fen, model, move2id, device, topk=args.topk)
    print(f"Top-{len(moves)} moves: {moves}")


if __name__ == "__main__":
    main()
