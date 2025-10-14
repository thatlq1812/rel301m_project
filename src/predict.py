import torch
import chess
import json
import os
import numpy as np
from models import PolicyNet
from dataset import FenMoveDataset

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
    return np.stack(planes, axis=0)

@torch.no_grad()
def predict_move(fen: str, model, move2id, device, topk=5):
    model.eval()
    x = fen_to_planes(fen)
    x = torch.from_numpy(x).unsqueeze(0).float().to(device)
    with torch.amp.autocast("cuda", enabled=(device=="cuda")):
        logits = model(x)
    # lọc theo legal
    board = chess.Board(fen)
    ids = [move2id[m.uci()] for m in board.legal_moves if m.uci() in move2id]
    li = logits[0, ids]
    k = min(topk, len(ids))
    idx = torch.topk(li, k=k).indices.cpu().tolist()
    inv_vocab = {v:k for k,v in move2id.items()}
    return [inv_vocab[ids[i]] for i in idx]

if __name__ == "__main__":
    # Load model
    ckpt_path = "models/policy_model.pt"
    checkpoint = torch.load(ckpt_path)
    move2id = checkpoint["vocab"]
    vocab_size = len(move2id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = PolicyNet(vocab_size)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Example prediction
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    predictions = predict_move(fen, model, move2id, device, topk=5)
    print(f"Predicted moves for starting position: {predictions}")