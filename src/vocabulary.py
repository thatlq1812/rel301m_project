import chess
import json
import os

def build_alphazero_4672():
    move2id = {}
    id2move = {}
    idx = 0
    
    directions = [
        (1,0),(0,1),(-1,0),(0,-1),   # rook
        (1,1),(-1,1),(1,-1),(-1,-1)  # bishop
    ]
    knight_offsets = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]
    promo_pieces = [chess.ROOK, chess.BISHOP, chess.KNIGHT]  # queen promotion coi như mặc định
    
    for sq in chess.SQUARES:
        r0, c0 = divmod(sq, 8)
        
        # sliding moves (56 = 8 directions × 7 steps)
        for dr,dc in directions:
            for k in range(1,8):
                r, c = r0+dr*k, c0+dc*k
                if 0 <= r < 8 and 0 <= c < 8:
                    to_sq = r*8+c
                    uci = chess.Move(sq, to_sq).uci()
                else:
                    uci = f"null_{sq}_{dr}_{dc}_{k}"  # dummy move
                move2id[uci] = idx
                id2move[idx] = uci
                idx += 1

        # knight moves (8)
        for dr,dc in knight_offsets:
            r, c = r0+dr, c0+dc
            if 0 <= r < 8 and 0 <= c < 8:
                to_sq = r*8+c
                uci = chess.Move(sq, to_sq).uci()
            else:
                uci = f"null_knight_{sq}_{dr}_{dc}"
            move2id[uci] = idx
            id2move[idx] = uci
            idx += 1

        # underpromotions (9 = 3 dirs × 3 promos)
        for dc in [-1,0,1]:
            for promo in promo_pieces:
                if r0 == 6:  # white pawn promotion
                    r, c = r0+1, c0+dc
                    if 0 <= c < 8:
                        to_sq = r*8+c
                        uci = chess.Move(sq, to_sq, promotion=promo).uci()
                    else:
                        uci = f"null_promo_w_{sq}_{dc}_{promo}"
                elif r0 == 1:  # black pawn promotion
                    r, c = r0-1, c0+dc
                    if 0 <= c < 8:
                        to_sq = r*8+c
                        uci = chess.Move(sq, to_sq, promotion=promo).uci()
                    else:
                        uci = f"null_promo_b_{sq}_{dc}_{promo}"
                else:
                    uci = f"null_promo_{sq}_{dc}_{promo}"
                move2id[uci] = idx
                id2move[idx] = uci
                idx += 1

    print(f"Vocab size: {len(move2id)}")  # phải ra 4672
    return move2id, id2move

if __name__ == "__main__":
    move2id, id2move = build_alphazero_4672()
    workdir = "data/working"
    os.makedirs(workdir, exist_ok=True)
    with open(os.path.join(workdir, "move_vocab_4672.json"), "w") as f:
        json.dump(move2id, f)