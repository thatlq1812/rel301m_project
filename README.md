# Chess Reinforcement Learning Toolkit

This repository contains an end-to-end workflow for training chess move prediction models.
The pipeline covers data acquisition, PGN preprocessing, supervised imitation learning,
self-play reinforcement learning, and interactive inference with the trained policy.

## Key Features
- **Automated data pipeline**: download PGNs from Kaggle, convert to JSONL `(fen, move)` pairs,
  deduplicate, and generate train/validation/test splits with a few CLI commands.
- **Supervised pre-training**: train a policy network on historical games using masked cross-entropy,
  optional mixed precision, and configurable dataloader settings.
- **Self-play RL**: fine-tune the policy/value network with an actor-critic loop that supports temperature
  sampling, entropy regularisation, and checkpointing.
- **Modular CLI**: every stage is exposed via `main.py` so you can run individual steps or the full pipeline.
- **Interactive play**: battle the trained model directly from the terminal to evaluate qualitative strength.

## Requirements

- Python 3.9 or newer
- GPU optional (PyTorch will fall back to CPU if CUDA is not available)
- Kaggle API credentials (`kagglehub` looks for `~/.kaggle/kaggle.json` on Linux/macOS or
  `C:\Users\<user>\.kaggle\kaggle.json` on Windows)

Install dependencies inside your environment:

```bash
pip install -r requirements.txt
```

> Tip: if you want GPU acceleration, install a CUDA-enabled wheel for PyTorch in your environment (e.g.,
> `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`).

## Command Line Interface

Use `python main.py --help` for the top-level menu. Key subcommands:

| Command | Description |
|---------|-------------|
| `download` | Fetch and extract the Kaggle dataset (PGN archives). |
| `process` | Convert PGNs to JSONL with `fen`, `move`, and `side_to_move`. |
| `clean-split` | Deduplicate records, create train/val/test splits. |
| `vocab` | Build the AlphaZero-style move vocabulary (4672 moves). |
| `pretrain` | Run supervised imitation learning. |
| `rl` | Launch self-play reinforcement learning fine-tuning. |
| `predict` | Print top-k legal moves for a given FEN. |
| `play` | Play an interactive terminal game against the policy. |
| `pipeline` | Orchestrate the workflow end-to-end. |

Example usage:

```bash
# Download dataset once
python main.py download --dataset ironicninja/raw-chess-games-pgn

# Process PGNs and build vocabulary
python main.py process --limit 5000
python main.py vocab --output data/working/move_vocab_4672.json

# Supervised training
python main.py pretrain --epochs 30 --batch-size 256

# Reinforcement learning fine-tuning
python main.py rl --iterations 20 --games-per-iter 16

# Predict from a checkpoint
python main.py predict --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

# Play against the model
python main.py play --checkpoint models/policy_model.pt --color black
```

### End-to-End Pipeline

Run the full workflow with one command (customise limits and epochs as needed):

```bash
python main.py pipeline \
  --dataset ironicninja/raw-chess-games-pgn \
  --limit 5000 \
  --run-pretrain \
  --run-rl
```

Skip completed stages with flags such as `--skip-download`, `--skip-process`, etc.

## Configuration Highlights

### Supervised (`pretrain`)
- `--epochs`, `--batch-size`, `--lr`, `--num-workers` control training time/resource use.
- Use `--resume` to continue from a previous checkpoint.
- Dataloaders enable pinned memory and AMP automatically if CUDA is detected.

### Reinforcement Learning (`rl`)
- Self-play control knobs: `--games-per-iter`, `--max-moves`, `--temperature`, `--random-move-prob`, `--gamma`.
- Optimisation knobs: `--iterations`, `--policy-epochs`, `--batch-size`, `--lr`, `--entropy-coef`, `--value-coef`.
- Checkpoints are written to `models/rl/` (final policy saved as `policy_value_final.pt`).

### Interactive Play (`play`)
- Accepts SAN (`e4`) or UCI (`e2e4`) moves, `moves` prints legal options, `resign` ends the game.
- Use `--deterministic` to force the agent to play greedily.
- Point to an RL checkpoint (`models/rl/policy_value_final.pt`) for the strongest opponent.

## Project Structure

```
.
├── main.py                  # CLI entry point
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── clean_split_data.py  # Dedup + split helpers
│   ├── data_processing.py   # PGN → JSONL conversion
│   ├── dataset.py           # Torch dataset + collate fn
│   ├── download_data.py     # Kaggle download/extract
│   ├── features.py          # FEN/board encoders, masks
│   ├── models.py            # Policy / policy-value networks
│   ├── play.py              # Interactive gameplay CLI
│   ├── predict.py           # FEN inference script
│   ├── rl_training.py       # Self-play actor-critic loop
│   └── train.py             # Supervised imitation trainer
├── data/                    # Input and processed data
└── models/                  # Saved checkpoints
```

## Tips & Troubleshooting
- **Dataset size**: limit PGNs during prototyping (e.g., `--limit 2000`) to save memory.
- **Windows DataLoader**: set `--num-workers 0` if you hit `MemoryError` with large datasets.
- **CUDA**: ensure you installed a CUDA-enabled PyTorch wheel; otherwise training runs on CPU.
- **Safety**: checkpoints generated here are safe to load; be cautious with untrusted `.pt` files (see PyTorch docs on `torch.load(weights_only=True)`).
- **Performance**: tune RL parameters (entropy coefficient, temperature) based on your hardware and desired exploration.

## License

MIT. Adjust attribution if you reuse this work in course submissions or derivative projects.
