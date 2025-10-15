# Deployment & Handover Guide

This document captures everything a teammate needs to reproduce, extend, or deploy
the chess reinforcement learning project after handover.

---

## 1. Prerequisites

| Requirement | Notes |
|-------------|-------|
| Python 3.9+ | Conda environment `rel` already exists in our workspace; create a fresh one if needed: `conda create -n rel python=3.10` |
| PyTorch     | Install CPU-only or CUDA build inside `rel`. For CUDA 12.1: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121` |
| Kaggle API  | Place `kaggle.json` under `~/.kaggle/` (Linux/macOS) or `C:\Users\<user>\.kaggle\` (Windows). |
| Git LFS     | Only needed if you plan to store large checkpoints in Git. |

Install Python deps:

```bash
conda activate rel
pip install -r requirements.txt
```

We use `kagglehub` for data download, `torch`/`torchvision` for training/inference, and `python-chess`
for PGN/FEN handling. Optional extras (matplotlib, pandas) power the control notebook visualisations.

---

## 2. Repository Layout Recap

```
.
├── main.py                  # CLI orchestrator
├── control.ipynb            # Notebook control book (toggle stages, plot metrics)
├── README.md                # Project overview & CLI docs
├── DEPLOY.md                # This handover note
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── download_data.py     # Kaggle download/extract
│   ├── data_processing.py   # PGN → JSONL conversion
│   ├── clean_split_data.py  # Deduplication + split helpers
│   ├── vocabulary.py        # AlphaZero move vocab builder
│   ├── dataset.py           # Torch dataset + collate fn
│   ├── features.py          # FEN encoders, legality masks
│   ├── models.py            # Policy & policy/value nets
│   ├── train.py             # Supervised imitation training
│   ├── rl_training.py       # Self-play actor-critic loop
│   ├── predict.py           # CLI inference util
│   └── play.py              # Interactive terminal game
├── data/
│   ├── input/               # Raw PGN files from Kaggle
│   └── working/             # Processed JSONL splits + vocab
└── models/
    ├── policy_model.pt      # Supervised checkpoint (if trained)
    └── rl/                  # RL checkpoints (`policy_value_final.pt`, etc.)
```

---

## 3. Minimal Quick Start

1. **Download data**  
   ```bash
   python main.py download --dataset ironicninja/raw-chess-games-pgn
   ```
2. **Process PGNs** (limit for prototyping)  
   ```bash
   python main.py process --limit 5000
   python main.py clean-split
   python main.py vocab
   ```
3. **Supervised training**  
   ```bash
   python main.py pretrain --epochs 30 --batch-size 256 --num-workers 0
   ```
4. **RL fine-tuning**  
   ```bash
   python main.py rl --iterations 20 --games-per-iter 8 --batch-size 128
   ```
5. **Interactive evaluation**  
   ```bash
   python main.py predict --fen "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
   python main.py play --checkpoint models/policy_model.pt --color white
   ```

All CLI commands accept `--help` to list tunable options. The README expands on configuration knobs
(entropy, temperature, resume, etc.).

---

## 4. Control Notebook Workflow

`control.ipynb` provides a UI-like experience:

1. Toggle `RUN_*` flags (download/process/pretrain/rl/etc.).
2. Adjust `PIPELINE_LIMIT`, epochs, RL configs.
3. Execute cells sequentially—the notebook captures metrics, displays RL logs, plots (if pandas/matplotlib installed), and lists checkpoints.

> **Memory tip (Windows)**: set `num_workers=0` in the notebook to avoid spawning additional DataLoader processes that duplicate the dataset in RAM.

---

## 5. Handling CUDA & Dependencies

- After activating `rel`, verify GPU visibility:

  ```bash
  python - <<'PY'
  import torch
  print("PyTorch:", torch.__version__)
  print("CUDA available:", torch.cuda.is_available())
  if torch.cuda.is_available():
      print("GPU:", torch.cuda.get_device_name(0))
  PY
  ```

- If `CUDA available` is `False`, reinstall PyTorch with a CUDA wheel for your driver version.
- RL training automatically uses GPU if available; otherwise it falls back to CPU.

---

## 6. Checkpoints & Artefacts

| Stage         | Default Output                        | Notes |
|---------------|----------------------------------------|-------|
| Supervised    | `models/policy_model.pt`               | Contains `model_state`, `optimizer_state`, `vocab`. |
| Reinforcement | `models/rl/policy_value_iter_XXX.pt`   | Iterative checkpoints (policy+value). Final: `policy_value_final.pt`. |
| Vocabulary    | `data/working/move_vocab_4672.json`    | JSON map `uci → id`. |
| Processed data| `data/working/{train,val,test}.jsonl`  | Each line: `{"fen","move","side_to_move"}`. |

To share results, compress checkpoints (`zip models -r models.zip`) or upload to cloud storage—Git should not version large `.pt` files by default.

---

## 7. Known Issues & Tips

- **MemoryError on Windows loaders**: use `--num-workers 0` or reduce `PIPELINE_LIMIT`.
- **Checkpoint mismatch warnings during RL**: the code now loads only compatible tensors from supervised training, so mismatched shapes are skipped (expected when architectures differ).
- **Timer / performance**: RL steps are intentionally small (games-per-iter, iterations). Scale cautiously to avoid overnight runs.
- **Safety warning from PyTorch (`weights_only`)**: We control our checkpoints, but consider using `torch.load(..., weights_only=True)` for external files.
- **Unicode output on Windows console**: run `chcp 65001` before viewing README or CLI logs to avoid mojibake.

---

## 8. Deployment Targets

- **Local dev**: run CLI/Notebook on any workstation with Python 3.9+ and optionally CUDA.
- **GPU workstation**: recommended for RL with larger iterations. Ensure CUDA drivers match PyTorch build.
- **Cloud/Jupyter**: upload repo, install requirements, run `control.ipynb`.
- **Production inference**: wrap `src.predict.predict_move` into an API or service; the module expects a loaded checkpoint and FEN strings.

---

## 9. Maintenance Checklist

- [ ] Update Kaggle dataset slug if the source changes.
- [ ] Periodically refresh requirements (PyTorch, python-chess, kagglehub).
- [ ] Document new experiments/results in a `reports/` folder (optional).
- [ ] Consider storing best checkpoints in a dedicated storage bucket.
- [ ] Keep README/DEPLOY updated whenever the CLI or pipeline changes.

---

## 10. Handover Contacts

- Original developer: _fill in name/contact here_
- Git repository: _provide remote URL or internal location_
- Issue tracker / task board: _link if applicable_

Feel free to reach out if questions arise during adoption. Happy hacking! 🧠♟️
