# Chess Reinfor### Prerequisites

1. **Python 3.7+** installed
2. **Kaggle API** setup (for automatic data download):
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Create a new API token (downloads `kaggle.json`)
   - Place `kaggle.json` in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Alternatively, set environment variables:
     ```
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```
   - Note: The dataset used is `raw-chess-games-pgn`. If this name is incorrect, check the exact name on Kaggle and update `src/download_data.py`.g Project

This project implements a reinforcement learning model for predicting chess moves based on FEN positions. It processes chess game data from PGN files, builds a vocabulary of possible moves, trains a neural network policy model, and allows for move prediction.

## Features

- **Automatic data download** from Kaggle
- Extract moves from PGN files and convert to FEN-move pairs
- Clean and split dataset into train/validation/test sets
- Build AlphaZero-style move vocabulary (4672 moves)
- Train a convolutional neural network for move prediction
- Predict top-k legal moves for a given FEN position

## Setup

### Prerequisites

1. **Python 3.7+** installed
2. **Kaggle API** setup (for automatic data download):
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Create a new API token (downloads `kaggle.json`)
   - Place `kaggle.json` in:
     - Linux/Mac: `~/.kaggle/kaggle.json`
     - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Alternatively, set environment variables:
     ```
     export KAGGLE_USERNAME=your_username
     export KAGGLE_KEY=your_api_key
     ```

### Installation

Clone or download the project, then install dependencies:
```
pip install -r requirements.txt
```

## Usage

### Automatic Run (Recommended)

Run the entire pipeline:
```
python main.py
```
This will:
1. Check and download data from Kaggle if needed
2. Install dependencies
3. Process data
4. Train the model
5. Run a prediction example

### Manual Steps

1. **Download Data** (optional, if not using main.py):
   ```
   python src/download_data.py
   ```
   This downloads the dataset `raw-chess-games-pgn` from Kaggle to `data/input/`.

2. **Process Data**:
   ```
   python src/data_processing.py
   ```
   Extracts moves from PGN to `data/working/move_dataset.jsonl`.

3. **Clean and Split Data**:
   ```
   python src/clean_split_data.py
   ```
   Creates train/val/test splits in `data/working/`.

4. **Build Vocabulary**:
   ```
   python src/vocabulary.py
   ```
   Generates `data/working/move_vocab_4672.json`.

5. **Train Model**:
   ```
   python src/train.py
   ```
   Trains and saves model to `models/policy_model.pt`.

6. **Predict Moves**:
   ```
   python src/predict.py
   ```
   Runs prediction example.

## Project Structure

- `src/`: Source code
  - `download_data.py`: Download data from Kaggle
  - `data_processing.py`: Extract moves from PGN
  - `clean_split_data.py`: Clean and split dataset
  - `vocabulary.py`: Build move vocabulary
  - `dataset.py`: PyTorch dataset class
  - `models.py`: Neural network models
  - `train.py`: Training script
  - `predict.py`: Prediction script
- `data/`: Data files
  - `input/`: Raw PGN files (downloaded from Kaggle)
  - `working/`: Processed datasets and vocab
- `models/`: Saved model checkpoints
- `requirements.txt`: Python dependencies
- `README.md`: This file
- `main.py`: Orchestration script

## Notes

- The dataset used is `raw-chess-games-pgn` from Kaggle
- Adjust file paths in scripts if your directory structure differs
- Training may take time depending on your hardware
- The model uses a simplified architecture; for better performance, consider deeper networks or more data
- Ensure you have sufficient RAM/CPU/GPU for training on large datasets