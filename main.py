#!/usr/bin/env python3
"""
Main script to run the chess RL project.
This script orchestrates data download, processing, training, and prediction.

Note: This script assumes you are running in the 'rel' conda environment.
If not, activate it first: conda activate rel
"""

import os
import subprocess
import sys

def run_command(cmd):
    """Run a shell command and print output."""
    # Prefix with conda run to ensure using 'rel' environment
    full_cmd = f"conda run -n rel {cmd}"
    print(f"Running: {full_cmd}")
    result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    print(result.stdout)

def ask_run_step(step_name):
    """Ask user if they want to run a step."""
    response = input(f"Do you want to run {step_name}? (y/n) [y]: ").strip().lower()
    return response in ('', 'y', 'yes')

def main():
    print("Chess RL Project Runner")

    # Step 0: Download data from Kaggle (optional, uncomment if needed)
    print("Step 0: Checking for data...")
    data_file = "data/input/lichess_db_standard_rated_2014-08.pgn"
    if not os.path.exists(data_file):
        if ask_run_step("data download"):
            print(f"Data file {data_file} not found. Attempting to download from Kaggle...")
            run_command("python src/download_data.py")
        else:
            print("Skipping data download.")
    else:
        print(f"Data file {data_file} already exists. Skipping download.")

    # Step 1: Install dependencies
    if ask_run_step("Step 1: Install dependencies"):
        print("Step 1: Installing dependencies...")
        run_command("pip install -r requirements.txt")
    else:
        print("Skipping Step 1.")

    # Step 2: Data processing (assuming PGN file is in data/input/)
    if ask_run_step("Step 2: Data processing"):
        print("Step 2: Processing data...")
        run_command("python src/data_processing.py")
    else:
        print("Skipping Step 2.")

    # Step 3: Clean and split data
    if ask_run_step("Step 3: Clean and split data"):
        print("Step 3: Cleaning and splitting data...")
        run_command("python src/clean_split_data.py")
    else:
        print("Skipping Step 3.")

    # Step 4: Build vocabulary
    if ask_run_step("Step 4: Build vocabulary"):
        print("Step 4: Building vocabulary...")
        run_command("python src/vocabulary.py")
    else:
        print("Skipping Step 4.")

    # Step 5: Train model
    if ask_run_step("Step 5: Train model"):
        print("Step 5: Training model...")
        run_command("python src/train.py")
    else:
        print("Skipping Step 5.")

    # Step 6: Predict (example)
    if ask_run_step("Step 6: Run prediction example"):
        print("Step 6: Running prediction example...")
        run_command("python src/predict.py")
    else:
        print("Skipping Step 6.")

    print("Project completed successfully!")

if __name__ == "__main__":
    main()