#!/usr/bin/env python3
"""
Command-line interface for the chess reinforcement learning project.

This entrypoint exposes task-focused subcommands for data preparation,
supervised pre-training, reinforcement learning fine-tuning, and inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.download_data import download_chess_data
from src import data_processing
from src import clean_split_data
from src import vocabulary
from src.train import run_supervised_training
from src.rl_training import (
    run_rl_training,
    SelfPlayConfig,
    OptimizationConfig,
)
from src.predict import predict_move, load_checkpoint
from src.play import build_play_parser

import torch


def cmd_download(args: argparse.Namespace) -> None:
    download_chess_data(dataset_name=args.dataset, download_path=str(args.output))


def cmd_process(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data_processing.extract_moves_from_pgn(str(args.input), str(args.output), limit=args.limit)


def cmd_clean_split(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True)
    clean_file = args.output / "move_dataset_clean.jsonl"
    clean_split_data.clean_dataset(str(args.input), str(clean_file))
    clean_split_data.split_jsonl_dataset(
        str(clean_file),
        str(args.output),
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )


def cmd_vocab(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    move2id, _ = vocabulary.build_alphazero_4672()
    with args.output.open("w", encoding="utf-8") as file:
        import json

        json.dump(move2id, file)
    print(f"Wrote vocabulary to {args.output}")


def cmd_pretrain(args: argparse.Namespace) -> None:
    run_supervised_training(
        workdir=args.workdir,
        vocab_path=args.vocab,
        checkpoint_path=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        num_workers=args.num_workers,
        resume=args.resume,
    )


def cmd_rl(args: argparse.Namespace) -> None:
    self_play_cfg = SelfPlayConfig(
        games_per_iteration=args.games_per_iter,
        max_moves=args.max_moves,
        temperature=args.temperature,
        random_move_prob=args.random_move_prob,
        gamma=args.gamma,
    )
    optim_cfg = OptimizationConfig(
        num_iterations=args.iterations,
        policy_epochs=args.policy_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        save_interval=args.save_interval,
    )
    run_rl_training(
        vocab_path=args.vocab,
        supervised_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        resume=args.resume,
        self_play_cfg=self_play_cfg,
        optim_cfg=optim_cfg,
    )


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(args.checkpoint, device)
    move2id = checkpoint["vocab"]

    from src.models import PolicyNet

    model = PolicyNet(len(move2id))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    moves = predict_move(args.fen, model, move2id, device, topk=args.topk)
    print(f"Top-{len(moves)} moves for given position: {moves}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    if not args.skip_download:
        cmd_download(
            argparse.Namespace(dataset=args.dataset, output=args.data_dir),
        )

    if not args.skip_process:
        cmd_process(
            argparse.Namespace(
                input=args.raw_pgn,
                output=args.workdir / "move_dataset.jsonl",
                limit=args.limit,
            )
        )

    if not args.skip_clean:
        cmd_clean_split(
            argparse.Namespace(
                input=args.workdir / "move_dataset.jsonl",
                output=args.workdir,
                seed=args.seed,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
        )

    if not args.skip_vocab:
        cmd_vocab(
            argparse.Namespace(output=args.vocab),
        )

    if args.run_pretrain:
        cmd_pretrain(
            argparse.Namespace(
                workdir=args.workdir,
                vocab=args.vocab,
                checkpoint=args.checkpoint,
                epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                resume=args.resume_pretrain,
            )
        )

    if args.run_rl:
        cmd_rl(
            argparse.Namespace(
                vocab=args.vocab,
                checkpoint=args.checkpoint,
                output_dir=args.rl_output,
                resume=args.resume_rl,
                games_per_iter=args.games_per_iter,
                max_moves=args.max_moves,
                temperature=args.temperature,
                random_move_prob=args.random_move_prob,
                gamma=args.gamma,
                iterations=args.iterations,
                policy_epochs=args.policy_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                entropy_coef=args.entropy_coef,
                value_coef=args.value_coef,
                save_interval=args.save_interval,
            )
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chess reinforcement learning project CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Download command.
    download_parser = subparsers.add_parser("download", help="Download PGN data from Kaggle.")
    download_parser.add_argument("--dataset", type=str, default="ironicninja/raw-chess-games-pgn")
    download_parser.add_argument("--output", type=Path, default=Path("data/input"))
    download_parser.set_defaults(func=cmd_download)

    # Process command.
    process_parser = subparsers.add_parser("process", help="Extract FEN-move pairs from PGN.")
    process_parser.add_argument("--input", type=Path, default=Path("data/input/lichess_db_standard_rated_2014-08.pgn"))
    process_parser.add_argument("--output", type=Path, default=Path("data/working/move_dataset.jsonl"))
    process_parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of games processed.")
    process_parser.set_defaults(func=cmd_process)

    # Clean & split command.
    clean_parser = subparsers.add_parser("clean-split", help="Deduplicate dataset and create splits.")
    clean_parser.add_argument("--input", type=Path, default=Path("data/working/move_dataset.jsonl"))
    clean_parser.add_argument("--output", type=Path, default=Path("data/working"))
    clean_parser.add_argument("--seed", type=int, default=42)
    clean_parser.add_argument("--train-ratio", type=float, default=0.8)
    clean_parser.add_argument("--val-ratio", type=float, default=0.1)
    clean_parser.set_defaults(func=cmd_clean_split)

    # Vocabulary command.
    vocab_parser = subparsers.add_parser("vocab", help="Build AlphaZero-style move vocabulary.")
    vocab_parser.add_argument("--output", type=Path, default=Path("data/working/move_vocab_4672.json"))
    vocab_parser.set_defaults(func=cmd_vocab)

    # Supervised training.
    pretrain_parser = subparsers.add_parser("pretrain", help="Run supervised imitation learning.")
    pretrain_parser.add_argument("--workdir", type=Path, default=Path("data/working"))
    pretrain_parser.add_argument("--vocab", type=Path, default=Path("data/working/move_vocab_4672.json"))
    pretrain_parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    pretrain_parser.add_argument("--epochs", type=int, default=30)
    pretrain_parser.add_argument("--batch-size", type=int, default=256)
    pretrain_parser.add_argument("--lr", type=float, default=1e-3)
    pretrain_parser.add_argument("--num-workers", type=int, default=2)
    pretrain_parser.add_argument("--resume", type=Path, default=None)
    pretrain_parser.set_defaults(func=cmd_pretrain)

    # Reinforcement learning.
    rl_parser = subparsers.add_parser("rl", help="Fine-tune the policy with self-play reinforcement learning.")
    rl_parser.add_argument("--vocab", type=Path, default=Path("data/working/move_vocab_4672.json"))
    rl_parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    rl_parser.add_argument("--output-dir", type=Path, default=Path("models/rl"))
    rl_parser.add_argument("--resume", type=Path, default=None)
    rl_parser.add_argument("--games-per-iter", type=int, default=32)
    rl_parser.add_argument("--max-moves", type=int, default=160)
    rl_parser.add_argument("--temperature", type=float, default=1.0)
    rl_parser.add_argument("--random-move-prob", type=float, default=0.05)
    rl_parser.add_argument("--gamma", type=float, default=0.99)
    rl_parser.add_argument("--iterations", type=int, default=50)
    rl_parser.add_argument("--policy-epochs", type=int, default=4)
    rl_parser.add_argument("--batch-size", type=int, default=128)
    rl_parser.add_argument("--lr", type=float, default=5e-4)
    rl_parser.add_argument("--entropy-coef", type=float, default=0.01)
    rl_parser.add_argument("--value-coef", type=float, default=0.5)
    rl_parser.add_argument("--save-interval", type=int, default=5)
    rl_parser.set_defaults(func=cmd_rl)

    # Prediction.
    predict_parser = subparsers.add_parser("predict", help="Run inference on a FEN position.")
    predict_parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    predict_parser.add_argument("--fen", type=str, default="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    predict_parser.add_argument("--topk", type=int, default=5)
    predict_parser.set_defaults(func=cmd_predict)

    # Interactive play.
    build_play_parser(subparsers)

    # Pipeline command.
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the end-to-end workflow.")
    pipeline_parser.add_argument("--dataset", type=str, default="ironicninja/raw-chess-games-pgn")
    pipeline_parser.add_argument("--data-dir", type=Path, default=Path("data/input"))
    pipeline_parser.add_argument("--raw-pgn", type=Path, default=Path("data/input/lichess_db_standard_rated_2014-08.pgn"))
    pipeline_parser.add_argument("--workdir", type=Path, default=Path("data/working"))
    pipeline_parser.add_argument("--vocab", type=Path, default=Path("data/working/move_vocab_4672.json"))
    pipeline_parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"))
    pipeline_parser.add_argument("--rl-output", type=Path, default=Path("models/rl"))
    pipeline_parser.add_argument("--limit", type=int, default=None)
    pipeline_parser.add_argument("--seed", type=int, default=42)
    pipeline_parser.add_argument("--train-ratio", type=float, default=0.8)
    pipeline_parser.add_argument("--val-ratio", type=float, default=0.1)
    pipeline_parser.add_argument("--batch-size", type=int, default=256)
    pipeline_parser.add_argument("--lr", type=float, default=1e-3)
    pipeline_parser.add_argument("--num-workers", type=int, default=2)
    pipeline_parser.add_argument("--pretrain-epochs", type=int, default=30)
    pipeline_parser.add_argument("--resume-pretrain", type=Path, default=None)
    pipeline_parser.add_argument("--resume-rl", type=Path, default=None)
    pipeline_parser.add_argument("--games-per-iter", type=int, default=32)
    pipeline_parser.add_argument("--max-moves", type=int, default=160)
    pipeline_parser.add_argument("--temperature", type=float, default=1.0)
    pipeline_parser.add_argument("--random-move-prob", type=float, default=0.05)
    pipeline_parser.add_argument("--gamma", type=float, default=0.99)
    pipeline_parser.add_argument("--iterations", type=int, default=10)
    pipeline_parser.add_argument("--policy-epochs", type=int, default=4)
    pipeline_parser.add_argument("--entropy-coef", type=float, default=0.01)
    pipeline_parser.add_argument("--value-coef", type=float, default=0.5)
    pipeline_parser.add_argument("--save-interval", type=int, default=5)
    pipeline_parser.add_argument("--skip-download", action="store_true")
    pipeline_parser.add_argument("--skip-process", action="store_true")
    pipeline_parser.add_argument("--skip-clean", action="store_true")
    pipeline_parser.add_argument("--skip-vocab", action="store_true")
    pipeline_parser.add_argument("--run-pretrain", action="store_true", help="Run supervised pre-training as part of the pipeline.")
    pipeline_parser.add_argument("--run-rl", action="store_true", help="Run reinforcement learning after pre-training.")
    pipeline_parser.set_defaults(func=cmd_pipeline)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
