"""
Reinforcement learning pipeline that fine-tunes the chess policy/value network
through self-play using an actor-critic style update.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm.auto import tqdm

import chess

from .features import board_to_planes, legal_move_mask, mask_logits
from .models import PolicyValueNet


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 32
    max_moves: int = 160
    temperature: float = 1.0
    random_move_prob: float = 0.05
    gamma: float = 0.99
    resign_threshold: float | None = None  # Set to e.g. -0.95 to enable resignations.


@dataclass
class OptimizationConfig:
    num_iterations: int = 50
    policy_epochs: int = 4
    batch_size: int = 128
    lr: float = 5e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    clip_grad_norm: float | None = 1.0
    save_interval: int = 5


@dataclass
class Transition:
    state: np.ndarray
    mask: np.ndarray
    action: int
    return_value: float


@dataclass
class GameSummary:
    result: float  # +1 white win, -1 black win, 0 draw/unknown.
    num_moves: int
    truncated: bool


class SelfPlayRunner:
    """Generates self-play rollouts from the current policy/value network."""

    def __init__(self, model: PolicyValueNet, move2id: Dict[str, int], device: torch.device, config: SelfPlayConfig):
        self.model = model
        self.move2id = move2id
        self.device = device
        self.config = config
        self.id2move = {idx: move for move, idx in move2id.items()}

    @torch.no_grad()
    def play_game(self) -> Tuple[List[Transition], GameSummary]:
        board = chess.Board()
        states: List[np.ndarray] = []
        masks: List[np.ndarray] = []
        actions: List[int] = []
        players: List[int] = []

        for move_idx in range(self.config.max_moves):
            if board.is_game_over():
                break

            planes = board_to_planes(board)
            mask = legal_move_mask(board, self.move2id)
            legal_indices = np.nonzero(mask > 0)[0].tolist()

            if not legal_indices:
                break

            states.append(planes)
            masks.append(mask)
            players.append(1 if board.turn == chess.WHITE else -1)

            tensor_inputs = torch.from_numpy(planes).unsqueeze(0).float().to(self.device)
            tensor_mask = torch.from_numpy(mask).unsqueeze(0).to(self.device)

            with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                logits, value = self.model(tensor_inputs)
                masked_logits = mask_logits(logits, tensor_mask)

            # Temperature sampling encourages exploration.
            temperature = max(self.config.temperature, 1e-3)
            scaled_logits = (masked_logits.squeeze(0)) / temperature

            if random.random() < self.config.random_move_prob:
                action_idx = random.choice(legal_indices)
            else:
                distribution = torch.distributions.Categorical(logits=scaled_logits)
                action_idx = int(distribution.sample().item())
                if action_idx not in legal_indices:
                    action_idx = random.choice(legal_indices)

            move_uci = self.id2move.get(action_idx)
            move = chess.Move.from_uci(move_uci) if move_uci else None
            if move not in board.legal_moves:
                move = None
            if move is None:
                admissible_moves = [m for m in board.legal_moves if m.uci() in self.move2id]
                move = random.choice(admissible_moves) if admissible_moves else random.choice(list(board.legal_moves))
                action_idx = self.move2id.get(move.uci(), action_idx)

            actions.append(action_idx)
            board.push(move)

        truncated = not board.is_game_over()
        outcome = board.outcome(claim_draw=True)
        if outcome is None or truncated:
            result_value = 0.0
        else:
            if outcome.winner is None:
                result_value = 0.0
            else:
                result_value = 1.0 if outcome.winner == chess.WHITE else -1.0

        rewards = [result_value * player for player in players]
        returns = self._discount_returns(rewards, self.config.gamma)

        transitions = [
            Transition(state=s, mask=m, action=a, return_value=ret)
            for s, m, a, ret in zip(states, masks, actions, returns)
        ]

        summary = GameSummary(result=result_value, num_moves=len(actions), truncated=truncated)
        return transitions, summary

    def generate(self) -> Tuple[List[Transition], List[GameSummary]]:
        self.model.eval()
        transitions: List[Transition] = []
        summaries: List[GameSummary] = []

        for _ in tqdm(range(self.config.games_per_iteration), desc="self-play", leave=False):
            game_transitions, summary = self.play_game()
            transitions.extend(game_transitions)
            summaries.append(summary)
        return transitions, summaries

    @staticmethod
    def _discount_returns(rewards: Sequence[float], gamma: float) -> List[float]:
        discounted: List[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + gamma * running
            discounted.append(running)
        return list(reversed(discounted))


class RLTrainer:
    """Handles the actor-critic optimisation loop."""

    def __init__(
        self,
        model: PolicyValueNet,
        move2id: Dict[str, int],
        device: torch.device,
        self_play_cfg: SelfPlayConfig,
        optim_cfg: OptimizationConfig,
    ):
        self.model = model.to(device)
        self.move2id = move2id
        self.device = device
        self.self_play = SelfPlayRunner(self.model, move2id, device, self_play_cfg)
        self.optim_cfg = optim_cfg
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=optim_cfg.lr, weight_decay=1e-4)
        self.scaler = GradScaler(enabled=device.type == "cuda")

    def train(self, output_dir: Path, resume_checkpoint: Path | None = None):
        output_dir.mkdir(parents=True, exist_ok=True)

        if resume_checkpoint:
            checkpoint = torch.load(resume_checkpoint, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            print(f"Resumed RL training from {resume_checkpoint}")

        for iteration in range(1, self.optim_cfg.num_iterations + 1):
            transitions, summaries = self.self_play.generate()
            if not transitions:
                print("No transitions generated; skipping iteration.")
                continue

            batch = self._prepare_batch(transitions)
            metrics = self._update_policy(batch)
            summary_stats = self._aggregate_summaries(summaries)

            print(
                f"[Iter {iteration:03d}] "
                f"loss={metrics['loss']:.5f} "
                f"policy={metrics['policy_loss']:.5f} "
                f"value={metrics['value_loss']:.5f} "
                f"entropy={metrics['entropy']:.5f} "
                f"avg_return={metrics['avg_return']:.3f} "
                f"win_rate={summary_stats['win_rate']:.2%} "
                f"draw_rate={summary_stats['draw_rate']:.2%} "
                f"avg_length={summary_stats['avg_length']:.1f}"
            )

            if iteration % self.optim_cfg.save_interval == 0:
                self._save_checkpoint(output_dir / f"policy_value_iter_{iteration:03d}.pt")

        # Save final checkpoint.
        self._save_checkpoint(output_dir / "policy_value_final.pt")

    def _prepare_batch(self, transitions: Sequence[Transition]):
        states = torch.from_numpy(np.stack([t.state for t in transitions])).float().to(self.device)
        masks = torch.from_numpy(np.stack([t.mask for t in transitions])).float().to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device)
        returns = torch.tensor([t.return_value for t in transitions], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            _, values = self.model(states)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-6)

        return {
            "states": states,
            "masks": masks,
            "actions": actions,
            "returns": returns,
            "advantages": advantages,
        }

    def _update_policy(self, batch):
        cfg = self.optim_cfg
        states = batch["states"]
        masks = batch["masks"]
        actions = batch["actions"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        num_samples = states.size(0)
        indices = torch.arange(num_samples, device=self.device)

        total_loss = total_policy = total_value = total_entropy = 0.0
        total_batches = 0

        for epoch in range(cfg.policy_epochs):
            shuffled = indices[torch.randperm(num_samples, device=self.device)]
            for start in tqdm(range(0, num_samples, cfg.batch_size), leave=False, desc=f"rl-opt ep{epoch+1}"):
                end = start + cfg.batch_size
                batch_idx = shuffled[start:end]

                batch_states = states[batch_idx]
                batch_masks = masks[batch_idx]
                batch_actions = actions[batch_idx]
                batch_returns = returns[batch_idx]
                batch_advantages = advantages[batch_idx]

                with autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
                    logits, values = self.model(batch_states)
                    masked_logits = mask_logits(logits, batch_masks)
                    log_probs = torch.log_softmax(masked_logits, dim=-1)
                    probs = torch.softmax(masked_logits, dim=-1)

                    selected_log_probs = log_probs.gather(1, batch_actions.unsqueeze(-1)).squeeze(-1)
                    policy_loss = -(batch_advantages * selected_log_probs).mean()
                    value_loss = F.mse_loss(values, batch_returns)
                    entropy = -(probs * log_probs).sum(dim=-1).mean()

                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                self.scaler.scale(loss).backward()

                if cfg.clip_grad_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.clip_grad_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += float(loss.item())
                total_policy += float(policy_loss.item())
                total_value += float(value_loss.item())
                total_entropy += float(entropy.item())
                total_batches += 1

        mean_loss = total_loss / max(1, total_batches)
        return {
            "loss": mean_loss,
            "policy_loss": total_policy / max(1, total_batches),
            "value_loss": total_value / max(1, total_batches),
            "entropy": total_entropy / max(1, total_batches),
            "avg_return": float(returns.mean().item()),
        }

    def _aggregate_summaries(self, summaries: Iterable[GameSummary]):
        summaries = list(summaries)
        if not summaries:
            return {"win_rate": 0.0, "draw_rate": 0.0, "avg_length": 0.0}

        wins = sum(1 for s in summaries if s.result > 0)
        losses = sum(1 for s in summaries if s.result < 0)
        draws = len(summaries) - wins - losses
        avg_length = sum(s.num_moves for s in summaries) / len(summaries)
        return {
            "win_rate": wins / len(summaries),
            "draw_rate": draws / len(summaries),
            "avg_length": avg_length,
        }

    def _save_checkpoint(self, path: Path):
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "vocab": self.move2id,
            },
            path,
        )
        print(f"Saved checkpoint to {path}")


def parse_cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reinforcement learning fine-tuning via self-play.")
    parser.add_argument("--vocab", type=Path, default=Path("data/working/move_vocab_4672.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/policy_model.pt"), help="Supervised checkpoint used for initialisation.")
    parser.add_argument("--output-dir", type=Path, default=Path("models/rl"))
    parser.add_argument("--resume", type=Path, help="Optional RL checkpoint to resume from.")

    # Self-play overrides.
    parser.add_argument("--games-per-iter", type=int, help="Number of self-play games per iteration.")
    parser.add_argument("--max-moves", type=int, help="Maximum ply per game.")
    parser.add_argument("--temperature", type=float, help="Sampling temperature during self-play.")
    parser.add_argument("--random-move-prob", type=float, help="Epsilon-greedy random move probability.")
    parser.add_argument("--gamma", type=float, help="Discount factor for returns.")

    # Optimisation overrides.
    parser.add_argument("--iterations", type=int, help="Number of RL iterations.")
    parser.add_argument("--policy-epochs", type=int, help="Number of policy optimisation epochs per iteration.")
    parser.add_argument("--batch-size", type=int, help="Mini-batch size for RL updates.")
    parser.add_argument("--lr", type=float, help="Learning rate for RL optimiser.")
    parser.add_argument("--entropy-coef", type=float, help="Entropy regularisation coefficient.")
    parser.add_argument("--value-coef", type=float, help="Value loss coefficient.")
    parser.add_argument("--save-interval", type=int, help="Iterations between checkpoints.")
    return parser.parse_args()


def _load_supervised_backbone(model: PolicyValueNet, checkpoint, allow_mismatch: bool = True):
    """
    Load weights from the supervised policy checkpoint. Because the supervised
    PolicyNet may differ in architecture (e.g., channel width) from the RL
    PolicyValueNet, we selectively load parameters whose shapes match and leave
    the rest initialised as-is.
    """
    target_state = model.state_dict()
    pretrained = checkpoint.get("model_state", {})

    if not allow_mismatch:
        model.load_state_dict(pretrained, strict=False)
        return

    compatible = {}
    skipped = []
    for key, tensor in pretrained.items():
        if key in target_state and target_state[key].shape == tensor.shape:
            compatible[key] = tensor
        else:
            skipped.append(key)

    target_state.update(compatible)
    model.load_state_dict(target_state)

    if compatible:
        print(f"Loaded {len(compatible)} tensors from supervised checkpoint.")
    if skipped:
        print(f"Skipped {len(skipped)} tensors due to shape mismatch: first few -> {skipped[:5]}")


def run_rl_training(
    vocab_path: Path,
    supervised_checkpoint: Path,
    output_dir: Path,
    resume: Path | None,
    self_play_cfg: SelfPlayConfig,
    optim_cfg: OptimizationConfig,
) -> None:
    with vocab_path.open("r", encoding="utf-8") as file:
        move2id = json.load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PolicyValueNet(len(move2id))
    if supervised_checkpoint.exists():
        base_ckpt = torch.load(supervised_checkpoint, map_location=device, weights_only=False)
        _load_supervised_backbone(model, base_ckpt, allow_mismatch=True)
        print(f"Loaded supervised weights from {supervised_checkpoint}")

    trainer = RLTrainer(model, move2id, device, self_play_cfg, optim_cfg)
    trainer.train(output_dir, resume_checkpoint=resume)


def main():
    args = parse_cli()

    self_play_cfg = SelfPlayConfig()
    optim_cfg = OptimizationConfig()

    if args.games_per_iter is not None:
        self_play_cfg.games_per_iteration = args.games_per_iter
    if args.max_moves is not None:
        self_play_cfg.max_moves = args.max_moves
    if args.temperature is not None:
        self_play_cfg.temperature = args.temperature
    if args.random_move_prob is not None:
        self_play_cfg.random_move_prob = args.random_move_prob
    if args.gamma is not None:
        self_play_cfg.gamma = args.gamma

    if args.iterations is not None:
        optim_cfg.num_iterations = args.iterations
    if args.policy_epochs is not None:
        optim_cfg.policy_epochs = args.policy_epochs
    if args.batch_size is not None:
        optim_cfg.batch_size = args.batch_size
    if args.lr is not None:
        optim_cfg.lr = args.lr
    if args.entropy_coef is not None:
        optim_cfg.entropy_coef = args.entropy_coef
    if args.value_coef is not None:
        optim_cfg.value_coef = args.value_coef
    if args.save_interval is not None:
        optim_cfg.save_interval = args.save_interval

    run_rl_training(
        vocab_path=args.vocab,
        supervised_checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        resume=args.resume,
        self_play_cfg=self_play_cfg,
        optim_cfg=optim_cfg,
    )


if __name__ == "__main__":
    main()

