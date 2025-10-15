"""Neural network architectures used throughout the project."""

from __future__ import annotations

import torch
import torch.nn as nn


class ChessConvBackbone(nn.Module):
    """Shared convolutional trunk used by policy and value networks."""

    def __init__(self, in_channels: int = 18, hidden_channels: int = 128, blocks: int = 3):
        super().__init__()
        layers = []
        channels = in_channels
        for _ in range(blocks):
            layers.extend(
                [
                    nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            channels = hidden_channels
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trunk(x)


class PolicyNet(nn.Module):
    """Simple convolutional policy network for supervised imitation learning."""

    def __init__(self, vocab_size: int, hidden_channels: int = 128):
        super().__init__()
        self.backbone = ChessConvBackbone(hidden_channels=hidden_channels)
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels * 8 * 8, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.policy_head(features)


class PolicyValueNet(nn.Module):
    """
    Joint policy/value network used during reinforcement learning. It shares the
    convolutional trunk and branches into separate heads for policy logits and
    scalar state value estimation.
    """

    def __init__(self, vocab_size: int, hidden_channels: int = 160):
        super().__init__()
        self.backbone = ChessConvBackbone(hidden_channels=hidden_channels, blocks=6)

        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(hidden_channels * 8 * 8, vocab_size),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)
        policy_logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return policy_logits, value


class ResBlock(nn.Module):
    """Standard residual block used for deeper architectures."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class ChessResNet(nn.Module):
    """Residual network variant retaining compatibility with earlier experiments."""

    def __init__(self, in_channels: int = 18, channels: int = 160, blocks: int = 12, vocab_size: int = 100000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, vocab_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(self.stem(x))
        return self.policy_head(features)
