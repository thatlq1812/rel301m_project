import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(18, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.Conv2d(128,128,3,padding=1), nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128*8*8, 1024), nn.ReLU(),
            nn.Linear(1024, vocab_size)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch),
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): 
        return self.act(x + self.f(x))

class ChessResNet(nn.Module):
    def __init__(self, in_ch=18, ch=160, n=12, vocab=100000):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
        self.trunk = nn.Sequential(*[ResBlock(ch) for _ in range(n)])
        self.policy = nn.Sequential(
            nn.Conv2d(ch, 32, 1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32*8*8, vocab)
        )
    def forward(self, x):
        h = self.trunk(self.stem(x))
        return self.policy(h)