from __future__ import annotations

import math

import torch
from torch import nn


def _group_count(channels: int) -> int:
    return max(1, math.gcd(channels, 8))


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class LightCurveEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        channels: tuple[int, ...],
        kernel_size: int,
        latent_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        current = input_channels
        for output in channels:
            blocks.append(ConvBlock(current, output, kernel_size, dropout))
            current = output
        self.features = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(current * 2, latent_size),
            nn.LayerNorm(latent_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        encoded = self.features(inputs)
        average = encoded.mean(dim=-1)
        maximum = encoded.amax(dim=-1)
        return self.projection(torch.cat([average, maximum], dim=-1))
