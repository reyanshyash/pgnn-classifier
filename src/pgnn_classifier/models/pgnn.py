from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .cnn import LightCurveEncoder


def _logit(value: float) -> float:
    return math.log(value / (1.0 - value))


@dataclass(frozen=True)
class PGNNConfig:
    scalar_count: int = 13
    input_channels: int = 3
    conv_channels: tuple[int, ...] = (16, 32, 48)
    kernel_size: int = 5
    lightcurve_latent: int = 64
    scalar_hidden: int = 64
    scalar_latent: int = 64
    fusion_hidden: int = 128
    fusion_latent: int = 64
    dropout: float = 0.15
    share_view_encoder: bool = True
    num_subclasses: int = 4

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["conv_channels"] = list(self.conv_channels)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "PGNNConfig":
        payload = dict(payload)
        payload["conv_channels"] = tuple(payload.get("conv_channels", (16, 32, 48)))
        return cls(**payload)


class ConstrainedPhysicalHead(nn.Module):
    """Predict depth, k=Rp/R*, b, and q=T14/P with safe transforms."""

    def __init__(self, input_size: int, epsilon: float = 1e-5) -> None:
        super().__init__()
        self.linear = nn.Linear(input_size, 4)
        self.epsilon = epsilon
        nn.init.zeros_(self.linear.weight)
        with torch.no_grad():
            self.linear.bias.copy_(
                torch.tensor(
                    [
                        _logit(0.004),
                        _logit(0.05),
                        _logit(0.45),
                        _logit(0.06),
                    ],
                    dtype=self.linear.bias.dtype,
                )
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raw = self.linear(inputs)
        depth_fraction = self.epsilon + 0.5 * torch.sigmoid(raw[:, 0])
        radius_ratio = self.epsilon + torch.sigmoid(raw[:, 1])
        impact_parameter = (1.0 + radius_ratio - self.epsilon) * torch.sigmoid(raw[:, 2])
        duration_fraction = 0.5 * torch.sigmoid(raw[:, 3])
        return torch.stack(
            [depth_fraction, radius_ratio, impact_parameter, duration_fraction], dim=-1
        )


class LightweightPGNN(nn.Module):
    """A compact TCE classifier with differentiable physical outputs."""

    def __init__(self, config: PGNNConfig) -> None:
        super().__init__()
        self.config = config
        encoder_args = dict(
            input_channels=config.input_channels,
            channels=config.conv_channels,
            kernel_size=config.kernel_size,
            latent_size=config.lightcurve_latent,
            dropout=config.dropout,
        )
        self.local_encoder = LightCurveEncoder(**encoder_args)
        self.global_encoder = self.local_encoder if config.share_view_encoder else LightCurveEncoder(**encoder_args)
        scalar_input_size = config.scalar_count * 4
        self.scalar_encoder = nn.Sequential(
            nn.Linear(scalar_input_size, config.scalar_hidden),
            nn.LayerNorm(config.scalar_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.scalar_hidden, config.scalar_latent),
            nn.LayerNorm(config.scalar_latent),
            nn.GELU(),
        )
        fusion_input = config.lightcurve_latent * 2 + config.scalar_latent
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, config.fusion_hidden),
            nn.LayerNorm(config.fusion_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_hidden, config.fusion_latent),
            nn.LayerNorm(config.fusion_latent),
            nn.GELU(),
        )
        self.classifier = nn.Linear(config.fusion_latent, 1)
        self.subclass_head = (
            nn.Linear(config.fusion_latent, config.num_subclasses) if config.num_subclasses > 0 else None
        )
        self.physical_head = ConstrainedPhysicalHead(config.fusion_latent)

    def forward(
        self,
        local_view: torch.Tensor,
        global_view: torch.Tensor,
        scalar_input: torch.Tensor,
    ) -> dict[str, torch.Tensor | None]:
        local_latent = self.local_encoder(local_view)
        global_latent = self.global_encoder(global_view)
        scalar_latent = self.scalar_encoder(scalar_input)
        fused = self.fusion(torch.cat([local_latent, global_latent, scalar_latent], dim=-1))
        return {
            "logit": self.classifier(fused).squeeze(-1),
            "subclass_logits": self.subclass_head(fused) if self.subclass_head is not None else None,
            "physical": self.physical_head(fused),
            "embedding": fused,
        }

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
