from __future__ import annotations

import math

import torch


def circular_duration_fraction(
    radius_ratio: torch.Tensor,
    scaled_semimajor_axis: torch.Tensor,
    impact_parameter: torch.Tensor,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Circular-orbit first-to-fourth-contact duration divided by period."""
    numerator = ((1.0 + radius_ratio).square() - impact_parameter.square()).clamp_min(epsilon)
    denominator = (scaled_semimajor_axis.square() - impact_parameter.square()).clamp_min(epsilon)
    argument = torch.sqrt((numerator / denominator).clamp(min=epsilon, max=1.0 - epsilon))
    return torch.asin(argument) / math.pi
