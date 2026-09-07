from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from .physics.transit import circular_duration_fraction


G_SI = 6.67430e-11
PERIOD_DAYS_INDEX = 0
STELLAR_DENSITY_INDEX = 12


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    weight_sum = weights.sum()
    if bool(weight_sum.detach() <= 0):
        return reference.sum() * 0.0
    return (values * weights).sum() / weight_sum.clamp_min(1e-8)


def _transformed_physics(values: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    depth_fraction, radius_ratio, impact, duration_fraction = values.unbind(dim=-1)
    impact_fraction = impact / (1.0 + radius_ratio).clamp_min(epsilon)
    return torch.stack(
        [
            torch.log(depth_fraction.clamp_min(epsilon)),
            torch.log(radius_ratio.clamp_min(epsilon)),
            impact_fraction,
            torch.log(duration_fraction.clamp_min(epsilon)),
        ],
        dim=-1,
    )


@dataclass(frozen=True)
class LossWeights:
    auxiliary: float = 0.20
    geometry: float = 0.05
    subclass: float = 0.10


class PGNNLoss:
    def __init__(self, positive_weight: float, weights: LossWeights, huber_delta: float = 0.10) -> None:
        self.positive_weight = float(positive_weight)
        self.weights = weights
        self.huber_delta = float(huber_delta)

    def __call__(
        self,
        outputs: dict[str, torch.Tensor | None],
        batch: dict[str, torch.Tensor],
        physics_scale: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        logits = outputs["logit"]
        physical = outputs["physical"]
        assert isinstance(logits, torch.Tensor) and isinstance(physical, torch.Tensor)

        labels = batch["label"]
        labeled = labels >= 0
        if bool(labeled.any()):
            class_weights = torch.where(
                labels[labeled] > 0.5,
                torch.full_like(labels[labeled], self.positive_weight),
                torch.ones_like(labels[labeled]),
            )
            classification_elements = F.binary_cross_entropy_with_logits(
                logits[labeled], labels[labeled], reduction="none"
            )
            classification = (classification_elements * class_weights).sum() / class_weights.sum().clamp_min(1.0)
        else:
            classification = logits.sum() * 0.0

        eligible_positive = batch["physics_eligible"] & (labels == 1)
        eligible = eligible_positive.to(physical.dtype).unsqueeze(-1)
        target_mask = batch["physics_target_mask"].to(physical.dtype)
        target_reliability = batch["physics_target_reliability"].to(physical.dtype)
        auxiliary_weights = eligible * target_mask * target_reliability
        target_transformed = _transformed_physics(batch["physics_targets"].clamp_min(1e-7))
        prediction_transformed = _transformed_physics(physical)
        auxiliary_elements = F.smooth_l1_loss(
            prediction_transformed,
            target_transformed,
            reduction="none",
            beta=self.huber_delta,
        )
        auxiliary = _weighted_mean(auxiliary_elements, auxiliary_weights, physical)

        depth_residual = F.smooth_l1_loss(
            torch.log(physical[:, 0].clamp_min(1e-7)),
            2.0 * torch.log(physical[:, 1].clamp_min(1e-7)),
            reduction="none",
            beta=self.huber_delta,
        )
        eligible_one_dimensional = eligible_positive.to(physical.dtype)
        depth_geometry = _weighted_mean(depth_residual, eligible_one_dimensional, physical)

        scalar_values = batch["scalar_values_raw"]
        scalar_mask = batch["scalar_mask"].to(physical.dtype)
        scalar_reliability = batch["scalar_reliability"].to(physical.dtype)
        period_days = scalar_values[:, PERIOD_DAYS_INDEX].clamp_min(1e-7)
        density = scalar_values[:, STELLAR_DENSITY_INDEX].clamp_min(1e-7)
        period_seconds = period_days * 86400.0
        scaled_axis_from_density = torch.pow(
            G_SI * density * period_seconds.square() / (3.0 * torch.pi), 1.0 / 3.0
        )
        expected_duration = circular_duration_fraction(
            physical[:, 1], scaled_axis_from_density, physical[:, 2]
        )
        duration_residual = F.smooth_l1_loss(
            torch.log(physical[:, 3].clamp_min(1e-7)),
            torch.log(expected_duration.clamp_min(1e-7)),
            reduction="none",
            beta=self.huber_delta,
        )
        duration_weights = (
            eligible_one_dimensional
            * scalar_mask[:, PERIOD_DAYS_INDEX]
            * scalar_mask[:, STELLAR_DENSITY_INDEX]
            * scalar_reliability[:, PERIOD_DAYS_INDEX]
            * scalar_reliability[:, STELLAR_DENSITY_INDEX]
        )
        duration_geometry = _weighted_mean(duration_residual, duration_weights, physical)
        geometry = depth_geometry + duration_geometry

        subclass_logits = outputs["subclass_logits"]
        subclass_labels = batch["subclass_label"]
        subclass_valid = subclass_labels >= 0
        if isinstance(subclass_logits, torch.Tensor) and bool(subclass_valid.any()):
            subclass = F.cross_entropy(subclass_logits[subclass_valid], subclass_labels[subclass_valid])
        else:
            subclass = logits.sum() * 0.0

        total = (
            classification
            + physics_scale * self.weights.auxiliary * auxiliary
            + physics_scale * self.weights.geometry * geometry
            + self.weights.subclass * subclass
        )
        components = {
            "total": total.detach(),
            "classification": classification.detach(),
            "auxiliary": auxiliary.detach(),
            "geometry": geometry.detach(),
            "subclass": subclass.detach(),
        }
        return total, components


def physics_ramp(epoch: int, warmup_epochs: int, ramp_epochs: int) -> float:
    if epoch < warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / ramp_epochs)
