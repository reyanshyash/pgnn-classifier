from __future__ import annotations

import math
import unittest

import torch

from pgnn_classifier.losses import LossWeights, PGNNLoss
from pgnn_classifier.models.pgnn import LightweightPGNN, PGNNConfig
from pgnn_classifier.physics.transit import circular_duration_fraction


def _batch(batch_size: int, eligible: bool) -> dict[str, torch.Tensor]:
    raw_scalars = torch.ones(batch_size, 13)
    raw_scalars[:, 0] = 5.0
    raw_scalars[:, 12] = 1408.0
    return {
        "label": torch.ones(batch_size),
        "subclass_label": torch.full((batch_size,), -1, dtype=torch.long),
        "physics_eligible": torch.full((batch_size,), eligible, dtype=torch.bool),
        "physics_targets": torch.tensor([[0.0025, 0.05, 0.3, 0.03]]).repeat(batch_size, 1),
        "physics_target_mask": torch.ones(batch_size, 4),
        "physics_target_reliability": torch.ones(batch_size, 4),
        "scalar_values_raw": raw_scalars,
        "scalar_mask": torch.ones(batch_size, 13),
        "scalar_reliability": torch.ones(batch_size, 13),
    }


class PhysicsLossTests(unittest.TestCase):
    def test_duration_equation(self) -> None:
        k = torch.tensor([0.1])
        a = torch.tensor([10.0])
        b = torch.tensor([0.0])
        expected = math.asin(1.1 / 10.0) / math.pi
        self.assertAlmostEqual(float(circular_duration_fraction(k, a, b)), expected, places=6)

    def test_physics_loss_has_gradient_when_eligible(self) -> None:
        config = PGNNConfig(conv_channels=(8, 8), lightcurve_latent=16, scalar_hidden=16, scalar_latent=16,
                            fusion_hidden=24, fusion_latent=16, dropout=0.0, num_subclasses=0)
        model = LightweightPGNN(config)
        outputs = model(torch.randn(3, 3, 31), torch.randn(3, 3, 101), torch.randn(3, 52))
        criterion = PGNNLoss(1.0, LossWeights(auxiliary=0.2, geometry=1.0, subclass=0.0))
        loss, components = criterion(outputs, _batch(3, True), physics_scale=1.0)
        self.assertGreater(float(components["geometry"]), 0.0)
        loss.backward()
        gradient = model.physical_head.linear.weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_physics_terms_are_zero_when_ineligible(self) -> None:
        physical = torch.tensor([[0.01, 0.2, 0.5, 0.1]], requires_grad=True)
        outputs = {"logit": torch.zeros(1, requires_grad=True), "physical": physical, "subclass_logits": None}
        criterion = PGNNLoss(1.0, LossWeights(auxiliary=1.0, geometry=1.0, subclass=0.0))
        _, components = criterion(outputs, _batch(1, False), physics_scale=1.0)
        self.assertEqual(float(components["auxiliary"]), 0.0)
        self.assertEqual(float(components["geometry"]), 0.0)

    def test_negative_label_overrides_bad_eligibility_metadata(self) -> None:
        physical = torch.tensor([[0.01, 0.2, 0.5, 0.1]], requires_grad=True)
        outputs = {"logit": torch.zeros(1, requires_grad=True), "physical": physical, "subclass_logits": None}
        batch = _batch(1, True)
        batch["label"] = torch.zeros(1)
        batch["subclass_label"] = torch.zeros(1, dtype=torch.long)
        criterion = PGNNLoss(1.0, LossWeights(auxiliary=1.0, geometry=1.0, subclass=0.0))
        _, components = criterion(outputs, batch, physics_scale=1.0)
        self.assertEqual(float(components["auxiliary"]), 0.0)
        self.assertEqual(float(components["geometry"]), 0.0)


if __name__ == "__main__":
    unittest.main()
