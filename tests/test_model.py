from __future__ import annotations

import unittest

import torch

from pgnn_classifier.models.pgnn import LightweightPGNN, PGNNConfig


class ModelTests(unittest.TestCase):
    def test_shapes_constraints_and_size(self) -> None:
        model = LightweightPGNN(PGNNConfig())
        for batch_size in (1, 4):
            outputs = model(
                torch.randn(batch_size, 3, 31),
                torch.randn(batch_size, 3, 301),
                torch.randn(batch_size, 52),
            )
            self.assertEqual(outputs["logit"].shape, (batch_size,))
            self.assertEqual(outputs["physical"].shape, (batch_size, 4))
            physical = outputs["physical"]
            self.assertTrue(torch.isfinite(physical).all())
            depth, k, impact, q = physical.unbind(-1)
            self.assertTrue(((depth > 0) & (depth < 0.51)).all())
            self.assertTrue(((k > 0) & (k < 1.01)).all())
            self.assertTrue(((impact >= 0) & (impact < 1 + k)).all())
            self.assertTrue(((q > 0) & (q < 0.5)).all())
        self.assertLess(model.trainable_parameter_count, 500_000)


if __name__ == "__main__":
    unittest.main()
