from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from pgnn_classifier.calibration import TemperatureScaler


class CalibrationTests(unittest.TestCase):
    def test_temperature_is_positive_and_does_not_worsen_fit_set_nll(self) -> None:
        labels = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        logits = torch.tensor([-8.0, -4.0, 3.0, -2.0, 5.0, 9.0])
        before = F.binary_cross_entropy_with_logits(logits, labels)
        scaler = TemperatureScaler()
        scaler.fit(logits, labels, max_iterations=50)
        after = F.binary_cross_entropy_with_logits(scaler.transform_logits(logits), labels)
        self.assertGreater(scaler.temperature, 0.0)
        self.assertLessEqual(float(after), float(before) + 1e-5)


if __name__ == "__main__":
    unittest.main()
