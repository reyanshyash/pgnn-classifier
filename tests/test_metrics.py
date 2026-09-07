from __future__ import annotations

import unittest

import numpy as np

from pgnn_classifier.metrics import binary_metrics


class MetricTests(unittest.TestCase):
    def test_perfect_ranking_has_unit_auc(self) -> None:
        metrics = binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
        self.assertAlmostEqual(float(metrics["roc_auc"]), 1.0)
        self.assertAlmostEqual(float(metrics["pr_auc"]), 1.0)


if __name__ == "__main__":
    unittest.main()
