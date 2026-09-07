from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from pgnn_classifier.data.synthetic import generate_synthetic_dataset
from pgnn_classifier.inference import predict_dataset
from pgnn_classifier.train import run_training


class SmokeTests(unittest.TestCase):
    def test_one_epoch_train_and_predict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = generate_synthetic_dataset(root / "data", stars=70, seed=11, global_length=101)
            config = {
                "seed": 11,
                "data": {
                    "local_length": 31,
                    "global_length": 101,
                    "batch_size": 32,
                    "num_workers": 0,
                    "split_ratios": {"train": 0.7, "validation": 0.1, "calibration": 0.1, "test": 0.1},
                },
                "model": {
                    "conv_channels": [8, 12],
                    "kernel_size": 5,
                    "lightcurve_latent": 16,
                    "scalar_hidden": 16,
                    "scalar_latent": 16,
                    "fusion_hidden": 32,
                    "fusion_latent": 16,
                    "dropout": 0.0,
                    "share_view_encoder": True,
                    "num_subclasses": 4,
                },
                "training": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "grad_clip_norm": 5.0,
                    "early_stopping_patience": 2,
                    "physics_warmup_epochs": 0,
                    "physics_ramp_epochs": 1,
                    "auxiliary_weight": 0.2,
                    "geometry_weight": 0.05,
                    "subclass_weight": 0.1,
                    "huber_delta": 0.1,
                    "device": "cpu",
                },
                "calibration": {"max_iterations": 20, "ece_bins": 10},
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            run = root / "run"
            summary = run_training(data, config_path, run)
            self.assertTrue((run / "best.pt").exists())
            self.assertLess(summary["trainable_parameters"], 500_000)
            predictions = predict_dataset(data, run / "best.pt", split="test", device_name="cpu")
            self.assertGreater(len(predictions), 0)
            self.assertTrue(predictions["planet_probability"].between(0.0, 1.0).all())

            checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["format_version"], 2)
            identity_map = checkpoint["split_identity_by_sample_id"]
            sample_id = next(iter(identity_map))
            identity_map[sample_id]["tic_id"] = "a-different-target"
            tampered_checkpoint = run / "tampered-target-identity.pt"
            torch.save(checkpoint, tampered_checkpoint)
            with self.assertRaisesRegex(ValueError, "target identity"):
                predict_dataset(data, tampered_checkpoint, split="test", device_name="cpu")


if __name__ == "__main__":
    unittest.main()
