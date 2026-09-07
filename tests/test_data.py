from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pgnn_classifier.data.dataset import ArrayStore, TCEDataset
from pgnn_classifier.data.convert_exominer import _decode_label, _scalar_channels
from pgnn_classifier.data.preprocessing import (
    ScalarStandardizer,
    multiplicative_injection,
    trapezoid_transit_model,
)
from pgnn_classifier.data.synthetic import generate_synthetic_dataset
from pgnn_classifier.data.schema import DatasetSchema, dataset_fingerprint, validate_dataset_directory


class DataTests(unittest.TestCase):
    def test_exominer_label_mapping_never_treats_unknown_as_negative(self) -> None:
        self.assertEqual(_decode_label("KP"), (1, -1))
        self.assertEqual(_decode_label("EB"), (0, 1))
        self.assertIsNone(_decode_label("UNK"))
        self.assertEqual(_decode_label("UNK", include_unknown=True), (-1, -1))

    def test_normalized_converter_disables_diagnostics_with_lost_provenance(self) -> None:
        scalar = np.arange(1, 14, dtype=np.float32)
        values, errors, reliability, mask = _scalar_channels(scalar, "normalized-pipeline")

        np.testing.assert_array_equal(mask[:4], np.ones(4, dtype=np.uint8))
        np.testing.assert_array_equal(reliability[:4], np.ones(4, dtype=np.float32))
        np.testing.assert_array_equal(mask[4:12], np.zeros(8, dtype=np.uint8))
        np.testing.assert_array_equal(reliability[4:12], np.zeros(8, dtype=np.float32))
        np.testing.assert_array_equal(values, scalar)
        np.testing.assert_array_equal(errors, np.zeros(13, dtype=np.float32))

    def test_centroid_error_feature_is_not_used_as_scalar_uncertainty(self) -> None:
        scalar = np.ones(13, dtype=np.float32)
        scalar[10] = 2.5
        scalar[11] = 0.4
        _, errors, reliability, mask = _scalar_channels(scalar, "raw-zenodo")

        self.assertEqual(mask[10], 1)
        self.assertEqual(mask[11], 1)
        self.assertEqual(reliability[10], 1.0)
        self.assertEqual(reliability[11], 1.0)
        self.assertEqual(errors[10], 0.0)
        self.assertEqual(errors[11], 0.0)

    def test_synthetic_dataset_loads_and_is_finite(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=20, seed=2)
            store = ArrayStore(directory)
            standardizer = store.fit_scalar_standardizer(np.arange(len(store)))
            sample = TCEDataset(store, [0], standardizer)[0]
            for key in ("local_view", "global_view", "scalar_input", "physics_targets"):
                self.assertTrue(sample[key].isfinite().all(), key)

    def test_zero_reliability_makes_value_neutral(self) -> None:
        standardizer = ScalarStandardizer(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
        first = standardizer.transform(
            np.array([1.0, 2.0]), np.array([0.1, 0.1]), np.array([0.0, 1.0]), np.ones(2)
        )
        second = standardizer.transform(
            np.array([9999.0, 2.0]), np.array([999.0, 0.1]), np.array([0.0, 1.0]), np.ones(2)
        )
        np.testing.assert_allclose(first, second)

    def test_multiplicative_injection(self) -> None:
        phase = np.linspace(-0.5, 0.5, 101, endpoint=False)
        model = trapezoid_transit_model(phase, depth=0.01, duration_fraction=0.08)
        flux = np.ones_like(model)
        injected = multiplicative_injection(flux, model)
        self.assertAlmostEqual(float(injected[0]), 1.0, places=6)
        self.assertLess(float(injected[np.argmin(np.abs(phase))]), 1.0)

    def test_schema_rejects_fractional_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            manifest_path = Path(directory) / "manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest["label"] = manifest["label"].astype(float)
            manifest.loc[0, "label"] = 0.7
            manifest.loc[0, "subclass_label"] = -1
            manifest.loc[0, "physics_eligible"] = False
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "exact integers"):
                validate_dataset_directory(directory)

    def test_schema_rejects_non_binary_labels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            manifest_path = Path(directory) / "manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "label"] = 2
            manifest.loc[0, "subclass_label"] = -1
            manifest.loc[0, "physics_eligible"] = False
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "label contains unsupported values"):
                validate_dataset_directory(directory)

    def test_schema_rejects_invalid_subclass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            manifest_path = Path(directory) / "manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "label"] = 0
            manifest.loc[0, "subclass_label"] = 4
            manifest.loc[0, "physics_eligible"] = False
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "subclass_label contains unsupported values"):
                validate_dataset_directory(directory)

    def test_schema_rejects_non_boolean_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            manifest_path = Path(directory) / "manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest["physics_eligible"] = manifest["physics_eligible"].astype(object)
            manifest.loc[0, "physics_eligible"] = "yes"
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "physics_eligible contains a non-boolean value"):
                validate_dataset_directory(directory)

    def test_schema_rejects_negative_sample_marked_physics_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            manifest_path = Path(directory) / "manifest.csv"
            manifest = pd.read_csv(manifest_path)
            manifest.loc[0, "label"] = 0
            manifest.loc[0, "subclass_label"] = 0
            manifest.loc[0, "physics_eligible"] = True
            manifest.to_csv(manifest_path, index=False)
            with self.assertRaisesRegex(ValueError, "Only label=1 samples may be physics_eligible"):
                validate_dataset_directory(directory)

    def test_schema_rejects_fractional_masks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=4)
            path = Path(directory) / "scalar_mask.npy"
            mask = np.load(path).astype(np.float32)
            mask[0, 0] = 0.5
            np.save(path, mask)
            with self.assertRaisesRegex(ValueError, "scalar_mask.npy must contain only 0 and 1"):
                validate_dataset_directory(directory)

    def test_schema_rejects_invalid_reliability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            generate_synthetic_dataset(directory, stars=8, seed=5)
            path = Path(directory) / "scalar_reliability.npy"
            reliability = np.load(path, mmap_mode="r+")
            reliability[0, 0] = 1.2
            reliability.flush()
            with self.assertRaisesRegex(ValueError, "lie in"):
                validate_dataset_directory(directory)

    def test_schema_rejects_invalid_physical_targets(self) -> None:
        invalid_targets = (
            (0, 0.0),
            (1, 1.0),
            (2, 2.0),
            (3, 0.5),
        )
        for target_index, invalid_value in invalid_targets:
            with self.subTest(target_index=target_index, invalid_value=invalid_value):
                with tempfile.TemporaryDirectory() as directory:
                    generate_synthetic_dataset(directory, stars=8, seed=6)
                    target_path = Path(directory) / "physics_targets.npy"
                    mask_path = Path(directory) / "physics_target_mask.npy"
                    reliability_path = Path(directory) / "physics_target_reliability.npy"
                    targets = np.load(target_path, mmap_mode="r+")
                    mask = np.load(mask_path, mmap_mode="r+")
                    reliability = np.load(reliability_path, mmap_mode="r+")
                    targets[0, target_index] = invalid_value
                    mask[0, target_index] = 1
                    reliability[0, target_index] = 1.0
                    if target_index == 2:
                        targets[0, 1] = 0.1
                        mask[0, 1] = 1
                        reliability[0, 1] = 1.0
                    targets.flush()
                    mask.flush()
                    reliability.flush()
                    with self.assertRaisesRegex(ValueError, "outside the declared physical domains"):
                        validate_dataset_directory(directory)

    def test_manifest_fingerprint_changes_with_target_identity(self) -> None:
        frame = pd.DataFrame(
            [{
                "sample_id": "a", "tic_id": "1", "label": 1, "subclass_label": -1,
                "physics_eligible": True, "source_kind": "observed", "parent_tic_id": ""
            }]
        )
        schema = DatasetSchema()
        original = dataset_fingerprint(frame, schema)
        frame.loc[0, "tic_id"] = "2"
        self.assertNotEqual(original, dataset_fingerprint(frame, schema))


if __name__ == "__main__":
    unittest.main()
