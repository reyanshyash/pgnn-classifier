from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .preprocessing import ScalarStandardizer, robust_normalize_view, stack_pre_normalized_view
from .schema import ARRAY_FILES, DatasetSchema, validate_dataset_directory


def _as_bool(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


class ArrayStore:
    """Memory-mapped array bundle aligned row-for-row with manifest.csv."""

    def __init__(self, root: str | Path, validate: bool = True) -> None:
        self.root = Path(root)
        self.schema = DatasetSchema.load(self.root / "schema.json")
        if validate:
            validate_dataset_directory(self.root, self.schema)
        self.manifest = pd.read_csv(
            self.root / "manifest.csv",
            dtype={"sample_id": str, "tic_id": str, "parent_tic_id": str},
        )
        self.arrays = {
            filename.removesuffix(".npy"): np.load(self.root / filename, mmap_mode="r")
            for filename in ARRAY_FILES
        }

    def __len__(self) -> int:
        return len(self.manifest)

    def fit_scalar_standardizer(self, indices: Sequence[int]) -> ScalarStandardizer:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            raise ValueError("Cannot fit scalar normalization on an empty training split")
        values = np.asarray(self.arrays["scalar_values"][indices])
        mask = np.asarray(self.arrays["scalar_mask"][indices]) * (
            np.asarray(self.arrays["scalar_reliability"][indices]) > 0
        )
        return ScalarStandardizer.fit(values, mask)


class TCEDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        store: ArrayStore,
        indices: Sequence[int],
        scalar_standardizer: ScalarStandardizer,
    ) -> None:
        self.store = store
        self.indices = np.asarray(indices, dtype=np.int64)
        self.scalar_standardizer = scalar_standardizer

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        index = int(self.indices[item])
        arrays = self.store.arrays
        row = self.store.manifest.iloc[index]

        view_builder = stack_pre_normalized_view if self.store.schema.views_pre_normalized else robust_normalize_view
        local_view = view_builder(
            arrays["local_flux"][index], arrays["local_error"][index], arrays["local_mask"][index]
        )
        global_view = view_builder(
            arrays["global_flux"][index], arrays["global_error"][index], arrays["global_mask"][index]
        )
        scalar_input = self.scalar_standardizer.transform(
            arrays["scalar_values"][index],
            arrays["scalar_errors"][index],
            arrays["scalar_reliability"][index],
            arrays["scalar_mask"][index],
        )
        physics_mask = (
            np.asarray(arrays["physics_target_mask"][index], dtype=np.float32)
            * np.isfinite(arrays["physics_targets"][index]).astype(np.float32)
        )
        physics_targets = np.nan_to_num(
            arrays["physics_targets"][index], nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32)
        physics_reliability = np.clip(
            np.nan_to_num(arrays["physics_target_reliability"][index], nan=0.0), 0.0, 1.0
        ).astype(np.float32)

        return {
            "local_view": torch.from_numpy(local_view),
            "global_view": torch.from_numpy(global_view),
            "scalar_input": torch.from_numpy(scalar_input),
            "scalar_values_raw": torch.from_numpy(
                np.nan_to_num(arrays["scalar_values"][index], nan=0.0, posinf=0.0, neginf=0.0).astype(
                    np.float32
                )
            ),
            "scalar_mask": torch.from_numpy(np.asarray(arrays["scalar_mask"][index], dtype=np.float32)),
            "scalar_reliability": torch.from_numpy(
                np.clip(
                    np.nan_to_num(arrays["scalar_reliability"][index], nan=0.0), 0.0, 1.0
                ).astype(np.float32)
            ),
            "physics_targets": torch.from_numpy(physics_targets),
            "physics_target_mask": torch.from_numpy(physics_mask),
            "physics_target_reliability": torch.from_numpy(physics_reliability),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "subclass_label": torch.tensor(int(row["subclass_label"]), dtype=torch.long),
            "physics_eligible": torch.tensor(_as_bool(row["physics_eligible"]), dtype=torch.bool),
            "sample_index": torch.tensor(index, dtype=torch.long),
        }
