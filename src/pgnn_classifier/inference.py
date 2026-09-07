from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data.dataset import ArrayStore, TCEDataset
from .data.preprocessing import ScalarStandardizer
from .data.schema import dataset_fingerprint
from .data.splitting import target_group_ids
from .engine import evaluate_loader
from .metrics import sigmoid
from .models.pgnn import LightweightPGNN, PGNNConfig
from .utils import choose_device


@torch.no_grad()
def _branch_ablation_probabilities(
    model: LightweightPGNN,
    loader: DataLoader,
    device: torch.device,
    temperature: float,
) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "without_local": [],
        "without_global": [],
        "without_scalars": [],
    }
    for batch in loader:
        local = batch["local_view"].to(device)
        global_view = batch["global_view"].to(device)
        scalars = batch["scalar_input"].to(device)
        variants = {
            "without_local": (torch.zeros_like(local), global_view, scalars),
            "without_global": (local, torch.zeros_like(global_view), scalars),
            "without_scalars": (local, global_view, torch.zeros_like(scalars)),
        }
        for name, inputs in variants.items():
            output = model(*inputs)["logit"]
            assert isinstance(output, torch.Tensor)
            parts[name].append(torch.sigmoid(output / temperature).cpu().numpy())
    return {name: np.concatenate(values) for name, values in parts.items()}


def load_checkpoint(
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> tuple[LightweightPGNN, dict[str, Any], ScalarStandardizer, torch.device]:
    device = choose_device(device_name)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = LightweightPGNN(PGNNConfig.from_dict(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    standardizer = ScalarStandardizer.from_state_dict(checkpoint["scalar_standardizer"])
    return model, checkpoint, standardizer, device


def _persisted_split_indices(
    store: ArrayStore,
    checkpoint: dict[str, Any],
    split: str,
) -> np.ndarray:
    expected_fingerprint = checkpoint.get("dataset_fingerprint")
    if not isinstance(expected_fingerprint, str) or not expected_fingerprint:
        raise ValueError(
            "Checkpoint has no dataset fingerprint; persisted splits cannot be reused safely"
        )
    actual_fingerprint = dataset_fingerprint(store.manifest, store.schema)
    if actual_fingerprint != expected_fingerprint:
        raise ValueError(
            "Persisted train/validation/test splits belong to a different dataset manifest"
        )

    identity_map = checkpoint.get("split_identity_by_sample_id")
    if not isinstance(identity_map, dict) or not identity_map:
        raise ValueError(
            "Checkpoint has no target-aware split identities; persisted splits cannot be reused safely"
        )
    sample_ids = store.manifest["sample_id"].astype(str)
    if len(identity_map) != len(sample_ids) or set(identity_map) != set(sample_ids):
        raise ValueError("Persisted split sample identities do not match the dataset manifest")

    current_group_ids = target_group_ids(store.manifest)
    assignments: list[str] = []
    for position, sample_id in enumerate(sample_ids):
        identity = identity_map.get(sample_id)
        if not isinstance(identity, dict):
            raise ValueError(f"Persisted identity is missing for sample_id={sample_id!r}")
        row = store.manifest.iloc[position]
        current_tic_id = "" if pd.isna(row["tic_id"]) else str(row["tic_id"]).strip()
        parent_value = row.get("parent_tic_id", "")
        current_parent_id = "" if pd.isna(parent_value) else str(parent_value).strip()
        expected_identity = (
            str(identity.get("tic_id", "")).strip(),
            str(identity.get("parent_tic_id", "")).strip(),
            str(identity.get("group_id", "")).strip(),
        )
        current_identity = (
            current_tic_id,
            current_parent_id,
            str(current_group_ids.iloc[position]).strip(),
        )
        if current_identity != expected_identity:
            raise ValueError(
                f"Persisted target identity does not match sample_id={sample_id!r}"
            )
        assignment = identity.get("split")
        if not isinstance(assignment, str) or not assignment:
            raise ValueError(f"Persisted split is missing for sample_id={sample_id!r}")
        assignments.append(assignment)

    indices = np.flatnonzero(np.asarray(assignments, dtype=object) == split)
    if indices.size == 0:
        raise ValueError(f"No dataset samples belong to split '{split}'")
    return indices


def predict_dataset(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    split: str | None = None,
    batch_size: int = 128,
    device_name: str = "auto",
    explain: bool = False,
) -> pd.DataFrame:
    store = ArrayStore(data_dir, validate=True)
    model, checkpoint, standardizer, device = load_checkpoint(checkpoint_path, device_name)
    if store.schema.scalar_count != model.config.scalar_count:
        raise ValueError("Dataset scalar feature count does not match the checkpoint")
    checkpoint_schema = checkpoint.get("dataset_schema", {})
    if checkpoint_schema and tuple(checkpoint_schema.get("scalar_features", ())) != store.schema.scalar_features:
        raise ValueError("Dataset scalar feature ordering does not match the checkpoint")
    if checkpoint_schema and tuple(checkpoint_schema.get("physics_targets", ())) != store.schema.physics_targets:
        raise ValueError("Dataset physics-target ordering does not match the checkpoint")
    if checkpoint_schema and int(checkpoint_schema.get("local_length", -1)) != store.schema.local_length:
        raise ValueError("Dataset local view length does not match the checkpoint")
    if checkpoint_schema and int(checkpoint_schema.get("global_length", -1)) != store.schema.global_length:
        raise ValueError("Dataset global view length does not match the checkpoint")
    if checkpoint_schema and bool(checkpoint_schema.get("views_pre_normalized", False)) != bool(
        store.schema.views_pre_normalized
    ):
        raise ValueError("Dataset light-curve normalization mode does not match the checkpoint")

    if split is None or split == "all":
        indices = np.arange(len(store), dtype=np.int64)
    else:
        indices = _persisted_split_indices(store, checkpoint, split)

    dataset = TCEDataset(store, indices, standardizer)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    outputs = evaluate_loader(model, loader, device)
    sample_indices = np.asarray(outputs["sample_indices"], dtype=np.int64)
    logits = np.asarray(outputs["logits"])
    probability = sigmoid(logits / float(checkpoint.get("temperature", 1.0)))
    threshold = float(checkpoint.get("threshold", 0.5))
    physical = np.asarray(outputs["physical"])
    result = store.manifest.iloc[sample_indices].reset_index(drop=True).copy()
    result["logit"] = logits
    result["planet_probability"] = probability
    result["decision_threshold"] = threshold
    result["predicted_label"] = (probability >= threshold).astype(int)
    for column, values in zip(store.schema.physics_targets, physical.T, strict=True):
        result[f"predicted_{column}"] = values
    for feature_index, feature_name in enumerate(store.schema.scalar_features):
        result[f"input_{feature_name}"] = np.asarray(
            store.arrays["scalar_values"][sample_indices, feature_index]
        )
        result[f"reliability_{feature_name}"] = np.asarray(
            store.arrays["scalar_reliability"][sample_indices, feature_index]
        )
        result[f"available_{feature_name}"] = np.asarray(
            store.arrays["scalar_mask"][sample_indices, feature_index]
        ).astype(int)
    subclass_logits = outputs.get("subclass_logits")
    if isinstance(subclass_logits, np.ndarray):
        result["predicted_subclass"] = np.argmax(subclass_logits, axis=1)
    if explain:
        ablated = _branch_ablation_probabilities(
            model,
            DataLoader(dataset, batch_size=batch_size, shuffle=False),
            device,
            float(checkpoint.get("temperature", 1.0)),
        )
        for name, probabilities in ablated.items():
            result[f"probability_{name}"] = probabilities
            result[f"support_from_{name.removeprefix('without_')}"] = probability - probabilities
    return result
