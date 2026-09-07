from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .calibration import TemperatureScaler
from .data.dataset import ArrayStore, TCEDataset
from .data.schema import dataset_fingerprint
from .data.splitting import make_grouped_splits, save_split_manifest
from .engine import evaluate_loader, train_one_epoch
from .losses import LossWeights, PGNNLoss, physics_ramp
from .metrics import best_f1_threshold, binary_metrics, sigmoid, subclass_metrics
from .models.pgnn import LightweightPGNN, PGNNConfig
from .utils import choose_device, load_config, save_json, set_seed


def _indices(frame: pd.DataFrame, split: str) -> np.ndarray:
    return np.flatnonzero(frame["split"].to_numpy() == split)


def _loader(
    store: ArrayStore,
    indices: np.ndarray,
    standardizer: Any,
    batch_size: int,
    workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TCEDataset(store, indices, standardizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )


def _checkpoint_payload(
    model: LightweightPGNN,
    model_config: PGNNConfig,
    standardizer: Any,
    config: dict[str, Any],
    split_frame: pd.DataFrame,
    epoch: int,
    validation_pr_auc: float,
    store: ArrayStore,
) -> dict[str, Any]:
    identity_frame = split_frame[
        ["sample_id", "tic_id", "parent_tic_id", "group_id", "split"]
    ].copy()
    for column in identity_frame.columns:
        identity_frame[column] = identity_frame[column].fillna("").astype(str)
    split_identity_by_sample_id = {
        row["sample_id"]: {
            "tic_id": row["tic_id"],
            "parent_tic_id": row["parent_tic_id"],
            "group_id": row["group_id"],
            "split": row["split"],
        }
        for row in identity_frame.to_dict(orient="records")
    }
    return {
        "format_version": 2,
        "model_state": model.state_dict(),
        "model_config": model_config.to_dict(),
        "scalar_standardizer": standardizer.state_dict(),
        "config": config,
        "epoch": epoch,
        "validation_pr_auc": validation_pr_auc,
        "temperature": 1.0,
        "threshold": 0.5,
        "dataset_schema": {
            "scalar_features": list(store.schema.scalar_features),
            "physics_targets": list(store.schema.physics_targets),
            "local_length": store.schema.local_length,
            "global_length": store.schema.global_length,
            "views_pre_normalized": store.schema.views_pre_normalized,
        },
        "dataset_fingerprint": dataset_fingerprint(store.manifest, store.schema),
        "split_by_sample_id": dict(
            zip(split_frame["sample_id"].astype(str), split_frame["split"].astype(str), strict=True)
        ),
        "split_identity_by_sample_id": split_identity_by_sample_id,
    }


def run_training(
    data_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    start_time = time.perf_counter()
    config = load_config(config_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")

    seed = int(config.get("seed", 42))
    set_seed(seed)
    store = ArrayStore(data_dir, validate=True)
    data_config = config["data"]
    if int(data_config["local_length"]) != store.schema.local_length:
        raise ValueError("Configured local_length does not match schema.json")
    if int(data_config["global_length"]) != store.schema.global_length:
        raise ValueError("Configured global_length does not match schema.json")
    split_frame = make_grouped_splits(store.manifest, data_config["split_ratios"], seed)
    for split in ("train", "validation", "calibration", "test"):
        if not (_indices(split_frame, split).size):
            raise ValueError(f"The {split} split is empty; use more distinct TIC groups")
        known_labels = set(split_frame.loc[split_frame["split"] == split, "label"].astype(int)) & {0, 1}
        if known_labels != {0, 1}:
            raise ValueError(
                f"The {split} split does not contain both classes. Use more positive/negative TIC groups "
                "or adjust the split ratios."
            )
    save_split_manifest(split_frame, output_dir / "splits.csv")

    train_indices = _indices(split_frame, "train")
    standardizer = store.fit_scalar_standardizer(train_indices)
    batch_size = int(data_config.get("batch_size", 64))
    workers = int(data_config.get("num_workers", 0))
    loaders = {
        split: _loader(
            store,
            _indices(split_frame, split),
            standardizer,
            batch_size,
            workers,
            shuffle=split == "train",
        )
        for split in ("train", "validation", "calibration", "test")
    }

    model_values = dict(config["model"])
    model_values["scalar_count"] = store.schema.scalar_count
    model_config = PGNNConfig.from_dict(model_values)
    model = LightweightPGNN(model_config)
    if model.trainable_parameter_count > 500_000:
        raise ValueError(
            f"Model has {model.trainable_parameter_count:,} parameters; lightweight limit is 500,000"
        )
    training_config = config["training"]
    device = choose_device(str(training_config.get("device", "auto")))
    model.to(device)

    train_labels = store.manifest.iloc[train_indices]["label"].to_numpy(dtype=int)
    positive = int((train_labels == 1).sum())
    negative = int((train_labels == 0).sum())
    if positive == 0 or negative == 0:
        raise ValueError("Training split needs both positive and negative labeled examples")
    positive_weight = negative / positive
    criterion = PGNNLoss(
        positive_weight=positive_weight,
        weights=LossWeights(
            auxiliary=float(training_config["auxiliary_weight"]),
            geometry=float(training_config["geometry_weight"]),
            subclass=float(training_config.get("subclass_weight", 0.0)),
        ),
        huber_delta=float(training_config.get("huber_delta", 0.10)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )

    checkpoint_path = output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_pr_auc = -np.inf
    stale_epochs = 0
    max_epochs = int(training_config["epochs"])
    for epoch in range(max_epochs):
        ramp = physics_ramp(
            epoch,
            int(training_config["physics_warmup_epochs"]),
            int(training_config["physics_ramp_epochs"]),
        )
        train_losses = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            ramp,
            float(training_config["grad_clip_norm"]),
        )
        validation = evaluate_loader(model, loaders["validation"], device, criterion, ramp)
        validation_probabilities = sigmoid(np.asarray(validation["logits"]))
        validation_metrics = binary_metrics(
            np.asarray(validation["labels"]),
            validation_probabilities,
            ece_bins=int(config["calibration"]["ece_bins"]),
        )
        validation_losses = validation["losses"]
        assert isinstance(validation_losses, dict)
        row: dict[str, float | int] = {
            "epoch": epoch + 1,
            "physics_scale": ramp,
            "train_total": train_losses["total"],
            "validation_total": validation_losses["total"],
            "validation_pr_auc": float(validation_metrics["pr_auc"]),
            "validation_roc_auc": float(validation_metrics["roc_auc"]),
        }
        history.append(row)
        current_pr_auc = float(validation_metrics["pr_auc"])
        print(
            f"epoch={epoch + 1:03d} train_loss={train_losses['total']:.4f} "
            f"validation_loss={validation_losses['total']:.4f} "
            f"validation_pr_auc={current_pr_auc:.4f} physics_scale={ramp:.2f}"
        )
        if current_pr_auc > best_pr_auc + 1e-6:
            best_pr_auc = current_pr_auc
            stale_epochs = 0
            torch.save(
                _checkpoint_payload(
                    model,
                    model_config,
                    standardizer,
                    config,
                    split_frame,
                    epoch + 1,
                    current_pr_auc,
                    store,
                ),
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(training_config["early_stopping_patience"]):
                break

    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    calibration_outputs = evaluate_loader(model, loaders["calibration"], device)
    calibrator = TemperatureScaler()
    calibrator.fit(
        torch.from_numpy(np.asarray(calibration_outputs["logits"])),
        torch.from_numpy(np.asarray(calibration_outputs["labels"])),
        int(config["calibration"]["max_iterations"]),
    )
    validation_outputs = evaluate_loader(model, loaders["validation"], device)
    validation_probabilities = sigmoid(np.asarray(validation_outputs["logits"]) / calibrator.temperature)
    threshold = best_f1_threshold(np.asarray(validation_outputs["labels"]), validation_probabilities)
    checkpoint["temperature"] = calibrator.temperature
    checkpoint["threshold"] = threshold
    checkpoint["model_state"] = model.state_dict()
    torch.save(checkpoint, checkpoint_path)

    test_inference_start = time.perf_counter()
    test_outputs = evaluate_loader(model, loaders["test"], device)
    test_inference_seconds = time.perf_counter() - test_inference_start
    test_probabilities = sigmoid(np.asarray(test_outputs["logits"]) / calibrator.temperature)
    test_metrics = binary_metrics(
        np.asarray(test_outputs["labels"]),
        test_probabilities,
        threshold=threshold,
        ece_bins=int(config["calibration"]["ece_bins"]),
    )
    validation_metrics = binary_metrics(
        np.asarray(validation_outputs["labels"]),
        validation_probabilities,
        threshold=threshold,
        ece_bins=int(config["calibration"]["ece_bins"]),
    )
    test_subclass: dict[str, Any] | None = None
    if isinstance(test_outputs.get("subclass_logits"), np.ndarray):
        test_subclass = subclass_metrics(
            np.asarray(test_outputs["subclass_labels"]),
            np.argmax(np.asarray(test_outputs["subclass_logits"]), axis=1),
        )
    elapsed = time.perf_counter() - start_time
    summary: dict[str, Any] = {
        "device": str(device),
        "total_samples": len(store),
        "split_samples": {
            split: int(_indices(split_frame, split).size)
            for split in ("train", "validation", "calibration", "test")
        },
        "trainable_parameters": model.trainable_parameter_count,
        "best_epoch": int(checkpoint["epoch"]),
        "temperature": calibrator.temperature,
        "threshold": threshold,
        "training_seconds": elapsed,
        "test_inference_seconds": test_inference_seconds,
        "test_inference_ms_per_sample": 1000.0 * test_inference_seconds / len(loaders["test"].dataset),
        "validation": validation_metrics,
        "test": test_metrics,
        "test_false_positive_subclasses": test_subclass,
    }
    save_json(summary, output_dir / "metrics.json")
    return summary
