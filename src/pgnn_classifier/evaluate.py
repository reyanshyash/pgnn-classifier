from __future__ import annotations

from pathlib import Path
from typing import Any

from .inference import predict_dataset
from .metrics import binary_metrics, subclass_metrics
from .utils import save_json


def run_evaluation(
    data_dir: str | Path,
    checkpoint_path: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
    device_name: str = "auto",
) -> dict[str, Any]:
    predictions = predict_dataset(data_dir, checkpoint_path, split=split, device_name=device_name)
    metrics = binary_metrics(
        predictions["label"].to_numpy(dtype=int),
        predictions["planet_probability"].to_numpy(dtype=float),
        threshold=float(predictions["decision_threshold"].iloc[0]),
    )
    if "predicted_subclass" in predictions.columns:
        metrics["false_positive_subclasses"] = subclass_metrics(
            predictions["subclass_label"].to_numpy(dtype=int),
            predictions["predicted_subclass"].to_numpy(dtype=int),
        )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        save_json(metrics, output_path.with_suffix(".metrics.json"))
    return metrics
