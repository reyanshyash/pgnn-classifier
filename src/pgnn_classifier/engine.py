from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from .losses import PGNNLoss
from .utils import move_batch


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: PGNNLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    physics_scale: float,
    grad_clip_norm: float,
) -> dict[str, float]:
    model.train()
    totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch["local_view"], batch["global_view"], batch["scalar_input"])
        loss, components = criterion(outputs, batch, physics_scale)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        batch_size = int(batch["label"].shape[0])
        samples += batch_size
        for name, value in components.items():
            totals[name] += float(value.cpu()) * batch_size
    return {name: value / max(samples, 1) for name, value in totals.items()}


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: PGNNLoss | None = None,
    physics_scale: float = 1.0,
) -> dict[str, np.ndarray | dict[str, float]]:
    model.eval()
    output_parts: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    loss_totals: defaultdict[str, float] = defaultdict(float)
    samples = 0
    for batch in loader:
        batch = move_batch(batch, device)
        outputs = model(batch["local_view"], batch["global_view"], batch["scalar_input"])
        logits = outputs["logit"]
        physical = outputs["physical"]
        subclass_logits = outputs["subclass_logits"]
        assert isinstance(logits, torch.Tensor) and isinstance(physical, torch.Tensor)
        output_parts["logits"].append(logits.cpu().numpy())
        output_parts["physical"].append(physical.cpu().numpy())
        output_parts["labels"].append(batch["label"].cpu().numpy())
        output_parts["subclass_labels"].append(batch["subclass_label"].cpu().numpy())
        output_parts["sample_indices"].append(batch["sample_index"].cpu().numpy())
        if isinstance(subclass_logits, torch.Tensor):
            output_parts["subclass_logits"].append(subclass_logits.cpu().numpy())
        if criterion is not None:
            _, components = criterion(outputs, batch, physics_scale)
            batch_size = int(batch["label"].shape[0])
            samples += batch_size
            for name, value in components.items():
                loss_totals[name] += float(value.cpu()) * batch_size

    if not output_parts["logits"]:
        raise ValueError("Data loader produced no batches")
    result: dict[str, np.ndarray | dict[str, float]] = {
        name: np.concatenate(parts, axis=0) for name, parts in output_parts.items()
    }
    if criterion is not None:
        result["losses"] = {name: value / max(samples, 1) for name, value in loss_totals.items()}
    return result
