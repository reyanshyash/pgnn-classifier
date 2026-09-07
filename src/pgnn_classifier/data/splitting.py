from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_RATIOS = {"train": 0.70, "validation": 0.10, "calibration": 0.10, "test": 0.10}


def _canonical_identifier(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text in {"nan", "None"} else text


def target_group_ids(manifest: pd.DataFrame) -> pd.Series:
    """Return the leakage-safe target key for every TCE/event.

    Synthetic injections inherit the parent target's key, so neither another
    event from that target nor its source light curve can land in a different
    split.
    """
    if "tic_id" not in manifest:
        raise ValueError("manifest must contain tic_id")
    parent = (
        manifest["parent_tic_id"]
        if "parent_tic_id" in manifest
        else pd.Series("", index=manifest.index, dtype=object)
    )
    parent_ids = parent.map(_canonical_identifier)
    tic_ids = manifest["tic_id"].map(_canonical_identifier)
    if (tic_ids == "").any():
        raise ValueError("tic_id values must be non-empty")
    return parent_ids.where(parent_ids != "", tic_ids).rename("group_id")


def make_grouped_splits(
    manifest: pd.DataFrame,
    ratios: dict[str, float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    ratios = ratios or DEFAULT_RATIOS
    split_names = list(ratios)
    ratio_values = np.asarray([ratios[name] for name in split_names], dtype=float)
    if (
        not split_names
        or not np.isfinite(ratio_values).all()
        or np.any(ratio_values < 0.0)
        or not np.isclose(ratio_values.sum(), 1.0)
    ):
        raise ValueError("split ratios must be non-negative and sum to one")

    result = manifest.copy()
    result["group_id"] = target_group_ids(result)
    rng = np.random.default_rng(seed)
    group_rows: list[tuple[str, np.ndarray, float]] = []
    for group_id, group in result.groupby("group_id", sort=True):
        labels = group["label"].to_numpy(dtype=int)
        subclasses = group["subclass_label"].to_numpy(dtype=int)
        vector = np.asarray(
            [
                len(group),
                np.sum(labels == 1),
                np.sum(labels == 0),
                np.sum(labels == -1),
                *[np.sum(subclasses == index) for index in range(4)],
                1,
            ],
            dtype=float,
        )
        group_rows.append((str(group_id), vector, float(rng.random())))
    if not group_rows:
        raise ValueError("Cannot split an empty manifest")
    group_rows.sort(key=lambda item: (-item[1][0], item[2]))

    totals = np.sum([item[1] for item in group_rows], axis=0)
    targets = ratio_values[:, None] * totals[None, :]
    counts = np.zeros_like(targets)
    # Samples/TCEs, rather than target count alone, drive the allocation. The
    # extra class terms keep rare positives and negatives close to their target
    # proportions, while subclasses and group count are softer objectives.
    feature_weights = np.asarray([1.5, 3.0, 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.25])
    active_indices = np.flatnonzero(ratio_values > 0.0)
    if active_indices.size == 0:
        raise ValueError("At least one split ratio must be positive")

    # Reserve enough remaining target groups to populate each requested split
    # and, when mathematically possible, give every split both known classes.
    # This avoids a common greedy failure where a small holdout gets only one
    # class even though the dataset has sufficient independent target groups.
    class_indices = (1, 2)  # positive and negative event counts in `vector`
    constrained_classes = {
        feature_index
        for feature_index in class_indices
        if sum(vector[feature_index] > 0 for _, vector, _ in group_rows) >= active_indices.size
    }
    remaining_support = {
        feature_index: sum(vector[feature_index] > 0 for _, vector, _ in group_rows)
        for feature_index in constrained_classes
    }
    assigned_group_counts = np.zeros(len(split_names), dtype=np.int64)
    require_nonempty = len(group_rows) >= active_indices.size
    remaining_groups = len(group_rows)
    allocation: dict[str, str] = {}
    for group_id, vector, _ in group_rows:
        remaining_groups -= 1
        for feature_index in constrained_classes:
            remaining_support[feature_index] -= int(vector[feature_index] > 0)

        candidates: list[tuple[float, int]] = []
        for split_index in active_indices:
            trial = counts.copy()
            trial[split_index] += vector
            feasible = True
            if require_nonempty:
                trial_group_counts = assigned_group_counts.copy()
                trial_group_counts[split_index] += 1
                empty_splits = int(np.sum(trial_group_counts[active_indices] == 0))
                feasible = remaining_groups >= empty_splits
            if feasible:
                for feature_index in constrained_classes:
                    missing_splits = int(np.sum(trial[active_indices, feature_index] == 0))
                    if remaining_support[feature_index] < missing_splits:
                        feasible = False
                        break
            if not feasible:
                continue
            normalized_error = (trial - targets) / np.sqrt(targets + 1.0)
            cost = float(np.sum((normalized_error**2) * feature_weights[None, :]))
            candidates.append((cost, int(split_index)))
        if not candidates:
            raise RuntimeError("Could not satisfy target-group split constraints")
        _, best = min(candidates)
        allocation[group_id] = split_names[best]
        counts[best] += vector
        assigned_group_counts[best] += 1
    result["split"] = result["group_id"].map(allocation)
    if result["split"].isna().any():
        raise RuntimeError("At least one group was not assigned to a split")
    assert_no_group_leakage(result)
    return result


def assert_no_group_leakage(split_manifest: pd.DataFrame) -> None:
    counts = split_manifest.groupby("group_id")["split"].nunique()
    leaking = counts[counts > 1]
    if not leaking.empty:
        raise ValueError(f"Group leakage detected for {len(leaking)} groups")


def save_split_manifest(frame: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
