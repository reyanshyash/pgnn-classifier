from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import DatasetSchema


REQUIRED_FIELDS = (
    "local_flux_view_fluxnorm",
    "local_flux_view_fluxnorm_var",
    "global_flux_view_fluxnorm",
    "global_flux_view_fluxnorm_var",
    "label",
    "target_id",
    "uid",
    "sector_run",
    "tce_period",
    "tce_duration",
    "tce_depth",
    "tce_model_snr",
    "tce_num_transits_obs_norm",
    "flux_odd_local_stat_abs_min_norm",
    "flux_even_local_stat_abs_min_norm",
    "flux_weak_secondary_local_stat_abs_min_norm",
    "tce_sdens_norm",
    "tce_sradius_norm",
    "tce_dikco_msky_norm",
    "tce_dikco_msky_err_norm",
)

POSITIVE_LABELS = {"KP", "CP"}
NEGATIVE_LABELS = {"BD", "EB", "FP", "NTP"}
SUBCLASS_LABELS = {"BD": 0, "EB": 1, "FP": 2, "NTP": 3}

# These fields are supplied by ExoMiner as normalized diagnostic features.  In
# an archive produced by the normalized pipeline, a finite number does not
# prove that the original measurement existed: upstream median replacement may
# already have erased that provenance.  Period, duration, depth, and model S/N
# are separate physical/source fields and are therefore not in this slice.
NORMALIZED_DIAGNOSTIC_SLICE = slice(4, 12)


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is needed only for conversion. Install with "
            "`pip install -e '.[exominer]'`, convert once, then train with PyTorch."
        ) from exc
    return tf


def discover_tfrecords(root: str | Path, pattern: str = "*.tfrec*") -> list[Path]:
    root = Path(root)
    paths = sorted(path for path in root.rglob(pattern) if path.is_file())
    if not paths:
        raise FileNotFoundError(
            f"No TFRecord shards matching '{pattern}' were found below {root}. "
            "Pass --pattern matching the extracted shard names."
        )
    return paths


def _example(tf: Any, serialized: Any) -> Any:
    value = serialized.numpy() if hasattr(serialized, "numpy") else serialized
    parsed = tf.train.Example()
    parsed.ParseFromString(value)
    return parsed


def _probe(tf: Any, paths: list[Path]) -> set[str]:
    dataset = tf.data.TFRecordDataset([str(paths[0])], num_parallel_reads=1)
    first = next(iter(dataset.take(1)), None)
    if first is None:
        raise ValueError(f"First shard contains no records: {paths[0]}")
    keys = set(_example(tf, first).features.feature)
    missing = sorted(set(REQUIRED_FIELDS) - keys)
    if missing:
        raise ValueError(f"TFRecord schema is missing required fields: {missing}")
    return keys


def _bytes(features: Any, key: str) -> str:
    values = features[key].bytes_list.value
    if len(values) != 1:
        raise ValueError(f"Expected one bytes value for '{key}', got {len(values)}")
    return values[0].decode("utf-8").strip()


def _integer(features: Any, key: str) -> int:
    values = features[key].int64_list.value
    if len(values) != 1:
        raise ValueError(f"Expected one integer value for '{key}', got {len(values)}")
    return int(values[0])


def _floats(features: Any, key: str, expected: int) -> np.ndarray:
    values = np.asarray(features[key].float_list.value, dtype=np.float32)
    if values.size != expected:
        raise ValueError(f"Expected {expected} floats for '{key}', got {values.size}")
    return values


def _float(features: Any, key: str) -> float:
    return float(_floats(features, key, 1)[0])


def _records(tf: Any, paths: list[Path]) -> Iterable[Any]:
    dataset = tf.data.TFRecordDataset([str(path) for path in paths], num_parallel_reads=1)
    for serialized in dataset:
        yield _example(tf, serialized)


def _decode_label(label: str, include_unknown: bool = False) -> tuple[int, int] | None:
    normalized = label.strip().upper()
    if normalized == "UNK":
        return (-1, -1) if include_unknown else None
    if normalized in POSITIVE_LABELS:
        return 1, -1
    if normalized in NEGATIVE_LABELS:
        return 0, SUBCLASS_LABELS[normalized]
    raise ValueError(f"Unsupported ExoMiner label '{label}'")


def _scalar_channels(
    scalar: np.ndarray,
    source_state: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply the explicit availability policy for converted scalar fields.

    The public records do not provide calibrated per-example uncertainties for
    this 13-feature vector.  In particular, ``tce_dikco_msky_err_norm`` is a
    normalized diagnostic feature, not an uncertainty in the units of the
    stored centroid-offset feature, so it must not be copied into
    ``scalar_errors`` or converted into a reliability score.
    """
    scalar = np.asarray(scalar, dtype=np.float32)
    if scalar.shape != (13,):
        raise ValueError(f"Expected 13 scalar features, got shape {scalar.shape}")
    if source_state not in {"raw-zenodo", "normalized-pipeline"}:
        raise ValueError("source_state must be 'raw-zenodo' or 'normalized-pipeline'")

    mask = np.isfinite(scalar)
    mask[0:4] &= scalar[0:4] > 0.0
    if source_state == "raw-zenodo":
        # Missing centroid values retain their published pre-normalization
        # sentinels only in this source state.
        mask[10] &= scalar[10] != 0.0
        mask[11] &= scalar[11] != -1.0
    else:
        # Once upstream imputation has occurred, neither availability nor
        # reliability can be recovered safely from finiteness.
        mask[NORMALIZED_DIAGNOSTIC_SLICE] = False

    values = np.nan_to_num(scalar)
    errors = np.zeros_like(values, dtype=np.float32)
    reliability = mask.astype(np.float32)
    return values, errors, reliability, mask.astype(np.uint8)


def _record_obs_type(features: Any, available_fields: set[str]) -> str:
    return _bytes(features, "obs_type").lower() if "obs_type" in available_fields else "2min"


def _count_selected(
    tf: Any,
    paths: list[Path],
    max_records: int | None,
    available_fields: set[str],
    obs_type_filter: str | None,
    include_unknown: bool,
) -> int:
    selected = 0
    for example in _records(tf, paths):
        features = example.features.feature
        decoded = _decode_label(_bytes(features, "label"), include_unknown)
        matches_obs_type = obs_type_filter is None or _record_obs_type(features, available_fields) == obs_type_filter
        if decoded is not None and matches_obs_type:
            selected += 1
            if max_records is not None and selected >= max_records:
                break
    return selected


def _open_arrays(output: Path, n: int, schema: DatasetSchema) -> dict[str, np.memmap]:
    shapes = {
        "local_flux": (n, schema.local_length),
        "local_error": (n, schema.local_length),
        "local_mask": (n, schema.local_length),
        "global_flux": (n, schema.global_length),
        "global_error": (n, schema.global_length),
        "global_mask": (n, schema.global_length),
        "scalar_values": (n, schema.scalar_count),
        "scalar_errors": (n, schema.scalar_count),
        "scalar_reliability": (n, schema.scalar_count),
        "scalar_mask": (n, schema.scalar_count),
        "physics_targets": (n, schema.physics_target_count),
        "physics_target_reliability": (n, schema.physics_target_count),
        "physics_target_mask": (n, schema.physics_target_count),
    }
    arrays: dict[str, np.memmap] = {}
    for name, shape in shapes.items():
        dtype = np.uint8 if name.endswith("mask") else np.float32
        arrays[name] = np.lib.format.open_memmap(output / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
    return arrays


def convert_exominer_tfrecords(
    input_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.tfrec*",
    source_state: str = "normalized-pipeline",
    expected_records: int | None = None,
    max_records: int | None = None,
    progress_every: int = 10_000,
    obs_type_filter: str | None = None,
    include_unknown: bool = False,
) -> Path:
    """Stream official ExoMiner TFRecords into the laptop-friendly array contract."""
    if source_state not in {"raw-zenodo", "normalized-pipeline"}:
        raise ValueError("source_state must be 'raw-zenodo' or 'normalized-pipeline'")
    if obs_type_filter is not None:
        obs_type_filter = obs_type_filter.strip().lower()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    tf = _tensorflow()
    paths = discover_tfrecords(input_dir, pattern)
    available_fields = _probe(tf, paths)
    if source_state == "normalized-pipeline":
        warnings.warn(
            "The normalized pipeline may already have imputed missing scalar values; original missingness "
            "cannot be recovered. Normalized diagnostic fields will be disabled by setting their "
            "availability masks and reliabilities to zero.",
            stacklevel=2,
        )
    n = expected_records or _count_selected(
        tf, paths, max_records, available_fields, obs_type_filter, include_unknown
    )
    if max_records is not None:
        n = min(n, max_records)
    if n <= 0:
        raise ValueError("No selected ExoMiner records were found")

    schema = DatasetSchema(local_length=31, global_length=301, views_pre_normalized=True)
    arrays = _open_arrays(output, n, schema)
    rows: list[dict[str, object]] = []
    output_index = 0
    for example in _records(tf, paths):
        features = example.features.feature
        label_name = _bytes(features, "label").upper()
        decoded = _decode_label(label_name, include_unknown)
        if decoded is None:
            continue
        record_obs_type = _record_obs_type(features, available_fields)
        if obs_type_filter is not None and record_obs_type != obs_type_filter:
            continue
        label, subclass = decoded
        try:
            local_flux = _floats(features, "local_flux_view_fluxnorm", 31)
            local_variance = _floats(features, "local_flux_view_fluxnorm_var", 31)
            global_flux = _floats(features, "global_flux_view_fluxnorm", 301)
            global_variance = _floats(features, "global_flux_view_fluxnorm_var", 301)
            period_days = _float(features, "tce_period")
            duration_days = _float(features, "tce_duration")
            depth_fraction = _float(features, "tce_depth") * 1e-6
            model_snr = _float(features, "tce_model_snr")
            remaining_scalars = [
                _float(features, "tce_num_transits_obs_norm"),
                _float(features, "flux_odd_local_stat_abs_min_norm"),
                _float(features, "flux_even_local_stat_abs_min_norm"),
                _float(features, "flux_weak_secondary_local_stat_abs_min_norm"),
                _float(features, "tce_sdens_norm"),
                _float(features, "tce_sradius_norm"),
                _float(features, "tce_dikco_msky_norm"),
                _float(features, "tce_dikco_msky_err_norm"),
            ]
        except ValueError as exc:
            uid = _bytes(features, "uid")
            raise ValueError(f"Invalid record uid={uid}: {exc}") from exc

        local_valid = np.isfinite(local_flux) & np.isfinite(local_variance) & (local_variance >= 0)
        global_valid = np.isfinite(global_flux) & np.isfinite(global_variance) & (global_variance >= 0)
        arrays["local_flux"][output_index] = np.nan_to_num(local_flux)
        arrays["local_error"][output_index] = np.sqrt(np.clip(np.nan_to_num(local_variance), 0.0, None))
        arrays["local_mask"][output_index] = local_valid.astype(np.uint8)
        arrays["global_flux"][output_index] = np.nan_to_num(global_flux)
        arrays["global_error"][output_index] = np.sqrt(np.clip(np.nan_to_num(global_variance), 0.0, None))
        arrays["global_mask"][output_index] = global_valid.astype(np.uint8)

        scalar = np.asarray(
            [period_days, duration_days, depth_fraction, model_snr, *remaining_scalars, np.nan],
            dtype=np.float32,
        )
        scalar_values, scalar_errors, scalar_reliability, scalar_mask = _scalar_channels(
            scalar, source_state
        )
        arrays["scalar_values"][output_index] = scalar_values
        arrays["scalar_errors"][output_index] = scalar_errors
        arrays["scalar_reliability"][output_index] = scalar_reliability
        arrays["scalar_mask"][output_index] = scalar_mask

        target = np.zeros(4, dtype=np.float32)
        target_mask = np.zeros(4, dtype=np.uint8)
        valid_depth = math.isfinite(depth_fraction) and 0.0 < depth_fraction < 0.5
        valid_duration = (
            math.isfinite(period_days)
            and math.isfinite(duration_days)
            and period_days > 0.0
            and 0.0 < duration_days / period_days < 0.5
        )
        if valid_depth:
            target[0] = depth_fraction
            target[1] = math.sqrt(depth_fraction)
            target_mask[0:2] = 1
        if valid_duration:
            target[3] = duration_days / period_days
            target_mask[3] = 1
        arrays["physics_targets"][output_index] = target
        arrays["physics_target_mask"][output_index] = target_mask
        arrays["physics_target_reliability"][output_index] = target_mask.astype(np.float32)

        uid = _bytes(features, "uid")
        uid_obs_type = _bytes(features, "uid_obs_type") if "uid_obs_type" in available_fields else uid
        obs_type = _bytes(features, "obs_type") if "obs_type" in available_fields else "2min"
        rows.append(
            {
                "sample_id": uid_obs_type,
                "tic_id": str(_integer(features, "target_id")),
                "label": label,
                "subclass_label": subclass,
                "physics_eligible": bool(
                    label_name == "KP" and (target_mask[[0, 1, 3]] == 1).all()
                ),
                "source_kind": "observed",
                "parent_tic_id": "",
                "sector": _bytes(features, "sector_run"),
                "obs_type": obs_type,
                "original_label": label_name,
            }
        )
        output_index += 1
        if progress_every > 0 and output_index % progress_every == 0:
            print(f"Converted {output_index:,}/{n:,} labeled records")
        if output_index >= n:
            break

    if output_index != n:
        raise ValueError(f"Expected to write {n} records but wrote {output_index}")
    for array in arrays.values():
        array.flush()
    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    schema.save(output / "schema.json")
    return output
