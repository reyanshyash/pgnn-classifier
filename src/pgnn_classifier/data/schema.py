from __future__ import annotations

import json
import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SCALAR_FEATURES: tuple[str, ...] = (
    "period_days",
    "duration_days",
    "depth_fraction",
    "model_snr",
    "num_transits_feature",
    "odd_depth_stat",
    "even_depth_stat",
    "secondary_depth_stat",
    "stellar_density_feature",
    "stellar_radius_feature",
    "centroid_offset_feature",
    "centroid_offset_error_feature",
    "stellar_density_kg_m3_optional",
)

PHYSICS_TARGETS: tuple[str, ...] = (
    "depth_fraction",
    "radius_ratio",
    "impact_parameter",
    "duration_fraction",
)

REQUIRED_MANIFEST_COLUMNS: tuple[str, ...] = (
    "sample_id",
    "tic_id",
    "label",
    "subclass_label",
    "physics_eligible",
    "source_kind",
    "parent_tic_id",
)

ARRAY_FILES: tuple[str, ...] = (
    "local_flux.npy",
    "local_error.npy",
    "local_mask.npy",
    "global_flux.npy",
    "global_error.npy",
    "global_mask.npy",
    "scalar_values.npy",
    "scalar_errors.npy",
    "scalar_reliability.npy",
    "scalar_mask.npy",
    "physics_targets.npy",
    "physics_target_reliability.npy",
    "physics_target_mask.npy",
)


@dataclass(frozen=True)
class DatasetSchema:
    local_length: int = 31
    global_length: int = 301
    scalar_features: tuple[str, ...] = field(default_factory=lambda: SCALAR_FEATURES)
    physics_targets: tuple[str, ...] = field(default_factory=lambda: PHYSICS_TARGETS)
    views_pre_normalized: bool = False
    version: int = 1

    @property
    def scalar_count(self) -> int:
        return len(self.scalar_features)

    @property
    def physics_target_count(self) -> int:
        return len(self.physics_targets)

    def save(self, path: str | Path) -> None:
        payload = asdict(self)
        payload["scalar_features"] = list(self.scalar_features)
        payload["physics_targets"] = list(self.physics_targets)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "DatasetSchema":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        payload["scalar_features"] = tuple(payload["scalar_features"])
        payload["physics_targets"] = tuple(payload["physics_targets"])
        return cls(**payload)


def require_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"manifest.csv is missing required columns: {missing}")


def _integer_column(frame: pd.DataFrame, name: str, allowed: set[int]) -> np.ndarray:
    numeric = pd.to_numeric(frame[name], errors="raise").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.rint(numeric)).all():
        raise ValueError(f"{name} values must be exact integers")
    integers = numeric.astype(np.int64)
    invalid = sorted(set(integers.tolist()) - allowed)
    if invalid:
        raise ValueError(f"{name} contains unsupported values: {invalid}")
    return integers


def _boolean_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    def parse(value: object) -> bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        if isinstance(value, (int, float, np.integer, np.floating)) and float(value) in {0.0, 1.0}:
            return bool(value)
        if isinstance(value, str) and value.strip().lower() in {"true", "false", "1", "0"}:
            return value.strip().lower() in {"true", "1"}
        raise ValueError(f"{name} contains a non-boolean value: {value!r}")

    return frame[name].map(parse).to_numpy(dtype=bool)


def _slices(length: int, chunk_size: int = 8192) -> Iterable[slice]:
    for start in range(0, length, chunk_size):
        yield slice(start, min(start + chunk_size, length))


def dataset_fingerprint(frame: pd.DataFrame, schema: DatasetSchema) -> str:
    """Bind persisted splits to manifest identity and feature semantics."""
    identity_columns = list(REQUIRED_MANIFEST_COLUMNS)
    canonical = frame[identity_columns].copy()
    for column in identity_columns:
        canonical[column] = canonical[column].fillna("").astype(str)
    canonical = canonical.sort_values("sample_id", kind="mergesort")
    schema_payload = asdict(schema)
    digest = hashlib.sha256()
    digest.update(json.dumps(schema_payload, sort_keys=True).encode("utf-8"))
    digest.update(canonical.to_csv(index=False, lineterminator="\n").encode("utf-8"))
    return digest.hexdigest()


def validate_dataset_directory(root: str | Path, schema: DatasetSchema | None = None) -> None:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")
    manifest_path = root / "manifest.csv"
    schema_path = root / "schema.json"
    if not manifest_path.exists() or not schema_path.exists():
        raise FileNotFoundError("Dataset requires manifest.csv and schema.json")
    for filename in ARRAY_FILES:
        if not (root / filename).exists():
            raise FileNotFoundError(f"Dataset is missing {filename}")

    schema = schema or DatasetSchema.load(schema_path)
    frame = pd.read_csv(manifest_path)
    require_columns(frame, REQUIRED_MANIFEST_COLUMNS)
    n = len(frame)
    if n == 0:
        raise ValueError("Dataset contains no samples")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id values must be unique")
    labels = _integer_column(frame, "label", {-1, 0, 1})
    subclasses = _integer_column(frame, "subclass_label", {-1, 0, 1, 2, 3})
    physics_eligible = _boolean_column(frame, "physics_eligible")
    if np.any(physics_eligible & (labels != 1)):
        raise ValueError("Only label=1 samples may be physics_eligible")
    if np.any((labels != 0) & (subclasses != -1)):
        raise ValueError("Positive and unknown samples must use subclass_label=-1")
    source_kind = frame["source_kind"].astype(str).str.strip().str.lower()
    if not set(source_kind.unique()).issubset({"observed", "injection"}):
        raise ValueError("source_kind must be 'observed' or 'injection'")
    parent = frame["parent_tic_id"].fillna("").astype(str).str.strip()
    if ((source_kind == "injection") & parent.isin({"", "nan", "None"})).any():
        raise ValueError("Every injection must provide parent_tic_id")

    expected_shapes = {
        "local_flux.npy": (n, schema.local_length),
        "local_error.npy": (n, schema.local_length),
        "local_mask.npy": (n, schema.local_length),
        "global_flux.npy": (n, schema.global_length),
        "global_error.npy": (n, schema.global_length),
        "global_mask.npy": (n, schema.global_length),
        "scalar_values.npy": (n, schema.scalar_count),
        "scalar_errors.npy": (n, schema.scalar_count),
        "scalar_reliability.npy": (n, schema.scalar_count),
        "scalar_mask.npy": (n, schema.scalar_count),
        "physics_targets.npy": (n, schema.physics_target_count),
        "physics_target_reliability.npy": (n, schema.physics_target_count),
        "physics_target_mask.npy": (n, schema.physics_target_count),
    }
    arrays: dict[str, np.ndarray] = {}
    for filename, expected in expected_shapes.items():
        array = np.load(root / filename, mmap_mode="r")
        arrays[filename.removesuffix(".npy")] = array
        if array.shape != expected:
            raise ValueError(f"{filename} has shape {array.shape}; expected {expected}")

    for mask_name in ("local_mask", "global_mask", "scalar_mask", "physics_target_mask"):
        array = arrays[mask_name]
        for chunk in _slices(n):
            if not np.isin(array[chunk], [0, 1]).all():
                raise ValueError(f"{mask_name}.npy must contain only 0 and 1")

    for reliability_name, mask_name in (
        ("scalar_reliability", "scalar_mask"),
        ("physics_target_reliability", "physics_target_mask"),
    ):
        reliability, mask = arrays[reliability_name], arrays[mask_name]
        for chunk in _slices(n):
            values = np.asarray(reliability[chunk])
            available = np.asarray(mask[chunk]) > 0
            if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
                raise ValueError(f"{reliability_name}.npy must be finite and lie in [0,1]")
            if np.any(values[~available] != 0.0):
                raise ValueError(f"{reliability_name}.npy must be zero wherever {mask_name}.npy is zero")

    for prefix in ("local", "global"):
        flux, error, mask = arrays[f"{prefix}_flux"], arrays[f"{prefix}_error"], arrays[f"{prefix}_mask"]
        for chunk in _slices(n):
            available = np.asarray(mask[chunk]) > 0
            flux_values, error_values = np.asarray(flux[chunk]), np.asarray(error[chunk])
            if not np.isfinite(flux_values[available]).all():
                raise ValueError(f"{prefix}_flux.npy must be finite wherever its mask is one")
            if not np.isfinite(error_values[available]).all() or np.any(error_values[available] < 0.0):
                raise ValueError(f"{prefix}_error.npy must be finite and non-negative wherever available")

    for values_name, error_name, mask_name in (
        ("scalar_values", "scalar_errors", "scalar_mask"),
        ("physics_targets", None, "physics_target_mask"),
    ):
        values, mask = arrays[values_name], arrays[mask_name]
        errors = arrays[error_name] if error_name else None
        for chunk in _slices(n):
            available = np.asarray(mask[chunk]) > 0
            chunk_values = np.asarray(values[chunk])
            if not np.isfinite(chunk_values[available]).all():
                raise ValueError(f"{values_name}.npy must be finite wherever its mask is one")
            if errors is not None:
                chunk_errors = np.asarray(errors[chunk])
                if not np.isfinite(chunk_errors[available]).all() or np.any(chunk_errors[available] < 0.0):
                    raise ValueError(f"{error_name}.npy must be finite and non-negative wherever available")

    targets, target_mask = arrays["physics_targets"], arrays["physics_target_mask"]
    for chunk in _slices(n):
        values, available = np.asarray(targets[chunk]), np.asarray(target_mask[chunk]) > 0
        depth_ok = (~available[:, 0]) | ((values[:, 0] > 0.0) & (values[:, 0] < 0.5))
        radius_ok = (~available[:, 1]) | ((values[:, 1] > 0.0) & (values[:, 1] < 1.0))
        impact_ok = (~available[:, 2]) | (
            (values[:, 2] >= 0.0)
            & ((~available[:, 1]) | (values[:, 2] < 1.0 + np.maximum(values[:, 1], 0.0)))
        )
        duration_ok = (~available[:, 3]) | ((values[:, 3] > 0.0) & (values[:, 3] < 0.5))
        if not (depth_ok & radius_ok & impact_ok & duration_ok).all():
            raise ValueError("physics_targets.npy contains values outside the declared physical domains")
