from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ScalarStandardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, mask: np.ndarray) -> "ScalarStandardizer":
        values = np.asarray(values, dtype=np.float64)
        valid = np.asarray(mask, dtype=bool) & np.isfinite(values)
        if values.ndim != 2 or valid.shape != values.shape:
            raise ValueError("values and mask must be matching [samples, features] arrays")

        means = np.zeros(values.shape[1], dtype=np.float64)
        scales = np.ones(values.shape[1], dtype=np.float64)
        for j in range(values.shape[1]):
            column = values[valid[:, j], j]
            if column.size == 0:
                continue
            means[j] = float(np.median(column))
            q25, q75 = np.percentile(column, [25.0, 75.0])
            robust_scale = float((q75 - q25) / 1.349)
            standard_scale = float(np.std(column))
            scales[j] = robust_scale if robust_scale > 1e-8 else max(standard_scale, 1.0)
        return cls(means.astype(np.float32), scales.astype(np.float32))

    def transform(
        self,
        values: np.ndarray,
        errors: np.ndarray,
        reliability: np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        errors = np.asarray(errors, dtype=np.float32)
        reliability = np.clip(np.nan_to_num(reliability, nan=0.0), 0.0, 1.0).astype(np.float32)
        available = (np.asarray(mask) > 0).astype(np.float32)
        usable = reliability * available

        normalized = np.nan_to_num((values - self.mean) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)
        normalized_error = np.nan_to_num(np.abs(errors) / self.scale, nan=0.0, posinf=0.0, neginf=0.0)
        return np.concatenate(
            [normalized * usable, normalized_error * usable, usable, available], axis=-1
        ).astype(np.float32)

    def state_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state_dict(cls, state: dict[str, list[float]]) -> "ScalarStandardizer":
        return cls(np.asarray(state["mean"], dtype=np.float32), np.asarray(state["scale"], dtype=np.float32))


def robust_normalize_view(
    flux: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    clip: float = 12.0,
) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float32)
    error = np.asarray(error, dtype=np.float32)
    valid = (np.asarray(mask) > 0) & np.isfinite(flux)
    if flux.ndim != 1 or error.shape != flux.shape or valid.shape != flux.shape:
        raise ValueError("flux, error, and mask must be matching one-dimensional arrays")

    if valid.any():
        center = float(np.median(flux[valid]))
        mad = float(np.median(np.abs(flux[valid] - center)))
        scale = max(1.4826 * mad, float(np.nanmedian(np.abs(error[valid]))), 1e-6)
    else:
        center, scale = 0.0, 1.0

    normalized_flux = np.nan_to_num((flux - center) / scale, nan=0.0, posinf=clip, neginf=-clip)
    normalized_error = np.nan_to_num(np.abs(error) / scale, nan=0.0, posinf=clip, neginf=0.0)
    normalized_flux = np.clip(normalized_flux, -clip, clip)
    normalized_error = np.clip(normalized_error, 0.0, clip)
    valid_channel = valid.astype(np.float32)
    normalized_flux *= valid_channel
    normalized_error *= valid_channel
    return np.stack([normalized_flux, normalized_error, valid_channel], axis=0).astype(np.float32)


def stack_pre_normalized_view(
    flux: np.ndarray,
    error: np.ndarray,
    mask: np.ndarray,
    clip: float = 20.0,
) -> np.ndarray:
    flux = np.asarray(flux, dtype=np.float32)
    error = np.asarray(error, dtype=np.float32)
    valid = (np.asarray(mask) > 0) & np.isfinite(flux) & np.isfinite(error)
    valid_channel = valid.astype(np.float32)
    safe_flux = np.clip(np.nan_to_num(flux, nan=0.0, posinf=clip, neginf=-clip), -clip, clip)
    safe_error = np.clip(np.nan_to_num(np.abs(error), nan=0.0, posinf=clip), 0.0, clip)
    return np.stack(
        [safe_flux * valid_channel, safe_error * valid_channel, valid_channel], axis=0
    ).astype(np.float32)


def multiplicative_injection(flux: np.ndarray, transit_model: np.ndarray) -> np.ndarray:
    """Inject a dimensionless transit model into relative or absolute flux."""
    flux = np.asarray(flux, dtype=np.float64)
    transit_model = np.asarray(transit_model, dtype=np.float64)
    if flux.shape != transit_model.shape:
        raise ValueError("flux and transit_model must have the same shape")
    if np.any(transit_model <= 0.0) or np.any(transit_model > 1.0):
        raise ValueError("transit_model must lie in the interval (0, 1]")
    return (flux * transit_model).astype(np.float32)


def trapezoid_transit_model(
    phase: np.ndarray,
    depth: float,
    duration_fraction: float,
    ingress_fraction: float = 0.15,
) -> np.ndarray:
    """Small dependency-free transit approximation for tests and augmentation demos."""
    if not (0.0 < depth < 1.0):
        raise ValueError("depth must lie in (0, 1)")
    if not (0.0 < duration_fraction < 0.5):
        raise ValueError("duration_fraction must lie in (0, 0.5)")
    if not (0.0 < ingress_fraction <= 0.5):
        raise ValueError("ingress_fraction must lie in (0, 0.5]")

    phase = ((np.asarray(phase, dtype=np.float64) + 0.5) % 1.0) - 0.5
    distance = np.abs(phase)
    half_duration = duration_fraction / 2.0
    ingress_width = max(duration_fraction * ingress_fraction, 1e-8)
    flat_edge = max(half_duration - ingress_width, 0.0)
    decrement = np.zeros_like(distance)
    decrement[distance <= flat_edge] = depth
    ingress = (distance > flat_edge) & (distance < half_duration)
    decrement[ingress] = depth * (half_duration - distance[ingress]) / ingress_width
    return (1.0 - decrement).astype(np.float32)
