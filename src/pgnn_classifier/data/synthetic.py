from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import multiplicative_injection, trapezoid_transit_model
from .schema import DatasetSchema


G_SI = 6.67430e-11


def _duration_fraction(k: float, a_over_rstar: float, b: float) -> float:
    numerator = max((1.0 + k) ** 2 - b**2, 1e-12)
    denominator = max(a_over_rstar**2 - b**2, numerator + 1e-12)
    argument = min(max(math.sqrt(numerator / denominator), 1e-8), 1.0 - 1e-8)
    return math.asin(argument) / math.pi


def _draw_view(
    rng: np.random.Generator,
    phase: np.ndarray,
    depth: float,
    duration_fraction: float,
    noise: float,
    secondary_depth: float = 0.0,
    v_shaped: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    baseline = 1.0 + rng.normal(0.0, noise, phase.size)
    ingress = 0.48 if v_shaped else 0.15
    model = trapezoid_transit_model(phase, depth, duration_fraction, ingress)
    flux = multiplicative_injection(baseline, model).astype(np.float64)
    if secondary_depth > 0.0:
        secondary = trapezoid_transit_model(phase - 0.5, secondary_depth, duration_fraction, ingress)
        flux = multiplicative_injection(flux, secondary)
    error = np.full(phase.size, noise, dtype=np.float32)
    mask = np.ones(phase.size, dtype=np.uint8)
    for _ in range(int(rng.integers(0, 3))):
        start = int(rng.integers(0, max(phase.size - 3, 1)))
        width = int(rng.integers(1, max(2, phase.size // 40)))
        mask[start : start + width] = 0
    flux[mask == 0] = np.nan
    error[mask == 0] = np.nan
    return flux.astype(np.float32), error, mask


def generate_synthetic_dataset(
    output: str | Path,
    stars: int = 300,
    seed: int = 42,
    local_length: int = 31,
    global_length: int = 301,
) -> Path:
    """Generate a scientifically structured smoke-test dataset, not publication data."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    schema = DatasetSchema(local_length=local_length, global_length=global_length)

    samples_per_star = rng.integers(1, 3, size=stars)
    n = int(samples_per_star.sum())
    local_phase = np.linspace(-0.08, 0.08, local_length, dtype=np.float32)
    global_phase = np.linspace(-0.5, 0.5, global_length, endpoint=False, dtype=np.float32)

    local_flux = np.empty((n, local_length), dtype=np.float32)
    local_error = np.empty_like(local_flux)
    local_mask = np.empty((n, local_length), dtype=np.uint8)
    global_flux = np.empty((n, global_length), dtype=np.float32)
    global_error = np.empty_like(global_flux)
    global_mask = np.empty((n, global_length), dtype=np.uint8)
    scalar_values = np.empty((n, schema.scalar_count), dtype=np.float32)
    scalar_errors = np.empty_like(scalar_values)
    scalar_reliability = np.empty_like(scalar_values)
    scalar_mask = np.ones((n, schema.scalar_count), dtype=np.uint8)
    physics_targets = np.empty((n, schema.physics_target_count), dtype=np.float32)
    physics_target_reliability = np.empty_like(physics_targets)
    physics_target_mask = np.ones((n, schema.physics_target_count), dtype=np.uint8)
    rows: list[dict[str, object]] = []

    index = 0
    for star_index, count in enumerate(samples_per_star):
        base_tic = f"{10_000_000 + star_index}"
        rho_star = float(1408.0 * rng.lognormal(0.0, 0.35))
        radius_star = float(rng.uniform(0.65, 1.55))
        for event_index in range(int(count)):
            label = int(rng.random() < 0.38)
            source_kind = "injection" if label == 1 and rng.random() < 0.25 else "observed"
            tic_id = f"{base_tic}-inj-{event_index}" if source_kind == "injection" else base_tic
            parent_tic_id = base_tic if source_kind == "injection" else ""
            period_days = float(np.exp(rng.uniform(np.log(0.7), np.log(30.0))))
            period_seconds = period_days * 86400.0
            a_over_rstar = float((G_SI * rho_star * period_seconds**2 / (3.0 * math.pi)) ** (1.0 / 3.0))

            if label == 1:
                depth = float(np.exp(rng.uniform(np.log(4e-4), np.log(2e-2))))
                k = math.sqrt(depth)
                impact_fraction = float(rng.uniform(0.05, 0.78))
                b = impact_fraction * (1.0 + k)
                q = _duration_fraction(k, a_over_rstar, b)
                odd_stat = abs(float(rng.normal(0.4, 0.35)))
                even_stat = abs(float(rng.normal(0.4, 0.35)))
                secondary_stat = abs(float(rng.normal(0.35, 0.30)))
                centroid_offset = abs(float(rng.normal(0.3, 0.30)))
                secondary_depth = 0.0
                v_shaped = False
                subclass = -1
            else:
                depth = float(np.exp(rng.uniform(np.log(3e-3), np.log(1.5e-1))))
                k = math.sqrt(depth)
                impact_fraction = float(rng.uniform(0.45, 0.98))
                b = impact_fraction * (1.0 + k)
                a_over_rstar *= float(rng.uniform(0.45, 1.8))
                q = min(_duration_fraction(k, max(a_over_rstar, 1.0 + k + 0.1), b), 0.2)
                subclass = int(rng.integers(0, 4))
                odd_stat = abs(float(rng.normal(4.0 if subclass in {0, 1} else 1.2, 1.5)))
                even_stat = abs(float(rng.normal(1.0, 0.8)))
                secondary_stat = abs(float(rng.normal(4.5 if subclass in {0, 2} else 1.0, 1.8)))
                centroid_offset = abs(float(rng.normal(4.0 if subclass in {1, 3} else 1.2, 1.8)))
                secondary_depth = depth * float(rng.uniform(0.08, 0.65)) if subclass in {0, 2} else 0.0
                v_shaped = subclass in {0, 1}

            noise = float(np.exp(rng.uniform(np.log(2e-4), np.log(2e-3))))
            local_flux[index], local_error[index], local_mask[index] = _draw_view(
                rng, local_phase, depth, q, noise, 0.0, v_shaped
            )
            global_flux[index], global_error[index], global_mask[index] = _draw_view(
                rng, global_phase, depth, q, noise, secondary_depth, v_shaped
            )
            duration_days = q * period_days
            mes = max(depth / noise * math.sqrt(max(int(rng.integers(2, 12)), 2)), 0.1)
            num_transits = float(max(2, int(27.4 / period_days) + int(rng.integers(0, 3))))
            contamination = float(np.clip(rng.beta(1.3, 7.0), 0.0, 0.95))
            centroid_error = float(max(0.05, rng.lognormal(-1.5, 0.3)))
            scalar_values[index] = np.asarray(
                [
                    period_days,
                    duration_days,
                    depth,
                    mes,
                    num_transits,
                    odd_stat,
                    even_stat,
                    secondary_stat,
                    rho_star,
                    radius_star,
                    centroid_offset * (1.0 + contamination),
                    centroid_error,
                    rho_star,
                ],
                dtype=np.float32,
            )
            scalar_errors[index] = np.maximum(np.abs(scalar_values[index]) * 0.05, 1e-5)
            scalar_reliability[index] = np.clip(1.0 - noise / 0.003 + rng.normal(0.0, 0.08, schema.scalar_count), 0.0, 1.0)
            physics_targets[index] = np.asarray([depth, k, b, q], dtype=np.float32)
            physics_target_reliability[index] = np.clip(
                rng.normal(0.90 if label == 1 else 0.55, 0.10, schema.physics_target_count), 0.0, 1.0
            )

            missing = rng.random(schema.scalar_count) < 0.06
            scalar_mask[index, missing] = 0
            scalar_values[index, missing] = np.nan
            scalar_errors[index, missing] = np.nan
            scalar_reliability[index, missing] = 0.0
            target_missing = rng.random(schema.physics_target_count) < 0.04
            physics_target_mask[index, target_missing] = 0
            physics_targets[index, target_missing] = np.nan
            physics_target_reliability[index, target_missing] = 0.0

            rows.append(
                {
                    "sample_id": f"TCE-{index:06d}",
                    "tic_id": tic_id,
                    "label": label,
                    "subclass_label": subclass,
                    "physics_eligible": bool(label == 1),
                    "source_kind": source_kind,
                    "parent_tic_id": parent_tic_id,
                    "sector": int(rng.integers(1, 89)),
                }
            )
            index += 1

    pd.DataFrame(rows).to_csv(output / "manifest.csv", index=False)
    schema.save(output / "schema.json")
    arrays = {
        "local_flux": local_flux,
        "local_error": local_error,
        "local_mask": local_mask,
        "global_flux": global_flux,
        "global_error": global_error,
        "global_mask": global_mask,
        "scalar_values": scalar_values,
        "scalar_errors": scalar_errors,
        "scalar_reliability": scalar_reliability,
        "scalar_mask": scalar_mask,
        "physics_targets": physics_targets,
        "physics_target_reliability": physics_target_reliability,
        "physics_target_mask": physics_target_mask,
    }
    for name, array in arrays.items():
        np.save(output / f"{name}.npy", array)
    return output
