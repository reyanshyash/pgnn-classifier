# Prepared dataset format

Every `.npy` file has the same leading dimension `N` and aligns row-for-row with `manifest.csv`. The trainer memory-maps these arrays.

## Light-curve arrays

| File | Shape | Meaning |
|---|---:|---|
| `local_flux.npy` | `[N, 31]` | Local folded view |
| `local_error.npy` | `[N, 31]` | Standard deviation, not variance |
| `local_mask.npy` | `[N, 31]` | Valid-bin mask |
| `global_flux.npy` | `[N, 301]` | Global folded view |
| `global_error.npy` | `[N, 301]` | Standard deviation, not variance |
| `global_mask.npy` | `[N, 301]` | Valid-bin mask |

Custom lengths are allowed when declared in `schema.json`; the official ExoMiner converter deliberately validates exactly 31 and 301 bins and never silently resizes them.

## Scalar arrays

Each scalar array has shape `[N, 13]` in this exact order:

1. `period_days`
2. `duration_days`
3. `depth_fraction`
4. `model_snr`
5. `num_transits_feature`
6. `odd_depth_stat`
7. `even_depth_stat`
8. `secondary_depth_stat`
9. `stellar_density_feature`
10. `stellar_radius_feature`
11. `centroid_offset_feature`
12. `centroid_offset_error_feature`
13. `stellar_density_kg_m3_optional`

Files:

| File | Meaning |
|---|---|
| `scalar_values.npy` | Raw/canonical values; the training fold fits robust normalization |
| `scalar_errors.npy` | Calibrated uncertainties in matching units where known, otherwise zero |
| `scalar_reliability.npy` | Continuous reliability in `[0,1]` |
| `scalar_mask.npy` | Availability mask in `{0,1}` |

The normalized ExoMiner stellar-density feature must not be copied into the final raw-density position. Leave the final mask zero unless density is known in kg/m³.

For an ExoMiner archive that has already passed through the normalized pipeline, upstream imputation makes original missingness unknowable. The converter uses the safe policy `mask=0, reliability=0` for scalar fields 5–12. These fields are excluded from normalization and model input even if their stored values are finite. Fields 1–4 and the light-curve views remain usable after their ordinary validity checks.

`centroid_offset_error_feature` is a normalized diagnostic scalar, not a calibrated error in the units of `centroid_offset_feature`. Do not copy it into `scalar_errors.npy` or use it to manufacture a per-example reliability. The bundled converter leaves scalar uncertainties at zero because this source does not supply compatible calibrated errors.

## Physical targets

Each physical array has shape `[N, 4]` in this order:

1. `depth_fraction`
2. `radius_ratio`
3. `impact_parameter`
4. `duration_fraction`

Files:

| File | Meaning |
|---|---|
| `physics_targets.npy` | Physical targets in the units above |
| `physics_target_reliability.npy` | Target reliability in `[0,1]` |
| `physics_target_mask.npy` | Target availability in `{0,1}` |

For the official public records, the converter sets radius ratio to the clearly documented approximation `sqrt(depth_fraction)` and leaves impact parameter unavailable. Supply impact parameter only from a vetted external fit.

## Manifest labels

`label=-1` means unknown and is excluded from classification loss. Never convert `UNK` into the negative class. `physics_eligible` must come from curated provenance—such as a confirmed/candidate planet label or a known injection—not from the network's own prediction.

For multiplicative injections, use the parent target in `parent_tic_id`; this keeps the injection and its source light curve in the same data split.
