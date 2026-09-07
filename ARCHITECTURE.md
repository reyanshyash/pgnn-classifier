# Architecture and scientific constraints

## Scope

The package receives an existing TCE ephemeris and phase-folded views. Candidate discovery belongs upstream to SPOC/TPS, BLS, or TLS. This prevents candidate-search cost and classification cost from being mixed together.

## Reliability-aware scalar representation

For scalar feature value `x`, uncertainty `s`, reliability `r`, and availability mask `m`, the model receives:

```text
[(standardize(x) * r * m), (standardize_error(s) * r * m), (r * m), m]
```

Statistics are fitted on training targets only. Missing standardized values are zero. Because the value and error are multiplied by reliability, changing an unreliable raw value cannot affect the classifier. The mask remains visible so the model can distinguish a genuine zero from missing data.

No value is filled from a confirmed-planet distribution.

## Physical head

Given unconstrained neural outputs `u`, the head constructs:

```text
depth = epsilon + 0.5 sigmoid(u_depth)
k     = epsilon + sigmoid(u_k)
b     = (1 + k - epsilon) sigmoid(u_b)
q     = 0.5 sigmoid(u_q)
```

where `k = Rp/R*` and `q = T14/P`. These transforms guarantee positive depth and radius ratio, a transiting impact-parameter range, and `0 < q < 0.5`.

The public ExoMiner depth provides targets for `depth` and the explicitly identified approximation `k ≈ sqrt(depth)`. Duration and period provide `q`. Impact parameter is masked unless a trustworthy external transit fit exists.

## Physics losses

Classification uses weighted binary cross-entropy on every labeled TCE. False-positive subclasses use a masked cross-entropy term.

The auxiliary physical loss is a reliability-weighted Huber loss on transformed physical outputs. It is enabled only when `physics_eligible` is true and the individual target is available.

The first circular consistency term is:

```text
log(depth) ≈ 2 log(k)
```

If raw stellar density in kg/m³ and period in days are both available, the code also computes:

```text
a/R* = [G rho* (P seconds)^2 / (3 pi)]^(1/3)

q_geometry = asin(sqrt(((1+k)^2-b^2) / ((a/R*)^2-b^2))) / pi
```

and compares `log(q)` with `log(q_geometry)`. The density-duration term becomes exactly zero when raw density is unavailable; a normalized density feature is never inserted into a dimensional equation.

The total objective is:

```text
L = L_classification
  + lambda_aux * ramp(epoch) * L_auxiliary
  + lambda_geometry * ramp(epoch) * L_geometry
  + lambda_subclass * L_subclass
```

The default maximum weights are `lambda_aux=0.20` and `lambda_geometry=0.05`. Classification is warmed up before the physics terms reach full strength. There are no learned loss weights and no PCGrad.

## Why negative examples are excluded from planet consistency

Odd/even differences, secondary eclipses, centroid offsets, and inconsistent geometry are legitimate evidence of eclipsing binaries or contamination. The classifier receives those values unchanged. A negative example is never altered or penalized for failing to look like a planet.

## Calibration and leakage prevention

The split unit is the target star rather than an individual TCE. An injection uses its parent TIC as the group. The four persisted partitions have separate roles:

- Training: weights and normalization statistics.
- Validation: early stopping, architecture decisions, and decision threshold.
- Calibration: temperature scaling only.
- Test: final evaluation only.

The test split should not be repeatedly inspected while developing the model.
