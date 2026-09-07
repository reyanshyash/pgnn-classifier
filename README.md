# PGNN Classifier

A compact, reliability-aware physics-guided neural network for **classifying already-detected TESS threshold-crossing events (TCEs)**. It is a vetter, not a blind transit-search algorithm.

The default model combines two phase-folded light-curve views with a small set of cached SPOC/ExoMiner diagnostics. It predicts a calibrated planet probability, a false-positive subclass, and four constrained transit quantities. The default network has approximately **77,000 trainable parameters** and the trainer rejects configurations above 500,000 parameters.

## What is implemented

- Shared small 1D CNN for local and global phase-folded views.
- Reliability-aware scalar input: value, uncertainty, reliability, and availability.
- Concatenation plus a small MLP instead of degenerate single-vector attention.
- No confirmed-planet prior imputation.
- Original odd/even, secondary, stellar, and centroid evidence is never overwritten.
- Differentiable physical outputs and masked Huber supervision.
- Circular-orbit depth/duration consistency only for physics-eligible planets or injections.
- Fixed global loss weights with warm-up; no adaptive weighting or PCGrad.
- TIC-grouped train/validation/calibration/test splitting.
- Temperature calibration on an untouched calibration split.
- PR-AUC, ROC-AUC, precision, recall, Brier score, NLL, ECE, and confusion matrix.
- Memory-mapped arrays, a synthetic end-to-end demo, official ExoMiner TFRecord converter, and tests.

## Architecture

```text
local folded flux/error/mask  ─┐
                               ├─ shared small CNN ─┐
global folded flux/error/mask ─┘                    │
                                                    ├─ fused MLP ─ planet logit
scalar value/error/reliability/mask ─ scalar MLP ──┤             ├ false-positive class
                                                    │             └ physical outputs
                                                    └──────────────────────┬─────────
                                                                           └ masked physics loss
```

The physical output vector is:

```text
[depth fraction, radius ratio Rp/R*, impact parameter, duration/period]
```

Depth, radius ratio, impact parameter and fractional duration are constrained to safe ranges by the model. The impact parameter target is optional because the public ExoMiner records do not provide a trustworthy value for every TCE.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the equations and design decisions.

## Installation

Python 3.10–3.12 is the safest choice for the optional TensorFlow converter. Training itself requires only PyTorch, NumPy, pandas, and PyYAML.

```bash
cd "PGNN classifier"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

To run the unit tests with the standard library:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Five-minute synthetic smoke test

The synthetic data validates plumbing and gradient flow. It is not scientifically valid training data.

```bash
pgnn generate-synthetic --output data/synthetic --stars 300 --seed 7
pgnn validate-data --data data/synthetic
pgnn train --data data/synthetic --config configs/smoke.yaml --output runs/smoke
pgnn predict --data data/synthetic --checkpoint runs/smoke/best.pt --split test --output runs/smoke/test_predictions.csv
```

For a real experiment, use `configs/lightweight.yaml` and tune only on the validation split.

## Using the official ExoMiner++ data

The most direct current sources are:

- [NASA ExoMiner code and specifications](https://github.com/nasa/ExoMiner)
- [Original TESS-SPOC two-minute dataset](https://doi.org/10.5281/zenodo.15466293)
- [ExoMiner++ 2.0 two-minute and FFI dataset](https://doi.org/10.5281/zenodo.17707413)

Download and extract the supervised/evaluation TFRecord archive. A catalog CSV by itself is insufficient because it does not contain the local and global flux tensors.

Install the converter dependency only in the conversion environment:

```bash
python -m pip install -e '.[exominer]'
```

Convert extracted uncompressed TFRecord shards:

```bash
pgnn convert-exominer \
  --input /path/to/extracted/shards \
  --output data/exominer_eval \
  --pattern '*.tfrec*' \
  --source-state normalized-pipeline \
  --obs-type 2min
```

If the shard names use another extension, change `--pattern`. Conversion streams through the archive and writes memory-mapped arrays; it does not load the full archive into RAM. Without `--expected-records`, it performs a counting pass first. `UNK` examples are skipped rather than incorrectly treating them as negatives.

For the separate unlabeled prediction archive, add `--include-unknown`. Those records are stored with `label=-1`; they can be scored using `pgnn predict` but are automatically excluded from supervised losses and evaluation metrics.

Use `--obs-type all` for a combined two-minute/FFI experiment or convert the two domains separately. Separate conversion is the safer first experiment because cadence and processing-domain shift can otherwise be confused with the class signal.

When an upstream normalized archive has already replaced missing values, their original provenance cannot be reconstructed. In `normalized-pipeline` mode, the converter therefore sets both availability and reliability to zero for the eight normalized diagnostic fields (`num_transits` through centroid error). Their finite, possibly imputed values cannot reach the classifier. Period, duration, depth, model S/N, and the two light-curve views remain enabled. Use `raw-zenodo` only when the source genuinely retains the documented missing-value sentinels.

The normalized centroid-error diagnostic is not a calibrated uncertainty in the units of the centroid-offset feature. The converter never copies it into `scalar_errors.npy` and does not derive a reliability score from it.

Then run:

```bash
pgnn validate-data --data data/exominer_eval
pgnn train --data data/exominer_eval --config configs/lightweight.yaml --output runs/exominer_v1
pgnn evaluate --data data/exominer_eval --checkpoint runs/exominer_v1/best.pt --split test --output runs/exominer_v1/test_predictions.csv
```

The converter uses the current NASA view sizes:

- Local view: 31 bins.
- Global view: 301 bins.
- Each view has flux, standard deviation derived from the stored variance, and a finite-value mask.

The label mapping is:

- `KP`, `CP` → planet (`1`).
- `BD`, `EB`, `FP`, `NTP` → non-planet (`0`).
- `UNK` → excluded from supervised training.

Only `KP` records are marked physics-eligible by the converter. `CP` records still train the binary classifier but do not receive the planet-consistency penalty; this avoids treating every candidate disposition as confirmed physical ground truth.

All observations from one `target_id` are assigned to one split.

## Data contract

A prepared dataset is one directory containing `manifest.csv`, `schema.json`, and aligned `.npy` arrays. Arrays are opened with memory mapping, so the complete dataset is not copied into memory.

Important manifest fields:

| Field | Meaning |
|---|---|
| `sample_id` | Unique TCE identifier |
| `tic_id` | Target/TIC grouping key |
| `label` | `1`, `0`, or `-1` for unknown |
| `subclass_label` | `BD=0`, `EB=1`, `FP=2`, `NTP=3`, otherwise `-1` |
| `physics_eligible` | True only for curated planets or known injections |
| `source_kind` | `observed` or `injection` |
| `parent_tic_id` | Original target for an injection |

The exact array names and shapes are documented in [DATA_FORMAT.md](DATA_FORMAT.md).

## Outputs

Training writes:

- `best.pt`: weights, feature order, normalization, persisted splits, calibration temperature, and threshold.
- `splits.csv`: auditable target-grouped assignments.
- `history.csv`: epoch-level training history.
- `metrics.json`: validation and untouched test results.
- `config.yaml`: exact configuration used.

The checkpoint also stores a manifest/schema fingerprint. Commands that request a persisted validation, calibration, or test split refuse to run if the dataset identity has changed. Use `--split all` when scoring a separate unlabeled dataset.

Prediction CSVs contain the calibrated planet probability and physical-head outputs. They are vetting scores, not automatic planet confirmations; high-value candidates still require scientific review and follow-up.

For optional leave-one-branch-out diagnostics:

```bash
pgnn predict --data data/exominer_eval --checkpoint runs/exominer_v1/best.pt \
  --split test --output runs/exominer_v1/explained_test.csv --explain
```

This performs three additional forward passes and reports how the probability changes when the local view, global view, or scalar branch is withheld. It is an inference-time diagnostic and does not increase the model size.

## Deliberately excluded from V1

- Blind BLS/TLS searching inside training.
- Transformers and cross-attention.
- Full target-pixel image branches.
- Learned replacement of unreliable values using planet distributions.
- Eccentricity, argument-of-periastron, and universal TTV fitting.
- Learned per-example loss weights and PCGrad.
- Forcing negative examples to satisfy planet geometry.

These should be evaluated later as isolated, compute-matched ablations rather than being assumed necessary.
