from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data.schema import DatasetSchema, validate_dataset_directory
from .data.synthetic import generate_synthetic_dataset
from .data.convert_exominer import convert_exominer_tfrecords
from .evaluate import run_evaluation
from .inference import predict_dataset
from .train import run_training


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pgnn",
        description="Lightweight reliability-aware PGNN for classifying TESS TCEs",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic = subparsers.add_parser("generate-synthetic", help="Create a smoke-test array bundle")
    synthetic.add_argument("--output", required=True, type=Path)
    synthetic.add_argument("--stars", type=int, default=300)
    synthetic.add_argument("--seed", type=int, default=42)
    synthetic.add_argument("--local-length", type=int, default=31)
    synthetic.add_argument("--global-length", type=int, default=301)

    validate = subparsers.add_parser("validate-data", help="Validate a prepared dataset")
    validate.add_argument("--data", required=True, type=Path)

    convert = subparsers.add_parser(
        "convert-exominer", help="Convert extracted official ExoMiner TFRecords once"
    )
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--output", required=True, type=Path)
    convert.add_argument("--pattern", default="*.tfrec*")
    convert.add_argument(
        "--source-state", default="normalized-pipeline", choices=["raw-zenodo", "normalized-pipeline"]
    )
    convert.add_argument("--expected-records", type=int)
    convert.add_argument("--max-records", type=int)
    convert.add_argument(
        "--obs-type",
        default="all",
        choices=["all", "2min", "ffi"],
        help="Optionally keep one cadence/domain from mixed ExoMiner++ 2.0 archives",
    )
    convert.add_argument(
        "--include-unknown",
        action="store_true",
        help="Keep UNK records with label=-1 so a trained checkpoint can score them",
    )

    train = subparsers.add_parser("train", help="Train, calibrate, and test the model")
    train.add_argument("--data", required=True, type=Path)
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate a persisted split")
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--checkpoint", required=True, type=Path)
    evaluate.add_argument("--split", default="test", choices=["train", "validation", "calibration", "test"])
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument("--device", default="auto")

    predict = subparsers.add_parser("predict", help="Write calibrated predictions")
    predict.add_argument("--data", required=True, type=Path)
    predict.add_argument("--checkpoint", required=True, type=Path)
    predict.add_argument("--split", default="all", choices=["all", "train", "validation", "calibration", "test"])
    predict.add_argument("--output", required=True, type=Path)
    predict.add_argument("--batch-size", type=int, default=128)
    predict.add_argument("--device", default="auto")
    predict.add_argument(
        "--explain",
        action="store_true",
        help="Run optional leave-one-branch-out diagnostics (three additional forward passes)",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if args.command == "generate-synthetic":
        output = generate_synthetic_dataset(
            args.output, args.stars, args.seed, args.local_length, args.global_length
        )
        print(f"Synthetic dataset written to {output}")
    elif args.command == "validate-data":
        validate_dataset_directory(args.data)
        schema = DatasetSchema.load(args.data / "schema.json")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "local_length": schema.local_length,
                    "global_length": schema.global_length,
                    "scalar_features": list(schema.scalar_features),
                    "physics_targets": list(schema.physics_targets),
                },
                indent=2,
            )
        )
    elif args.command == "convert-exominer":
        output = convert_exominer_tfrecords(
            args.input,
            args.output,
            pattern=args.pattern,
            source_state=args.source_state,
            expected_records=args.expected_records,
            max_records=args.max_records,
            obs_type_filter=None if args.obs_type == "all" else args.obs_type,
            include_unknown=args.include_unknown,
        )
        print(f"Converted ExoMiner data written to {output}")
    elif args.command == "train":
        print(json.dumps(run_training(args.data, args.config, args.output), indent=2))
    elif args.command == "evaluate":
        print(
            json.dumps(
                run_evaluation(args.data, args.checkpoint, args.split, args.output, args.device),
                indent=2,
            )
        )
    elif args.command == "predict":
        predictions = predict_dataset(
            args.data,
            args.checkpoint,
            split=args.split,
            batch_size=args.batch_size,
            device_name=args.device,
            explain=args.explain,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(args.output, index=False)
        print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
