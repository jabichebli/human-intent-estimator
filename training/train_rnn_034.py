#!/usr/bin/env python3

import argparse
from pathlib import Path

from train_cnn_common import (
    ALLOWED_SELECTION_METRICS,
    FEATURE_SLICES,
    TrainingConfig,
    run_training,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "left_right"
DATASET_TAG = "_w300_e060_hpure"
TRAIN_BAGS = list(range(1, 15))
DEFAULT_FEATURES = ["accel", "dq"]


def optional_float(value):
    if value.lower() in {"none", "null", "off"}:
        return None
    return float(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default=DATASET_TAG,
        help="Processed dataset suffix to train on.",
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        type=int,
        default=TRAIN_BAGS,
        help="Left/right bag ids to train on.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=sorted(FEATURE_SLICES),
        default=DEFAULT_FEATURES,
        help="Feature blocks to feed the GRU.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--artifact-stem",
        default="intent_left_right_gru",
        help="Output artifact stem under the left/right models folder.",
    )
    parser.add_argument(
        "--nonzero-threshold",
        type=optional_float,
        default=None,
        help=(
            "Deployment threshold: predict 3/4 only when the best nonzero "
            "probability is at least this value; otherwise predict 0. "
            "Use 'none' to disable thresholding and use argmax."
        ),
    )
    parser.add_argument(
        "--selection-metric",
        default="accuracy",
        choices=sorted(ALLOWED_SELECTION_METRICS),
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument(
        "--export-artifacts",
        dest="export_artifacts",
        action="store_true",
        default=True,
        help="Export .pt, .onnx, and .deploy.yaml artifacts.",
    )
    parser.add_argument(
        "--no-export-artifacts",
        dest="export_artifacts",
        action="store_false",
        help="Train/evaluate without writing model artifacts.",
    )
    return parser.parse_args()


def make_config(args):
    return TrainingConfig(
        data_dir=DATA_DIR,
        train_x_filenames=[f"X_lr_{bag}{args.tag}.npy" for bag in args.bags],
        train_y_filenames=[f"y_lr_{bag}{args.tag}.npy" for bag in args.bags],
        val_x_filename=None,
        val_y_filename=None,
        test_x_filename=None,
        test_y_filename=None,
        derived_val_fraction=0.2,
        derived_split_mode="stratified_windows",
        split_seed=42,
        selected_features=args.features,
        artifact_stem=args.artifact_stem,
        export_artifacts=args.export_artifacts,
        batch_size=64,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        label_smoothing=0.0,
        onnx_opset_version=18,
        manual_class_loss_weights=None,
        nonzero_prediction_threshold=args.nonzero_threshold,
        selection_split="val",
        selection_metric=args.selection_metric,
        zero_division=0,
        show_learning_curves=False,
        classifier_hidden_dim=128,
        dropout=args.dropout,
        early_stopping_patience=15,
        early_stopping_min_delta=1e-4,
        seed=42,
        weight_decay=1e-4,
        use_lr_scheduler=True,
        use_delta_features=False,
        append_delta_features=True,
        use_gravity_comp=False,
        train_sampling_mode="uniform_files",
        model_type="gru",
        gru_hidden_dim=args.hidden_dim,
        gru_num_layers=args.num_layers,
        gru_bidirectional=True,
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Training left/right GRU with tag: {args.tag}")
    print(f"Training left/right GRU with bags: {args.bags}")
    print(f"Training left/right GRU with features: {args.features}")
    run_training(make_config(args))
