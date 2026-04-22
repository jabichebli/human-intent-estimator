#!/usr/bin/env python3

import argparse
from pathlib import Path

from train_cnn_common import FEATURE_SLICES, TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "up_down"
DATASET_TAG = "_w300_e060_hpure"
TRAIN_BAGS = list(range(2, 29))
ARM_FEATURE_WIDTH = FEATURE_SLICES["arm_angles"][1] - FEATURE_SLICES["arm_angles"][0]


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
        help="Up/down bag ids to train on.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        choices=sorted(FEATURE_SLICES),
        default=["arm_angles", "arm_currents"],
        help="Feature blocks to feed the GRU.",
    )
    parser.add_argument(
        "--arm-joints",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Optional 0-based arm joint indices to keep or drop inside arm_angles/arm_currents. "
            f"Valid range: 0 to {ARM_FEATURE_WIDTH - 1}. For the current 7-wide arm vector, "
            "the gripper is typically index 6."
        ),
    )
    parser.add_argument(
        "--arm-joint-mode",
        choices=["keep", "drop"],
        default="keep",
        help="Interpret --arm-joints as the joints to keep or the joints to drop.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--artifact-stem",
        default="intent_up_down_gru",
        help="Output artifact stem under the up/down models folder.",
    )
    parser.add_argument(
        "--nonzero-threshold",
        type=optional_float,
        default=0.85,
        help=(
            "Deployment threshold: predict 5/6 only when the best nonzero "
            "probability is at least this value; otherwise predict 0. "
            "Use 'none' to disable thresholding and use argmax."
        ),
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


def resolve_arm_joint_indices(args):
    if args.arm_joints is None:
        return None

    unique_joints = sorted(set(args.arm_joints))
    invalid = [joint for joint in unique_joints if joint < 0 or joint >= ARM_FEATURE_WIDTH]
    if invalid:
        raise ValueError(
            f"--arm-joints contains invalid indices {invalid}; valid range is 0 to {ARM_FEATURE_WIDTH - 1}."
        )

    if args.arm_joint_mode == "keep":
        selected = unique_joints
    else:
        drop_set = set(unique_joints)
        selected = [joint for joint in range(ARM_FEATURE_WIDTH) if joint not in drop_set]

    if not selected:
        raise ValueError("Resolved arm joint subset is empty. Choose at least one arm joint.")

    return tuple(selected)


def make_config(args):
    arm_joint_indices = resolve_arm_joint_indices(args)
    feature_subindices = None
    if arm_joint_indices is not None:
        feature_subindices = {
            "arm_angles": arm_joint_indices,
            "arm_currents": arm_joint_indices,
        }

    return TrainingConfig(
        data_dir=DATA_DIR,
        train_x_filenames=[f"X_ud_{bag}{args.tag}.npy" for bag in args.bags],
        train_y_filenames=[f"y_ud_{bag}{args.tag}.npy" for bag in args.bags],
        val_x_filename=None,
        val_y_filename=None,
        test_x_filename=None,
        test_y_filename=None,
        derived_val_fraction=0.2,
        derived_split_mode="stratified_segments",
        split_seed=42,
        selected_features=args.features,
        artifact_stem=args.artifact_stem,
        feature_subindices=feature_subindices,
        export_artifacts=args.export_artifacts,
        batch_size=64,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        label_smoothing=0.0,
        onnx_opset_version=18,
        manual_class_loss_weights=None,
        nonzero_prediction_threshold=args.nonzero_threshold,
        selection_split="val",
        selection_metric="macro_f1",
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
    print(f"Training up/down GRU with tag: {args.tag}")
    print(f"Training up/down GRU with bags: {args.bags}")
    print(f"Training up/down GRU with features: {args.features}")
    if args.arm_joints is not None:
        print(
            f"Training up/down GRU with arm_joints={args.arm_joints} "
            f"(mode={args.arm_joint_mode})"
        )
    run_training(make_config(args))
