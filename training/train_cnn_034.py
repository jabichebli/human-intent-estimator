#!/usr/bin/env python3

from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "left_right"

# Choose from: "ff", "accel", "q", "dq", "arm_angles", "arm_currents"
# This config trains the 0/3/4 classifier from the left/right bags.
# Bags are plain-named (no dataset tag) — re-parse with rosbag_parser.py and add a tag
# once you have tuned window/exclusion settings (see train_cnn_056.py for reference).
# Appending delta features and file-balanced sampling match the 056 production approach.
# Use eval_left_right_lobo.py for deployment-confidence checks across held-out bags.
config = TrainingConfig(
    data_dir=DATA_DIR,
    # Train on all bags for the deployment artifact — LOBO already proved generalization.
    # Holding back bag 3 as a test set here would just mean the exported model
    # trains on less data than LOBO's best folds without any new information.
    train_x_filenames=["X_lr_1.npy", "X_lr_2.npy", "X_lr_3.npy"],
    train_y_filenames=["y_lr_1.npy", "y_lr_2.npy", "y_lr_3.npy"],
    val_x_filename=None,
    val_y_filename=None,
    test_x_filename=None,
    test_y_filename=None,
    derived_val_fraction=0.2,
    derived_split_mode="stratified_windows",
    split_seed=42,
    selected_features=["ff", "accel", "dq"],
    artifact_stem="intent_left_right",
    export_artifacts=True,
    batch_size=64,
    learning_rate=5e-4,
    num_epochs=100,
    label_smoothing=0.0,
    onnx_opset_version=18,
    manual_class_loss_weights=None,
    nonzero_prediction_threshold=None,
    selection_split="val",
    selection_metric="accuracy",
    zero_division=0,
    show_learning_curves=False,
    conv_channels=(32, 64, 128),
    kernel_sizes=(7, 5, 3),
    pool_after_layers=(1, 2),
    classifier_hidden_dim=128,
    dropout=0.2,
    early_stopping_patience=15,
    early_stopping_min_delta=1e-4,
    seed=42,
    weight_decay=1e-4,
    use_lr_scheduler=True,
    use_delta_features=False,
    append_delta_features=True,
    use_gravity_comp=False,
    train_sampling_mode="uniform_files",
)


if __name__ == "__main__":
    run_training(config)
