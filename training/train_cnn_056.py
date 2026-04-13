#!/usr/bin/env python3

from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "up_down"

# Choose from: "ff", "accel", "q", "dq", "arm_angles", "arm_currents"
# This config trains the 0/5/6 classifier from the 150 ms processed up/down bags.
# Best local result used the full feature set with a slightly wider temporal receptive field.
# Tune architecture with conv_channels, kernel_sizes, pool_after_layers, classifier_hidden_dim, and dropout.
# Early stopping monitors validation loss with early_stopping_patience and early_stopping_min_delta.
config = TrainingConfig(
    data_dir=DATA_DIR,
    train_x_filenames=["X_ud_3.npy", "X_ud_4.npy", "X_ud_5.npy", "X_ud_6.npy"],
    train_y_filenames=["y_ud_3.npy", "y_ud_4.npy", "y_ud_5.npy", "y_ud_6.npy"],
    val_x_filename=None,
    val_y_filename=None,
    test_x_filename=None,
    test_y_filename=None,
    derived_val_fraction=0.25,
    split_seed=42,
    selected_features=["ff", "accel", "arm_angles", "arm_currents"],
    artifact_stem="intent_up_down",
    export_artifacts=True,
    batch_size=64,
    learning_rate=5e-4,
    num_epochs=100,
    onnx_opset_version=18,
    manual_class_loss_weights=None,
    nonzero_prediction_threshold=None,
    selection_split="val",
    zero_division=0,
    show_learning_curves=False,
    conv_channels=(16, 32, 64),
    kernel_sizes=(7, 5, 3),
    pool_after_layers=(1, 2),
    classifier_hidden_dim=128,
    dropout=0.2,
    early_stopping_patience=8,
    early_stopping_min_delta=1e-4,
)


if __name__ == "__main__":
    run_training(config)
