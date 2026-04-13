#!/usr/bin/env python3

from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]

# Choose from: "ff", "accel", "q", "dq", "arm_angles", "arm_currents"
# Example: set selected_features to ["ff", "arm_currents"] to train with currents too.
# Tune architecture with conv_channels, kernel_sizes, pool_after_layers, classifier_hidden_dim, and dropout.
# Early stopping monitors validation loss with early_stopping_patience and early_stopping_min_delta.
config = TrainingConfig(
    data_dir=ROOT_DIR / "bag_data" / "processed_data" / "left_right",
    train_x_filenames=["X_lr_1.npy", "X_lr_2.npy"],
    train_y_filenames=["y_lr_1.npy", "y_lr_2.npy"],
    val_x_filename=None,
    val_y_filename=None,
    test_x_filename="X_lr_3.npy",
    test_y_filename="y_lr_3.npy",
    derived_val_fraction=0.25,
    split_seed=42,
    selected_features=["ff", "accel", "dq"],
    artifact_stem="intent_left_right",
    export_artifacts=True,
    batch_size=64,
    learning_rate=1e-3,
    num_epochs=50,
    onnx_opset_version=18,
    manual_class_loss_weights=None,
    nonzero_prediction_threshold=None,
    selection_split="val",
    zero_division=0,
    conv_channels=(8, 16, 32),
    kernel_sizes=(5, 5, 3),
    pool_after_layers=(1, 2),
    classifier_hidden_dim=32,
    dropout=0.4,
    early_stopping_patience=4,
    early_stopping_min_delta=1e-3,
)


if __name__ == "__main__":
    run_training(config)
