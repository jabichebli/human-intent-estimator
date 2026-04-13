#!/usr/bin/env python3

from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]

# Choose from: "ff", "accel", "q", "dq", "arm_angles", "arm_currents"
# This config uses the full feature set, including arm motor currents.
# Tune architecture with conv_channels, kernel_sizes, pool_after_layers, classifier_hidden_dim, and dropout.
# Early stopping monitors validation loss with early_stopping_patience and early_stopping_min_delta.
config = TrainingConfig(
    data_dir=ROOT_DIR / "bag data" / "Processed Data" / "front_back",
    train_x_filenames=["X_fb_1.npy", "X_fb_2.npy"],
    train_y_filenames=["y_fb_1.npy", "y_fb_2.npy"],
    val_x_filename=None,
    val_y_filename=None,
    test_x_filename=None,
    test_y_filename=None,
    derived_val_fraction=0.2,
    split_seed=42,
    selected_features=["arm_angles", "arm_currents"],
    artifact_stem="intent_front_back",
    export_artifacts=False,
    batch_size=64,
    learning_rate=3e-4,
    num_epochs=20,
    onnx_opset_version=18,
    manual_class_loss_weights=[1.0, 1.0, 1.0],
    nonzero_prediction_threshold=0.8,
    selection_split="val",
    zero_division=0,
    conv_channels=(16, 32, 64),
    kernel_sizes=(5, 5, 3),
    pool_after_layers=(1, 2),
    classifier_hidden_dim=64,
    dropout=0.3,
    early_stopping_patience=3,
    early_stopping_min_delta=1e-4,
)


if __name__ == "__main__":
    run_training(config)
