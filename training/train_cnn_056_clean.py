#!/usr/bin/env python3

from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "up_down"
DATASET_TAG = "_w300_e060_hpure"

# Clean-label candidate: train on the full up/down set except the first-pass beam
# bags that looked most suspicious in LOBO for rest/intent boundary quality.
EXCLUDED_BAGS = [8, 10, 13] # EXCLUDED_BAGS = [8, 9, 10, 11, 12, 13]
TRAIN_BAGS = [bag for bag in range(2, 29) if bag not in EXCLUDED_BAGS]


config = TrainingConfig(
    data_dir=DATA_DIR,
    train_x_filenames=[f"X_ud_{bag}{DATASET_TAG}.npy" for bag in TRAIN_BAGS],
    train_y_filenames=[f"y_ud_{bag}{DATASET_TAG}.npy" for bag in TRAIN_BAGS],
    val_x_filename=None,
    val_y_filename=None,
    test_x_filename=None,
    test_y_filename=None,
    derived_val_fraction=0.2,
    derived_split_mode="stratified_segments",
    split_seed=42,
    selected_features=["arm_angles", "arm_currents"],
    artifact_stem="intent_up_down_clean_minus_8_10_13",
    export_artifacts=True,
    batch_size=64,
    learning_rate=5e-4,
    num_epochs=100,
    label_smoothing=0.0,
    onnx_opset_version=18,
    manual_class_loss_weights=None,
    nonzero_prediction_threshold=None,
    selection_split="val",
    selection_metric="macro_f1",
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
    print(f"Training clean up/down candidate with bags: {TRAIN_BAGS}")
    print(f"Excluded bags: {EXCLUDED_BAGS}")
    run_training(config)
