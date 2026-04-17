#!/usr/bin/env python3

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


FEATURE_SLICES = {
    "ff": (0, 4),
    "accel": (4, 7),
    "q": (7, 19),
    "dq": (19, 31),
    "arm_angles": (31, 38),
    "arm_currents": (38, 45),
}

ALLOWED_LABEL_SETS = [
    (0, 1, 2),
    (0, 3, 4),
    (0, 5, 6),
]

ALLOWED_SELECTION_METRICS = {
    "accuracy",
    "macro_f1",
    "worst_class_f1",
}

ALLOWED_MODEL_TYPES = {
    "cnn",
    "gru",
}

ALLOWED_TRAIN_SAMPLING_MODES = {
    "shuffle",
    "uniform_files",
}

ALLOWED_DERIVED_SPLIT_MODES = {
    "stratified_windows",
    "stratified_segments",
}


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    train_x_filenames: list[str]
    train_y_filenames: list[str]
    selected_features: list[str]
    artifact_stem: str
    val_x_filename: str | None = None
    val_y_filename: str | None = None
    test_x_filename: str | None = None
    test_y_filename: str | None = None
    derived_val_fraction: float = 0.5
    derived_split_mode: str = "stratified_windows"
    split_seed: int = 42
    export_artifacts: bool = True
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    label_smoothing: float = 0.0
    onnx_opset_version: int = 18
    manual_class_loss_weights: list[float] | None = None
    nonzero_prediction_threshold: float | None = None
    selection_split: str = "val"
    selection_metric: str = "accuracy"
    zero_division: int = 0
    show_learning_curves: bool = True
    conv_channels: tuple[int, ...] = (32, 64, 128)
    kernel_sizes: tuple[int, ...] = (5, 5, 3)
    pool_after_layers: tuple[int, ...] = (1, 2)
    classifier_hidden_dim: int = 128
    dropout: float = 0.2
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    seed: int | None = None
    weight_decay: float = 0.0
    use_lr_scheduler: bool = False
    use_delta_features: bool = False
    append_delta_features: bool = False
    use_gravity_comp: bool = False
    train_segment_min_gap_sec: float | None = None
    train_sampling_mode: str = "shuffle"
    sampling_hz: int = 200
    model_type: str = "cnn"
    gru_hidden_dim: int = 64
    gru_num_layers: int = 1
    gru_bidirectional: bool = True


def write_deploy_yaml(
    path,
    model_name,
    num_features,
    num_timesteps,
    raw_label_set,
    label_to_index,
    index_to_label,
    x_mean,
    x_std,
    selected_features,
    onnx_opset_version,
    delta_feature_mode,
    nonzero_prediction_threshold=None,
    sampling_hz=200,
    gravity_comp_W=None,
    gravity_comp_b=None,
    model_type="cnn",
    architecture_metadata=None,
):
    feature_lines = []
    for name in selected_features:
        start, end = FEATURE_SLICES[name]
        feature_lines.extend([
            f"  - name: {name}",
            f"    source_start: {start}",
            f"    source_end: {end}",
            f"    width: {end - start}",
        ])

    window_ms = int(round(num_timesteps / sampling_hz * 1000))
    num_raw_features = num_features // 2 if delta_feature_mode == "append" else num_features

    lines = [
        "model:",
        "  format: onnx",
        f"  path: {model_name}",
        "  input_name: input",
        "  output_name: logits",
        f"  opset_version: {onnx_opset_version}",
        "  input_layout: NCT",
        "  dynamic_batch: true",
        f"  num_features: {num_features}",
        f"  num_timesteps: {num_timesteps}",
        f"  sampling_hz: {sampling_hz}",
        f"  window_ms: {window_ms}",
        "architecture:",
        f"  model_type: {model_type}",
        *[
            f"  {key}: {value}"
            for key, value in (architecture_metadata or {}).items()
        ],
        "features:",
        "  selected:",
        *feature_lines,
        "labels:",
        f"  raw_label_set: {list(raw_label_set)}",
        "  label_to_index:",
        *[f"    {int(label)}: {int(index)}" for label, index in label_to_index.items()],
        "  index_to_label:",
        *[f"    {int(index)}: {int(label)}" for index, label in index_to_label.items()],
        "inference:",
        "  output_type: logits",
        "  postprocess: softmax",
        "  prediction_rule: nonzero_threshold",
        f"  nonzero_prediction_threshold: {nonzero_prediction_threshold if nonzero_prediction_threshold is not None else 'null'}",
        "  zero_index: 0",
        "  nonzero_indices: [1, 2]",
        "preprocessing:",
        "  normalize: true",
        f"  delta_features: {str(delta_feature_mode != 'off').lower()}",
        f"  delta_feature_mode: {delta_feature_mode}",
        # 'append' mode: concat [raw, raw - raw[0]] along feature axis.
        # delta[t] = raw[t] - raw[window_start], NOT frame-to-frame differences.
        # x_mean/x_std cover the full concatenated vector (raw + delta).
        *([
            f"  delta_formula: window_start_subtraction",
            f"  num_raw_features: {num_raw_features}",
        ] if delta_feature_mode != "off" else []),
        f"  gravity_comp: {str(gravity_comp_W is not None).lower()}",
        *([
            "  gravity_comp_type: sinusoidal",
            f"  gravity_comp_W: {gravity_comp_W.tolist()}",
            *(
                [f"  gravity_comp_b: {gravity_comp_b.tolist()}"]
                if gravity_comp_b is not None
                else []
            ),
        ] if gravity_comp_W is not None else []),
        f"  x_mean: {x_mean.reshape(-1).astype(float).tolist()}",
        f"  x_std: {x_std.reshape(-1).astype(float).tolist()}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def infer_label_set(*label_arrays):
    raw_labels_present = sorted(set().union(*[np.unique(y).tolist() for y in label_arrays]))
    matches = [labels for labels in ALLOWED_LABEL_SETS if set(raw_labels_present).issubset(labels)]
    if len(matches) != 1:
        raise ValueError(
            f"Unsupported or ambiguous raw labels: {raw_labels_present}. "
            f"Expected labels to fit exactly one of {ALLOWED_LABEL_SETS}."
        )
    raw_label_set = matches[0]
    label_to_index = {label: idx for idx, label in enumerate(raw_label_set)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return raw_label_set, label_to_index, index_to_label


def remap_labels(y, label_to_index):
    return np.array([label_to_index[int(label)] for label in y], dtype=np.int64)


def select_features(X, selected_features):
    for name in selected_features:
        if name not in FEATURE_SLICES:
            raise ValueError(f"Unknown feature name: {name}")
    blocks = [X[:, :, FEATURE_SLICES[name][0]:FEATURE_SLICES[name][1]] for name in selected_features]
    return np.concatenate(blocks, axis=2)


def load_and_concat_datasets(data_dir, x_filenames, y_filenames):
    if len(x_filenames) != len(y_filenames):
        raise ValueError(
            f"Mismatched train file lists: {len(x_filenames)} X files vs {len(y_filenames)} y files."
        )
    if len(x_filenames) == 0:
        raise ValueError("At least one train X/y file pair is required.")

    X_parts = []
    y_parts = []
    for x_filename, y_filename in zip(x_filenames, y_filenames):
        X_part = np.load(data_dir / x_filename)
        y_part = np.load(data_dir / y_filename)
        if len(X_part) != len(y_part):
            raise ValueError(
                f"Sample count mismatch for {x_filename} and {y_filename}: "
                f"{len(X_part)} vs {len(y_part)}."
            )
        X_parts.append(X_part)
        y_parts.append(y_part)

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def sidecar_filename_for_x(x_filename, prefix):
    if not x_filename.startswith("X_"):
        raise ValueError(
            f"Expected X filename to start with 'X_', got {x_filename!r}."
        )
    return prefix + x_filename[2:]


def segment_gap_keep_indices(timestamps_ns, segment_ids, min_gap_ns):
    if min_gap_ns <= 0:
        return np.arange(len(segment_ids), dtype=np.int64)

    if not (len(timestamps_ns) == len(segment_ids)):
        raise ValueError(
            "Segment subsampling sidecars must match timestamp/segment lengths: "
            f"len(t)={len(timestamps_ns)}, len(seg)={len(segment_ids)}."
        )

    keep_indices = []
    segment_start = 0

    while segment_start < len(segment_ids):
        segment_id = segment_ids[segment_start]
        segment_end = segment_start + 1
        while segment_end < len(segment_ids) and segment_ids[segment_end] == segment_id:
            segment_end += 1

        keep_indices.append(segment_start)
        last_kept_t_ns = timestamps_ns[segment_start]

        for idx in range(segment_start + 1, segment_end - 1):
            if timestamps_ns[idx] - last_kept_t_ns >= min_gap_ns:
                keep_indices.append(idx)
                last_kept_t_ns = timestamps_ns[idx]

        last_idx = segment_end - 1
        if last_idx != keep_indices[-1]:
            keep_indices.append(last_idx)

        segment_start = segment_end

    return np.array(keep_indices, dtype=np.int64)


def subsample_by_segment_gap(X, y, timestamps_ns, segment_ids, min_gap_ns):
    if not (len(X) == len(y) == len(timestamps_ns) == len(segment_ids)):
        raise ValueError(
            "Segment subsampling sidecars must match X/y lengths: "
            f"len(X)={len(X)}, len(y)={len(y)}, len(t)={len(timestamps_ns)}, "
            f"len(seg)={len(segment_ids)}."
        )

    keep_indices = segment_gap_keep_indices(timestamps_ns, segment_ids, min_gap_ns)
    return X[keep_indices], y[keep_indices]


def load_and_concat_train_datasets(config: TrainingConfig):
    X_parts = []
    y_parts = []
    source_parts = []
    group_parts = []
    original_samples = 0
    kept_samples = 0

    if len(config.train_x_filenames) != len(config.train_y_filenames):
        raise ValueError(
            f"Mismatched train file lists: {len(config.train_x_filenames)} X files vs "
            f"{len(config.train_y_filenames)} y files."
        )
    if len(config.train_x_filenames) == 0:
        raise ValueError("At least one train X/y file pair is required.")

    min_gap_ns = None
    if config.train_segment_min_gap_sec is not None:
        min_gap_ns = int(round(config.train_segment_min_gap_sec * 1e9))

    for source_idx, (x_filename, y_filename) in enumerate(
        zip(config.train_x_filenames, config.train_y_filenames)
    ):
        X_part = np.load(config.data_dir / x_filename)
        y_part = np.load(config.data_dir / y_filename)
        if len(X_part) != len(y_part):
            raise ValueError(
                f"Sample count mismatch for {x_filename} and {y_filename}: "
                f"{len(X_part)} vs {len(y_part)}."
            )

        original_samples += len(y_part)

        need_segment_sidecar = min_gap_ns is not None or config.derived_split_mode == "stratified_segments"
        segment_ids = None
        timestamps_ns = None
        if need_segment_sidecar:
            t_filename = sidecar_filename_for_x(x_filename, "t_")
            seg_filename = sidecar_filename_for_x(x_filename, "seg_")
            timestamps_ns = np.load(config.data_dir / t_filename)
            segment_ids = np.load(config.data_dir / seg_filename)

        if min_gap_ns is not None:
            keep_indices = segment_gap_keep_indices(
                timestamps_ns=timestamps_ns,
                segment_ids=segment_ids,
                min_gap_ns=min_gap_ns,
            )
            X_part = X_part[keep_indices]
            y_part = y_part[keep_indices]
            segment_ids = segment_ids[keep_indices]

        kept_samples += len(y_part)
        X_parts.append(X_part)
        y_parts.append(y_part)
        source_parts.append(np.full(len(y_part), source_idx, dtype=np.int64))
        if config.derived_split_mode == "stratified_segments":
            global_group_ids = (np.int64(source_idx) << 32) + segment_ids.astype(np.int64)
            group_parts.append(global_group_ids)

    train_group_ids = None
    if config.derived_split_mode == "stratified_segments":
        train_group_ids = np.concatenate(group_parts, axis=0)

    return (
        np.concatenate(X_parts, axis=0),
        np.concatenate(y_parts, axis=0),
        np.concatenate(source_parts, axis=0),
        train_group_ids,
        original_samples,
        kept_samples,
    )


def load_dataset_pair(data_dir, x_filename, y_filename):
    X = np.load(data_dir / x_filename)
    y = np.load(data_dir / y_filename)
    if len(X) != len(y):
        raise ValueError(
            f"Sample count mismatch for {x_filename} and {y_filename}: {len(X)} vs {len(y)}."
        )
    return X, y


def stratified_split_indices(y, first_fraction, seed):
    if not 0.0 < first_fraction < 1.0:
        raise ValueError(f"first_fraction must be between 0 and 1, got {first_fraction}.")

    rng = np.random.default_rng(seed)
    first_indices = []
    second_indices = []

    for label in np.unique(y):
        label_indices = np.where(y == label)[0]
        shuffled = rng.permutation(label_indices)
        if len(shuffled) == 1:
            split_count = 1
        else:
            split_count = int(round(len(shuffled) * first_fraction))
            split_count = min(max(split_count, 1), len(shuffled) - 1)
        first_indices.append(shuffled[:split_count])
        second_indices.append(shuffled[split_count:])

    first_indices = np.sort(np.concatenate(first_indices))
    second_indices = np.sort(np.concatenate(second_indices))

    if len(first_indices) == 0 or len(second_indices) == 0:
        raise ValueError("Derived val/test split produced an empty dataset.")

    return first_indices, second_indices


def stratified_group_split_indices(y, group_ids, first_fraction, seed):
    if not 0.0 < first_fraction < 1.0:
        raise ValueError(f"first_fraction must be between 0 and 1, got {first_fraction}.")
    if len(y) != len(group_ids):
        raise ValueError(f"y and group_ids must have the same length, got {len(y)} and {len(group_ids)}.")

    rng = np.random.default_rng(seed)
    first_group_ids = []
    second_group_ids = []

    for label in np.unique(y):
        label_group_ids = np.unique(group_ids[y == label])
        shuffled = rng.permutation(label_group_ids)
        if len(shuffled) == 1:
            split_count = 1
        else:
            split_count = int(round(len(shuffled) * first_fraction))
            split_count = min(max(split_count, 1), len(shuffled) - 1)
        first_group_ids.append(shuffled[:split_count])
        second_group_ids.append(shuffled[split_count:])

    first_group_ids = np.concatenate(first_group_ids)
    second_group_ids = np.concatenate(second_group_ids)
    first_indices = np.where(np.isin(group_ids, first_group_ids))[0]
    second_indices = np.where(np.isin(group_ids, second_group_ids))[0]

    if len(first_indices) == 0 or len(second_indices) == 0:
        raise ValueError("Derived val/test split produced an empty dataset.")

    return np.sort(first_indices), np.sort(second_indices)


def derived_split_indices(y, first_fraction, seed, split_mode, group_ids=None):
    if split_mode == "stratified_windows":
        return stratified_split_indices(y, first_fraction=first_fraction, seed=seed)
    if split_mode == "stratified_segments":
        if group_ids is None:
            raise ValueError("group_ids are required for derived_split_mode='stratified_segments'.")
        return stratified_group_split_indices(
            y,
            group_ids=group_ids,
            first_fraction=first_fraction,
            seed=seed,
        )
    raise ValueError(f"Unsupported derived split mode: {split_mode!r}")


def stratified_split_dataset(X, y, first_fraction, seed):
    first_indices, second_indices = stratified_split_indices(y, first_fraction, seed)
    return X[first_indices], y[first_indices], X[second_indices], y[second_indices]


def stratified_train_val_test_split(X, y, val_fraction, seed):
    if not 0.0 < val_fraction < 0.5:
        raise ValueError(
            f"val_fraction must be between 0 and 0.5 when deriving both val and test, got {val_fraction}."
        )

    X_val, y_val, X_remaining, y_remaining = stratified_split_dataset(
        X,
        y,
        first_fraction=val_fraction,
        seed=seed,
    )
    test_fraction_of_remaining = val_fraction / (1.0 - val_fraction)
    X_test, y_test, X_train, y_train = stratified_split_dataset(
        X_remaining,
        y_remaining,
        first_fraction=test_fraction_of_remaining,
        seed=seed + 1,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


def predict_classes(logits, nonzero_threshold=None):
    if nonzero_threshold is None:
        return logits.argmax(dim=1)

    probs = torch.softmax(logits, dim=1)
    nonzero_probs, nonzero_idx = probs[:, 1:].max(dim=1)
    return torch.where(
        nonzero_probs >= nonzero_threshold,
        nonzero_idx + 1,
        torch.zeros_like(nonzero_idx),
    )


def evaluate(model, loader, criterion, device, nonzero_threshold=None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = predict_classes(logits, nonzero_threshold)
            total_loss += loss.item() * xb.size(0)
            total_correct += (preds == yb).sum().item()
            total_count += xb.size(0)
    return total_loss / total_count, total_correct / total_count


def compute_prediction_metrics(targets, preds, zero_division):
    present_labels = np.unique(targets)
    report_dict = classification_report(
        targets,
        preds,
        labels=present_labels,
        output_dict=True,
        zero_division=zero_division,
    )
    per_class_f1 = [report_dict[str(int(label))]["f1-score"] for label in present_labels]
    return {
        "accuracy": float((preds == targets).mean()),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "worst_class_f1": float(min(per_class_f1)),
    }


def evaluate_with_metrics(
    model,
    loader,
    criterion,
    device,
    zero_division,
    nonzero_threshold=None,
):
    model.eval()
    total_loss = 0.0
    total_count = 0
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            preds = predict_classes(logits, nonzero_threshold)
            total_loss += loss.item() * xb.size(0)
            total_count += xb.size(0)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(yb.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    metrics = compute_prediction_metrics(all_targets, all_preds, zero_division)
    return total_loss / total_count, metrics


def format_class_counts(y, raw_label_set, index_to_label):
    counts = np.bincount(y, minlength=len(raw_label_set)).astype(np.int64)
    return {int(index_to_label[idx]): int(count) for idx, count in enumerate(counts)}


def print_section(title):
    bar = "=" * 16
    print(f"\n{bar} {title} {bar}")


def plot_learning_curves(
    train_losses,
    train_accs,
    val_losses,
    val_accs,
    test_losses,
    test_accs,
):
    epochs = np.arange(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, train_losses, label="train_loss", linewidth=2)
    if not np.all(np.isnan(val_losses)):
        axes[0].plot(epochs, val_losses, label="val_loss", linewidth=2)
    if not np.all(np.isnan(test_losses)):
        axes[0].plot(epochs, test_losses, label="test_loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_accs, label="train_acc", linewidth=2)
    if not np.all(np.isnan(val_accs)):
        axes[1].plot(epochs, val_accs, label="val_acc", linewidth=2)
    if not np.all(np.isnan(test_accs)):
        axes[1].plot(epochs, test_accs, label="test_acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    plt.show()


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def validate_paired_filenames(x_filename, y_filename, name):
    if (x_filename is None) != (y_filename is None):
        raise ValueError(f"{name}_x_filename and {name}_y_filename must both be set or both be None.")


def validate_model_config(config: TrainingConfig):
    if not 0.0 < config.derived_val_fraction < 1.0:
        raise ValueError(
            f"derived_val_fraction must be between 0 and 1, got {config.derived_val_fraction}."
        )
    if config.model_type not in ALLOWED_MODEL_TYPES:
        raise ValueError(
            f"model_type must be one of {sorted(ALLOWED_MODEL_TYPES)}, "
            f"got {config.model_type!r}."
        )
    if (
        config.val_x_filename is None
        and config.test_x_filename is None
        and config.derived_val_fraction >= 0.5
    ):
        raise ValueError(
            "derived_val_fraction must be less than 0.5 when deriving both val and test "
            f"from the train files, got {config.derived_val_fraction}."
        )
    if len(config.conv_channels) == 0:
        raise ValueError("conv_channels must contain at least one layer.")
    if len(config.kernel_sizes) != len(config.conv_channels):
        raise ValueError(
            f"kernel_sizes must have the same length as conv_channels, got "
            f"{len(config.kernel_sizes)} vs {len(config.conv_channels)}."
        )
    if any(channel <= 0 for channel in config.conv_channels):
        raise ValueError(f"conv_channels must be positive, got {config.conv_channels}.")
    if any(kernel <= 0 or kernel % 2 == 0 for kernel in config.kernel_sizes):
        raise ValueError(f"kernel_sizes must be positive odd integers, got {config.kernel_sizes}.")
    if any(index < 0 or index >= len(config.conv_channels) for index in config.pool_after_layers):
        raise ValueError(
            f"pool_after_layers entries must be between 0 and {len(config.conv_channels) - 1}, "
            f"got {config.pool_after_layers}."
        )
    if len(set(config.pool_after_layers)) != len(config.pool_after_layers):
        raise ValueError(f"pool_after_layers must not contain duplicates, got {config.pool_after_layers}.")
    if config.classifier_hidden_dim <= 0:
        raise ValueError(f"classifier_hidden_dim must be positive, got {config.classifier_hidden_dim}.")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError(f"dropout must be in [0, 1), got {config.dropout}.")
    if config.early_stopping_patience is not None and config.early_stopping_patience <= 0:
        raise ValueError(
            f"early_stopping_patience must be positive or None, got {config.early_stopping_patience}."
        )
    if not 0.0 <= config.label_smoothing < 1.0:
        raise ValueError(f"label_smoothing must be in [0, 1), got {config.label_smoothing}.")
    if config.early_stopping_min_delta < 0.0:
        raise ValueError(
            f"early_stopping_min_delta must be non-negative, got {config.early_stopping_min_delta}."
        )
    if config.train_segment_min_gap_sec is not None and config.train_segment_min_gap_sec < 0.0:
        raise ValueError(
            "train_segment_min_gap_sec must be non-negative or None, got "
            f"{config.train_segment_min_gap_sec}."
        )
    if config.use_delta_features and config.append_delta_features:
        raise ValueError("use_delta_features and append_delta_features cannot both be enabled.")
    if config.selection_metric not in ALLOWED_SELECTION_METRICS:
        raise ValueError(
            f"selection_metric must be one of {sorted(ALLOWED_SELECTION_METRICS)}, "
            f"got {config.selection_metric!r}."
        )
    if config.derived_split_mode not in ALLOWED_DERIVED_SPLIT_MODES:
        raise ValueError(
            f"derived_split_mode must be one of {sorted(ALLOWED_DERIVED_SPLIT_MODES)}, "
            f"got {config.derived_split_mode!r}."
        )
    if config.train_sampling_mode not in ALLOWED_TRAIN_SAMPLING_MODES:
        raise ValueError(
            f"train_sampling_mode must be one of {sorted(ALLOWED_TRAIN_SAMPLING_MODES)}, "
            f"got {config.train_sampling_mode!r}."
        )
    if config.gru_hidden_dim <= 0:
        raise ValueError(f"gru_hidden_dim must be positive, got {config.gru_hidden_dim}.")
    if config.gru_num_layers <= 0:
        raise ValueError(f"gru_num_layers must be positive, got {config.gru_num_layers}.")


class PushCNN(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len,
        conv_channels,
        kernel_sizes,
        pool_after_layers,
        classifier_hidden_dim,
        dropout,
    ):
        super().__init__()
        layers = []
        prev_channels = in_channels
        pool_after_layers = set(pool_after_layers)
        for layer_idx, (out_channels, kernel_size) in enumerate(zip(conv_channels, kernel_sizes)):
            layers.append(
                nn.Conv1d(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            if layer_idx in pool_after_layers:
                layers.append(nn.MaxPool1d(kernel_size=2))
            prev_channels = out_channels
        self.features = nn.Sequential(*layers)
        with torch.no_grad():
            flat_dim = self.features(torch.zeros(1, in_channels, seq_len)).numel()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PushGRU(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        seq_len,
        hidden_dim,
        num_layers,
        bidirectional,
        classifier_hidden_dim,
        dropout,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1
        self.gru = nn.GRU(
            input_size=in_channels,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * self.num_directions, classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, num_classes),
        )

    def forward(self, x):
        # Deployment/runtime keeps the existing NCT input layout; recurrent layers use NTF.
        x = x.transpose(1, 2)
        _, h_n = self.gru(x)
        h_n = h_n.view(self.num_layers, self.num_directions, x.size(0), self.hidden_dim)
        last_layer_h = h_n[-1].transpose(0, 1).reshape(x.size(0), -1)
        return self.classifier(last_layer_h)


def build_model(config: TrainingConfig, in_channels, num_classes, seq_len):
    if config.model_type == "cnn":
        return PushCNN(
            in_channels=in_channels,
            num_classes=num_classes,
            seq_len=seq_len,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            pool_after_layers=config.pool_after_layers,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout=config.dropout,
        )
    if config.model_type == "gru":
        return PushGRU(
            in_channels=in_channels,
            num_classes=num_classes,
            seq_len=seq_len,
            hidden_dim=config.gru_hidden_dim,
            num_layers=config.gru_num_layers,
            bidirectional=config.gru_bidirectional,
            classifier_hidden_dim=config.classifier_hidden_dim,
            dropout=config.dropout,
        )
    raise ValueError(f"Unsupported model_type: {config.model_type!r}")


def architecture_metadata(config: TrainingConfig):
    if config.model_type == "cnn":
        return {
            "conv_channels": list(config.conv_channels),
            "kernel_sizes": list(config.kernel_sizes),
            "pool_after_layers": list(config.pool_after_layers),
            "classifier_hidden_dim": config.classifier_hidden_dim,
            "dropout": config.dropout,
        }
    if config.model_type == "gru":
        return {
            "gru_hidden_dim": config.gru_hidden_dim,
            "gru_num_layers": config.gru_num_layers,
            "gru_bidirectional": str(config.gru_bidirectional).lower(),
            "classifier_hidden_dim": config.classifier_hidden_dim,
            "dropout": config.dropout,
        }
    raise ValueError(f"Unsupported model_type: {config.model_type!r}")


def format_model_description(config: TrainingConfig):
    if config.model_type == "cnn":
        return (
            f"type=cnn | conv_channels={list(config.conv_channels)} | "
            f"kernel_sizes={list(config.kernel_sizes)} | "
            f"pool_after_layers={list(config.pool_after_layers)} | "
            f"classifier_hidden_dim={config.classifier_hidden_dim} | "
            f"dropout={config.dropout}"
        )
    if config.model_type == "gru":
        return (
            f"type=gru | hidden_dim={config.gru_hidden_dim} | "
            f"num_layers={config.gru_num_layers} | "
            f"bidirectional={config.gru_bidirectional} | "
            f"classifier_hidden_dim={config.classifier_hidden_dim} | "
            f"dropout={config.dropout}"
        )
    raise ValueError(f"Unsupported model_type: {config.model_type!r}")


def run_training(config: TrainingConfig):
    if config.seed is not None:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    validate_paired_filenames(config.val_x_filename, config.val_y_filename, "val")
    validate_paired_filenames(config.test_x_filename, config.test_y_filename, "test")
    validate_model_config(config)

    if config.selection_split not in {"val", "test"}:
        raise ValueError(f"selection_split must be 'val' or 'test', got {config.selection_split!r}.")

    X_train, y_train, train_source_ids, train_group_ids, original_train_samples, kept_train_samples = (
        load_and_concat_train_datasets(config)
    )

    X_val = y_val = None
    X_test = y_test = None
    val_source_label = config.val_x_filename
    test_source_label = config.test_x_filename

    if config.val_x_filename is not None:
        X_val_source, y_val_source = load_dataset_pair(
            config.data_dir,
            config.val_x_filename,
            config.val_y_filename,
        )
        if config.test_x_filename is None:
            val_group_ids = None
            if config.derived_split_mode == "stratified_segments":
                seg_filename = sidecar_filename_for_x(config.val_x_filename, "seg_")
                val_group_ids = np.load(config.data_dir / seg_filename).astype(np.int64)
            val_indices, test_indices = derived_split_indices(
                y_val_source,
                first_fraction=config.derived_val_fraction,
                seed=config.split_seed,
                split_mode=config.derived_split_mode,
                group_ids=val_group_ids,
            )
            X_val, y_val = X_val_source[val_indices], y_val_source[val_indices]
            X_test, y_test = X_val_source[test_indices], y_val_source[test_indices]
            test_source_label = f"{config.val_x_filename} (derived split)"
        else:
            X_val, y_val = X_val_source, y_val_source
    elif config.test_x_filename is not None:
        val_indices, train_indices = derived_split_indices(
            y_train,
            first_fraction=config.derived_val_fraction,
            seed=config.split_seed,
            split_mode=config.derived_split_mode,
            group_ids=train_group_ids,
        )
        X_val, y_val = X_train[val_indices], y_train[val_indices]
        X_train, y_train = X_train[train_indices], y_train[train_indices]
        train_source_ids = train_source_ids[train_indices]
        if train_group_ids is not None:
            train_group_ids = train_group_ids[train_indices]
        val_source_label = "train files (derived split)"

    if config.test_x_filename is not None:
        X_test, y_test = load_dataset_pair(
            config.data_dir,
            config.test_x_filename,
            config.test_y_filename,
        )

    if X_test is None or y_test is None:
        if X_val is None or y_val is None:
            val_indices, remaining_indices = derived_split_indices(
                y_train,
                first_fraction=config.derived_val_fraction,
                seed=config.split_seed,
                split_mode=config.derived_split_mode,
                group_ids=train_group_ids,
            )
            X_val, y_val = X_train[val_indices], y_train[val_indices]
            X_remaining, y_remaining = X_train[remaining_indices], y_train[remaining_indices]
            remaining_source_ids = train_source_ids[remaining_indices]
            remaining_group_ids = None if train_group_ids is None else train_group_ids[remaining_indices]
            test_fraction_of_remaining = config.derived_val_fraction / (1.0 - config.derived_val_fraction)
            test_indices, train_indices = derived_split_indices(
                y_remaining,
                first_fraction=test_fraction_of_remaining,
                seed=config.split_seed + 1,
                split_mode=config.derived_split_mode,
                group_ids=remaining_group_ids,
            )
            X_test, y_test = X_remaining[test_indices], y_remaining[test_indices]
            X_train, y_train = X_remaining[train_indices], y_remaining[train_indices]
            train_source_ids = remaining_source_ids[train_indices]
            if remaining_group_ids is not None:
                train_group_ids = remaining_group_ids[train_indices]
            val_source_label = "train files (derived split)"
            test_source_label = "train files (derived split)"
    if config.selection_split == "val" and (X_val is None or y_val is None):
        raise ValueError("selection_split='val' requires a val dataset.")

    label_arrays = [y_train, y_test]
    if y_val is not None:
        label_arrays.append(y_val)
    raw_label_set, label_to_index, index_to_label = infer_label_set(*label_arrays)

    y_train = remap_labels(y_train, label_to_index)
    if y_val is not None:
        y_val = remap_labels(y_val, label_to_index)
    y_test = remap_labels(y_test, label_to_index)

    X_train = select_features(X_train, config.selected_features)
    if X_val is not None:
        X_val = select_features(X_val, config.selected_features)
    X_test = select_features(X_test, config.selected_features)

    _, T, F = X_train.shape
    datasets_to_check = [("test", X_test)]
    if X_val is not None:
        datasets_to_check.append(("val", X_val))
    for split_name, X_split in datasets_to_check:
        _, T_split, F_split = X_split.shape
        if T != T_split or F != F_split:
            raise ValueError(
                f"Train/{split_name} shape mismatch: "
                f"train (T={T}, F={F}) vs {split_name} (T={T_split}, F={F_split})"
            )

    num_classes = len(raw_label_set)
    X_train = X_train.astype(np.float32)
    if X_val is not None:
        X_val = X_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.int64)
    if y_val is not None:
        y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    delta_feature_mode = "off"
    if config.use_delta_features:
        # Subtract the first timestep from every timestep in each window.
        # X shape: (N, T, F). X[:, 0:1, :] broadcasts over T.
        # NOTE: only useful if windows capture push onset. For sustained-push
        # windows the delta is near-zero for all classes → not recommended.
        X_train = X_train - X_train[:, 0:1, :]
        if X_val is not None:
            X_val = X_val - X_val[:, 0:1, :]
        X_test = X_test - X_test[:, 0:1, :]
        delta_feature_mode = "replace"
    elif config.append_delta_features:
        X_train = np.concatenate([X_train, X_train - X_train[:, 0:1, :]], axis=2)
        if X_val is not None:
            X_val = np.concatenate([X_val, X_val - X_val[:, 0:1, :]], axis=2)
        X_test = np.concatenate([X_test, X_test - X_test[:, 0:1, :]], axis=2)
        delta_feature_mode = "append"

    # Gravity compensation: subtract expected neutral current predicted from arm
    # angles using a linear fit on neutral training samples.  Residual current =
    # actual_current − gravity_component is configuration-invariant:  neutral → 0,
    # up push → negative, down push → positive regardless of arm height.
    gravity_comp_W = None
    gravity_comp_b = None
    if config.use_gravity_comp:
        required_feats = {"arm_angles", "arm_currents"}
        if not required_feats.issubset(set(config.selected_features)):
            raise ValueError(
                f"use_gravity_comp requires 'arm_angles' and 'arm_currents' in "
                f"selected_features, got {config.selected_features}."
            )
        # Compute local feature offsets within the selected feature block.
        offset = 0
        angle_start = angle_end = current_start = current_end = None
        for feat in config.selected_features:
            w = FEATURE_SLICES[feat][1] - FEATURE_SLICES[feat][0]
            if feat == "arm_angles":
                angle_start, angle_end = offset, offset + w
            elif feat == "arm_currents":
                current_start, current_end = offset, offset + w
            offset += w

        # Fit sinusoidal model: arm_currents ≈ [sin(θ), cos(θ), 1] @ W, using only
        # neutral training samples.  sin/cos basis is physically correct (gravity
        # torques follow sin(joint_angle)) and avoids the linear-extrapolation error
        # that causes the model to over-predict expected current at extreme angles.
        neutral_mask = y_train == 0
        if neutral_mask.sum() < 20:
            raise RuntimeError(
                f"Only {neutral_mask.sum()} neutral training samples — too few to "
                "fit gravity compensation. Need ≥20."
            )
        X_neutral = X_train[neutral_mask]
        n_ang = angle_end - angle_start
        angles_flat = X_neutral[:, :, angle_start:angle_end].reshape(-1, n_ang)
        currents_flat = X_neutral[:, :, current_start:current_end].reshape(-1, current_end - current_start)
        angles_rad = angles_flat * np.float32(np.pi / 180.0)
        A = np.hstack([
            np.sin(angles_rad),
            np.cos(angles_rad),
            np.ones((len(angles_flat), 1), dtype=np.float32),
        ]).astype(np.float32)
        W_b, _, _, _ = np.linalg.lstsq(A, currents_flat, rcond=None)
        # W_b shape: (2*n_ang+1, n_currents) — bias is the last row
        gravity_comp_W = W_b.astype(np.float32)
        gravity_comp_b = None  # absorbed into last row of gravity_comp_W

        def _apply_gcomp(X_arr):
            ang = X_arr[:, :, angle_start:angle_end].reshape(-1, n_ang)
            ang_rad = ang * np.float32(np.pi / 180.0)
            basis = np.hstack([
                np.sin(ang_rad),
                np.cos(ang_rad),
                np.ones((len(ang), 1), dtype=np.float32),
            ]).astype(np.float32)
            pred = (basis @ gravity_comp_W).reshape(X_arr.shape[0], X_arr.shape[1], -1)
            out = X_arr.copy()
            out[:, :, current_start:current_end] -= pred
            return out

        X_train = _apply_gcomp(X_train)
        if X_val is not None:
            X_val = _apply_gcomp(X_val)
        X_test = _apply_gcomp(X_test)
        print(
            f"Gravity comp (sinusoidal): fit on {neutral_mask.sum()} neutral samples | "
            f"W shape={gravity_comp_W.shape} (sin+cos+bias basis)"
        )

    _, T, F = X_train.shape
    x_mean = X_train.mean(axis=(0, 1), keepdims=True)
    x_std = X_train.std(axis=(0, 1), keepdims=True)
    x_std[x_std < 1e-6] = 1.0

    X_train = np.transpose((X_train - x_mean) / x_std, (0, 2, 1))
    if X_val is not None:
        X_val = np.transpose((X_val - x_mean) / x_std, (0, 2, 1))
    X_test = np.transpose((X_test - x_mean) / x_std, (0, 2, 1))

    _dl_generator = torch.Generator()
    if config.seed is not None:
        _dl_generator.manual_seed(config.seed)
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    train_sampler = None
    train_shuffle = True
    train_sampling_desc = config.train_sampling_mode
    if config.train_sampling_mode == "uniform_files":
        file_counts = np.bincount(train_source_ids, minlength=len(config.train_x_filenames)).astype(np.float64)
        sample_weights = np.array(
            [1.0 / max(file_counts[source_idx], 1.0) for source_idx in train_source_ids],
            dtype=np.float64,
        )
        sample_weights *= len(sample_weights) / sample_weights.sum()
        train_sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights),
            replacement=True,
            generator=_dl_generator,
        )
        train_shuffle = False
        train_sampling_desc += f" (file_counts={file_counts.astype(int).tolist()})"
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        generator=_dl_generator,
    )
    val_loader = None
    if X_val is not None and y_val is not None:
        val_loader = DataLoader(
            TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=config.batch_size,
            shuffle=False,
        )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
        batch_size=config.batch_size,
        shuffle=False,
    )

    device = resolve_device()
    model = build_model(
        config=config,
        in_channels=F,
        num_classes=num_classes,
        seq_len=T,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    if config.manual_class_loss_weights is not None:
        if len(config.manual_class_loss_weights) != num_classes:
            raise ValueError(
                f"manual_class_loss_weights must have length {num_classes}, "
                f"got {len(config.manual_class_loss_weights)}."
            )
        class_weights = torch.tensor(config.manual_class_loss_weights, dtype=torch.float32, device=device)
    else:
        class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
        class_weights = torch.tensor(class_weights / class_weights.mean(), dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        if config.use_lr_scheduler
        else None
    )

    print(f"Device={device} | features={config.selected_features} | raw_labels={list(raw_label_set)}")
    print_section("Run Setup")
    print(
        f"Input: timesteps={T}, features={F} | train_files={len(config.train_x_filenames)} | "
        f"val_file={val_source_label} | test_file={test_source_label}"
    )
    print(
        "Model: "
        f"{format_model_description(config)}"
    )
    print(
        "Early stopping: "
        f"patience={config.early_stopping_patience} | "
        f"min_delta={config.early_stopping_min_delta}"
    )
    print(
        f"Optimizer: lr={config.learning_rate} | weight_decay={config.weight_decay} | "
        f"label_smoothing={config.label_smoothing} | "
        f"lr_scheduler={'ReduceLROnPlateau(factor=0.5,patience=5)' if config.use_lr_scheduler else 'none'} | "
        f"seed={config.seed}"
    )
    print(f"Selection: split={config.selection_split} | metric={config.selection_metric}")
    print(f"Derived split mode: {config.derived_split_mode}")
    print(f"Delta feature mode: {delta_feature_mode}")
    print(
        "Train segment subsampling: "
        f"min_gap_sec={config.train_segment_min_gap_sec} | "
        f"kept={kept_train_samples}/{original_train_samples}"
    )
    print(f"Train sampling: {train_sampling_desc}")
    val_count = len(y_val) if y_val is not None else 0
    print(f"Samples: train={len(y_train)}, val={val_count}, test={len(y_test)}")

    class_count_parts = [
        f"train={format_class_counts(y_train, raw_label_set, index_to_label)}",
    ]
    if y_val is not None:
        class_count_parts.append(f"val={format_class_counts(y_val, raw_label_set, index_to_label)}")
    class_count_parts.append(f"test={format_class_counts(y_test, raw_label_set, index_to_label)}")
    print("Class counts: " + " | ".join(class_count_parts))

    print(f"Loss weights: {class_weights.detach().cpu().numpy().tolist()}")
    print(f"Thresholded inference: {config.nonzero_prediction_threshold}")

    best_selection_score = -1.0
    best_selection_metrics = None
    best_state = None
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    test_losses = []
    test_accs = []
    best_val_loss = np.inf
    early_stopping_wait = 0

    if config.early_stopping_patience is not None and val_loader is None:
        print("Early stopping requested, but no validation split is available; disabling early stopping.")

    print_section("Training")
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            preds = predict_classes(logits)
            train_loss += loss.item() * xb.size(0)
            train_correct += (preds == yb).sum().item()
            train_count += xb.size(0)

        train_loss /= train_count
        train_acc = train_correct / train_count
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        epoch_metrics = [
            f"Epoch {epoch+1:02d}/{config.num_epochs}",
            f"train_loss={train_loss:.4f}",
            f"train_acc={train_acc:.4f}",
        ]

        val_acc = None
        val_loss = None
        val_metrics = None
        if val_loader is not None:
            val_loss, val_metrics = evaluate_with_metrics(
                model,
                val_loader,
                criterion,
                device,
                zero_division=config.zero_division,
            )
            val_acc = val_metrics["accuracy"]
            epoch_metrics.extend([f"val_loss={val_loss:.4f}", f"val_acc={val_acc:.4f}"])
            if config.selection_metric != "accuracy":
                epoch_metrics.append(f"val_{config.selection_metric}={val_metrics[config.selection_metric]:.4f}")
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            if scheduler is not None:
                prev_lr = optimizer.param_groups[0]["lr"]
                scheduler.step(val_loss)
                new_lr = optimizer.param_groups[0]["lr"]
                if new_lr != prev_lr:
                    epoch_metrics.append(f"lr={new_lr:.2e}")
        else:
            val_losses.append(np.nan)
            val_accs.append(np.nan)

        if config.selection_split == "val":
            selection_metrics = val_metrics
            test_losses.append(np.nan)
            test_accs.append(np.nan)
        else:
            test_loss, test_metrics = evaluate_with_metrics(
                model,
                test_loader,
                criterion,
                device,
                zero_division=config.zero_division,
            )
            test_acc = test_metrics["accuracy"]
            selection_metrics = test_metrics
            epoch_metrics.extend([f"test_loss={test_loss:.4f}", f"test_acc={test_acc:.4f}"])
            if config.selection_metric != "accuracy":
                epoch_metrics.append(
                    f"test_{config.selection_metric}={test_metrics[config.selection_metric]:.4f}"
                )
            test_losses.append(test_loss)
            test_accs.append(test_acc)

        selection_score = selection_metrics[config.selection_metric]

        if config.early_stopping_patience is not None and val_loss is not None:
            if val_loss < best_val_loss - config.early_stopping_min_delta:
                best_val_loss = val_loss
                early_stopping_wait = 0
            else:
                early_stopping_wait += 1
            epoch_metrics.append(
                f"early_stop_wait={early_stopping_wait}/{config.early_stopping_patience}"
            )

        epoch_metrics.append(
            f"best_{config.selection_split}_{config.selection_metric}="
            f"{max(best_selection_score, selection_score):.4f}"
        )
        print(" | ".join(epoch_metrics))

        if selection_score > best_selection_score:
            best_selection_score = selection_score
            best_selection_metrics = dict(selection_metrics)
            best_state = {
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "num_classes": num_classes,
                "num_features": F,
                "num_timesteps": T,
                "raw_label_set": raw_label_set,
                "label_to_index": label_to_index,
                "index_to_label": index_to_label,
                "x_mean": x_mean,
                "x_std": x_std,
                "model_type": config.model_type,
                "conv_channels": config.conv_channels,
                "kernel_sizes": config.kernel_sizes,
                "pool_after_layers": config.pool_after_layers,
                "gru_hidden_dim": config.gru_hidden_dim,
                "gru_num_layers": config.gru_num_layers,
                "gru_bidirectional": config.gru_bidirectional,
                "classifier_hidden_dim": config.classifier_hidden_dim,
                "dropout": config.dropout,
                "train_segment_min_gap_sec": config.train_segment_min_gap_sec,
                "derived_split_mode": config.derived_split_mode,
                "delta_feature_mode": delta_feature_mode,
                "selection_metric": config.selection_metric,
                f"best_{config.selection_split}_{config.selection_metric}": best_selection_score,
                f"best_{config.selection_split}_acc": best_selection_metrics["accuracy"],
            }

        if (
            config.early_stopping_patience is not None
            and val_loss is not None
            and early_stopping_wait >= config.early_stopping_patience
        ):
            print(
                f"Early stopping triggered at epoch {epoch + 1}: "
                f"val_loss did not improve by more than {config.early_stopping_min_delta} "
                f"for {config.early_stopping_patience} consecutive epochs."
            )
            break

    if best_state is None:
        raise RuntimeError("Training finished without producing a best checkpoint.")
    if best_selection_metrics is None:
        raise RuntimeError("Training finished without recording best selection metrics.")

    out_dir = config.data_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = out_dir / f"{config.artifact_stem}.pt"
    onnx_path = out_dir / f"{config.artifact_stem}.onnx"
    deploy_yaml_path = out_dir / f"{config.artifact_stem}.deploy.yaml"

    if config.export_artifacts:
        torch.save(best_state, checkpoint_path)
        print("Saved best model to:", checkpoint_path)

        model_cpu = build_model(
            config=config,
            in_channels=F,
            num_classes=num_classes,
            seq_len=T,
        ).cpu()
        model_cpu.load_state_dict(best_state["model_state_dict"])
        model_cpu.eval()
        torch.onnx.export(
            model_cpu,
            torch.zeros(1, F, T, dtype=torch.float32),
            onnx_path,
            input_names=["input"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch_size")}},
            dynamo=True,
            opset_version=config.onnx_opset_version,
            verbose=False,
        )
        print("Saved ONNX model to:", onnx_path)

        write_deploy_yaml(
            deploy_yaml_path,
            onnx_path.name,
            F,
            T,
            raw_label_set,
            label_to_index,
            index_to_label,
            x_mean,
            x_std,
            config.selected_features,
            config.onnx_opset_version,
            delta_feature_mode,
            nonzero_prediction_threshold=config.nonzero_prediction_threshold,
            sampling_hz=config.sampling_hz,
            gravity_comp_W=gravity_comp_W,
            gravity_comp_b=gravity_comp_b,
            model_type=config.model_type,
            architecture_metadata=architecture_metadata(config),
        )
        print("Saved deploy metadata to:", deploy_yaml_path)
    else:
        print("Artifact export disabled; skipping .pt, .onnx, and deploy yaml output.")

    model.load_state_dict(best_state["model_state_dict"])
    model.eval()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    _, thresholded_test_acc = evaluate(
        model,
        test_loader,
        criterion,
        device,
        nonzero_threshold=config.nonzero_prediction_threshold,
    )

    print_section("Final Evaluation")
    print(
        f"Best {config.selection_split}_{config.selection_metric}={best_selection_score:.4f} | "
        f"best_{config.selection_split}_acc={best_selection_metrics['accuracy']:.4f} | "
        f"final_test_loss={test_loss:.4f} | final_test_acc={test_acc:.4f} | "
        f"thresholded_test_acc={thresholded_test_acc:.4f}"
    )

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            all_preds.append(
                predict_classes(logits, config.nonzero_prediction_threshold).cpu().numpy()
            )
            all_targets.append(yb.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_preds_raw = np.array([index_to_label[int(label)] for label in all_preds], dtype=np.int64)
    all_targets_raw = np.array([index_to_label[int(label)] for label in all_targets], dtype=np.int64)
    present_labels_raw = np.unique(all_targets_raw)

    print_section("Confusion Matrix")
    print("Confusion matrix in raw labels (rows=true, cols=pred):")
    print(confusion_matrix(all_targets_raw, all_preds_raw, labels=np.array(raw_label_set)))

    report_dict = classification_report(
        all_targets_raw,
        all_preds_raw,
        labels=present_labels_raw,
        output_dict=True,
        zero_division=config.zero_division,
    )

    print_section("Classification Report")
    print("Classification report in raw labels (present test classes only):")
    print(
        classification_report(
            all_targets_raw,
            all_preds_raw,
            labels=present_labels_raw,
            digits=4,
            zero_division=config.zero_division,
        )
    )

    per_class_metrics = {
        int(label): {
            "precision": float(report_dict[str(int(label))]["precision"]),
            "recall": float(report_dict[str(int(label))]["recall"]),
            "f1-score": float(report_dict[str(int(label))]["f1-score"]),
            "support": int(report_dict[str(int(label))]["support"]),
        }
        for label in present_labels_raw
    }

    if config.show_learning_curves:
        plot_learning_curves(
            train_losses,
            train_accs,
            val_losses,
            val_accs,
            test_losses,
            test_accs,
        )

    return {
        "selection_metric": config.selection_metric,
        "best_selection_score": float(best_selection_score),
        "best_selection_acc": float(best_selection_metrics["accuracy"]),
        "final_test_loss": float(test_loss),
        "final_test_acc": float(test_acc),
        "thresholded_test_acc": float(thresholded_test_acc),
        "raw_label_set": tuple(int(label) for label in raw_label_set),
        "train_samples": int(len(y_train)),
        "val_samples": int(0 if y_val is None else len(y_val)),
        "test_samples": int(len(y_test)),
        "per_class_metrics": per_class_metrics,
        "macro_precision": float(report_dict["macro avg"]["precision"]),
        "macro_recall": float(report_dict["macro avg"]["recall"]),
        "macro_f1": float(report_dict["macro avg"]["f1-score"]),
        "worst_class_f1": min(metrics["f1-score"] for metrics in per_class_metrics.values()),
    }
