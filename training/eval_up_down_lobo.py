#!/usr/bin/env python3

import argparse
from pathlib import Path

from train_cnn_common import TrainingConfig, run_training


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data" / "processed_data" / "up_down"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tag",
        default="_w300_e060_hpure",
        help="Dataset suffix to evaluate, e.g. '_w300_e045' or '_w300_e060'.",
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 6],
        help="Bag ids to use for leave-one-bag-out evaluation.",
    )
    parser.add_argument(
        "--derived-val-fraction",
        type=float,
        default=0.2,
        help="Fraction of the non-held-out training bags reserved for validation.",
    )
    parser.add_argument(
        "--train-sampling-mode",
        default="uniform_files",
        choices=["shuffle", "uniform_files"],
        help="Training sampler mode.",
    )
    parser.add_argument(
        "--derived-split-mode",
        default="stratified_windows",
        choices=["stratified_windows", "stratified_segments"],
        help="How to derive validation folds from the training bags.",
    )
    parser.add_argument(
        "--delta-mode",
        default="append",
        choices=["replace", "append", "none"],
        help=(
            "How to use delta features. "
            "'replace': feed only change-from-window-start (pose-invariant, default). "
            "'append': feed raw features + deltas alongside each other. "
            "'none': feed only raw absolute features."
        ),
    )
    return parser.parse_args()


def make_config(
    tag: str,
    train_bags: list[int],
    test_bag: int,
    derived_val_fraction: float,
    train_sampling_mode: str,
    derived_split_mode: str,
    delta_mode: str,
) -> TrainingConfig:
    use_delta_features = delta_mode == "replace"
    append_delta_features = delta_mode == "append"
    return TrainingConfig(
        data_dir=DATA_DIR,
        train_x_filenames=[f"X_ud_{bag}{tag}.npy" for bag in train_bags],
        train_y_filenames=[f"y_ud_{bag}{tag}.npy" for bag in train_bags],
        val_x_filename=None,
        val_y_filename=None,
        test_x_filename=f"X_ud_{test_bag}{tag}.npy",
        test_y_filename=f"y_ud_{test_bag}{tag}.npy",
        selected_features=["arm_angles", "arm_currents", "ff", "accel", "dq"],
        artifact_stem=f"intent_up_down{tag}_lobo_test{test_bag}",
        derived_val_fraction=derived_val_fraction,
        derived_split_mode=derived_split_mode,
        split_seed=42,
        export_artifacts=False,
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
        use_delta_features=use_delta_features,
        append_delta_features=append_delta_features,
        use_gravity_comp=False,
        train_segment_min_gap_sec=None,
        train_sampling_mode=train_sampling_mode,
    )


def main():
    args = parse_args()
    results = []

    for test_bag in args.bags:
        train_bags = [bag for bag in args.bags if bag != test_bag]
        print(f"\n===== tag={args.tag} | train={train_bags} | test={test_bag} =====")
        result = run_training(
            make_config(
                tag=args.tag,
                train_bags=train_bags,
                test_bag=test_bag,
                derived_val_fraction=args.derived_val_fraction,
                train_sampling_mode=args.train_sampling_mode,
                derived_split_mode=args.derived_split_mode,
                delta_mode=args.delta_mode,
            )
        )
        results.append((test_bag, result))
        per_class = result["per_class_metrics"]
        class_f1_str = "  ".join(
            f"label{label}: f1={m['f1-score']:.3f} p={m['precision']:.3f} r={m['recall']:.3f}"
            for label, m in sorted(per_class.items())
        )
        print(
            f"--> test_bag={test_bag} | acc={result['final_test_acc']:.4f}"
            f" | macro_f1={result['macro_f1']:.4f}"
            f" | worst_class_f1={result['worst_class_f1']:.4f}"
            f"\n    {class_f1_str}"
        )

    avg_test_acc = sum(result["final_test_acc"] for _, result in results) / len(results)
    avg_macro_f1 = sum(result["macro_f1"] for _, result in results) / len(results)
    avg_worst_class_f1 = sum(result["worst_class_f1"] for _, result in results) / len(results)
    min_test_acc = min(result["final_test_acc"] for _, result in results)
    min_macro_f1 = min(result["macro_f1"] for _, result in results)
    min_worst_class_f1 = min(result["worst_class_f1"] for _, result in results)

    print("\n===== LOBO Summary =====")
    print(f"tag={args.tag}")
    print(f"bags={args.bags}")
    print(f"derived_val_fraction={args.derived_val_fraction}")
    print(f"derived_split_mode={args.derived_split_mode}")
    print(f"train_sampling_mode={args.train_sampling_mode}")
    print(f"delta_mode={args.delta_mode}")
    print(f"avg_test_acc={avg_test_acc:.4f}")
    print(f"avg_macro_f1={avg_macro_f1:.4f}")
    print(f"avg_worst_class_f1={avg_worst_class_f1:.4f}")
    print(f"min_test_acc={min_test_acc:.4f}")
    print(f"min_macro_f1={min_macro_f1:.4f}")
    print(f"min_worst_class_f1={min_worst_class_f1:.4f}")


if __name__ == "__main__":
    main()
