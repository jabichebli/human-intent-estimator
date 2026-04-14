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
        help="Dataset suffix to evaluate, e.g. '_w300', '_w300_e045', '_w300_e060'.",
    )
    parser.add_argument(
        "--append-delta-features",
        dest="append_delta_features",
        action="store_true",
        default=True,
        help="Append per-window delta features alongside the raw features.",
    )
    parser.add_argument(
        "--no-append-delta-features",
        dest="append_delta_features",
        action="store_false",
        help="Use only the raw selected features.",
    )
    parser.add_argument(
        "--train-bags",
        nargs="+",
        type=int,
        default=[2, 3, 4],
        help="Bag ids used for training.",
    )
    parser.add_argument(
        "--eval-bags",
        nargs=2,
        type=int,
        default=[5, 6],
        help="Two held-out bag ids used for swapped val/test evaluation.",
    )
    return parser.parse_args()


def make_config(
    tag: str,
    train_bags: list[int],
    val_bag: int,
    test_bag: int,
    append_delta_features: bool,
) -> TrainingConfig:
    return TrainingConfig(
        data_dir=DATA_DIR,
        train_x_filenames=[f"X_ud_{bag}{tag}.npy" for bag in train_bags],
        train_y_filenames=[f"y_ud_{bag}{tag}.npy" for bag in train_bags],
        val_x_filename=f"X_ud_{val_bag}{tag}.npy",
        val_y_filename=f"y_ud_{val_bag}{tag}.npy",
        test_x_filename=f"X_ud_{test_bag}{tag}.npy",
        test_y_filename=f"y_ud_{test_bag}{tag}.npy",
        selected_features=["arm_angles", "arm_currents", "ff", "accel"],
        artifact_stem=f"intent_up_down{tag}_val{val_bag}_test{test_bag}",
        derived_val_fraction=0.5,
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
        use_delta_features=False,
        append_delta_features=append_delta_features,
        use_gravity_comp=False,
        train_segment_min_gap_sec=None,
        train_sampling_mode="uniform_files",
    )


def main():
    args = parse_args()
    eval_a, eval_b = args.eval_bags

    results = []
    for val_bag, test_bag in [(eval_a, eval_b), (eval_b, eval_a)]:
        print(f"\n===== tag={args.tag} | val={val_bag} | test={test_bag} =====")
        result = run_training(
            make_config(
                args.tag,
                args.train_bags,
                val_bag,
                test_bag,
                args.append_delta_features,
            )
        )
        results.append((val_bag, test_bag, result))

    avg_test_acc = sum(result["final_test_acc"] for _, _, result in results) / len(results)
    avg_macro_f1 = sum(result["macro_f1"] for _, _, result in results) / len(results)
    avg_worst_class_f1 = sum(result["worst_class_f1"] for _, _, result in results) / len(results)

    print("\n===== Whole-Bag Summary =====")
    print(f"tag={args.tag}")
    print(f"train_bags={args.train_bags}")
    print(f"eval_bags={args.eval_bags}")
    print(f"append_delta_features={args.append_delta_features}")
    print(f"avg_test_acc={avg_test_acc:.4f}")
    print(f"avg_macro_f1={avg_macro_f1:.4f}")
    print(f"avg_worst_class_f1={avg_worst_class_f1:.4f}")


if __name__ == "__main__":
    main()
