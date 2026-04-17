#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SUMMARY_FIELDS = [
    "avg_macro_f1",
    "min_macro_f1",
    "avg_worst_class_f1",
    "min_worst_class_f1",
    "avg_test_acc",
    "min_test_acc",
    "avg_thresholded_test_acc",
    "min_thresholded_test_acc",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print compact summaries from grouped LOBO CSV files."
    )
    parser.add_argument("csv_files", nargs="+", type=Path)
    parser.add_argument(
        "--worst",
        type=int,
        default=4,
        help="Number of worst held-out groups to print per CSV.",
    )
    return parser.parse_args()


def as_float(row, key):
    value = row.get(key, "")
    if value == "":
        return None
    return float(value)


def load_rows(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def format_value(value):
    if value is None:
        return ""
    return f"{value:.4f}"


def label_f1_fields(rows):
    fields = set()
    for row in rows:
        for field in row:
            if field.startswith("label_") and field.endswith("_f1"):
                fields.add(field)
    return sorted(fields, key=lambda field: int(field.split("_")[1]))


def main():
    args = parse_args()
    print("===== Summary Ranking =====")
    summaries = []
    all_rows = {}
    for path in args.csv_files:
        rows = load_rows(path)
        all_rows[path] = rows
        summary = next((row for row in rows if row.get("row_type") == "summary"), None)
        if summary is None:
            print(f"{path}: missing summary row")
            continue
        summaries.append((path, summary))

    summaries.sort(
        key=lambda item: (
            as_float(item[1], "avg_macro_f1") or -1.0,
            as_float(item[1], "min_macro_f1") or -1.0,
        ),
        reverse=True,
    )

    header = ["csv", *SUMMARY_FIELDS]
    print(",".join(header))
    for path, summary in summaries:
        values = [path.name]
        values.extend(format_value(as_float(summary, field)) for field in SUMMARY_FIELDS)
        print(",".join(values))

    print("\n===== Worst Held-Out Groups =====")
    for path, _summary in summaries:
        group_rows = [
            row for row in all_rows[path]
            if row.get("row_type") in {"group", "bag"}
        ]
        group_rows.sort(key=lambda row: as_float(row, "macro_f1") or -1.0)
        per_label_fields = label_f1_fields(group_rows)
        print(f"\n{path.name}")
        print(",".join(["test_bag", "macro_f1", "worst_class_f1", *per_label_fields]))
        for row in group_rows[: args.worst]:
            print(
                ",".join([
                    row.get("test_bag", ""),
                    format_value(as_float(row, "macro_f1")),
                    format_value(as_float(row, "worst_class_f1")),
                    *[format_value(as_float(row, field)) for field in per_label_fields],
                ])
            )


if __name__ == "__main__":
    main()
