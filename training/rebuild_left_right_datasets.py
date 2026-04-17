#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
import subprocess
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_BAGS_DIR = ROOT_DIR / "bag_data" / "raw_bag"
LEFT_RIGHT_DIR = ROOT_DIR / "bag_data" / "processed_data" / "left_right"
PARSER = ROOT_DIR / "training" / "rosbag_parser.py"
DEFAULT_TAG = "_w300_e060_hpure"
RAW_BAG_RE = re.compile(r"^go2_data_lr_(\d+)$")
RAW_STORAGE_EXTS = {".db3", ".mcap", ".sqlite3"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild processed left/right .npy datasets from raw go2_data_lr_* bags "
            "and report exact duplicate raw/processed datasets."
        )
    )
    parser.add_argument(
        "--bags",
        nargs="+",
        type=int,
        default=None,
        help="Specific left/right bag ids to rebuild. Default: discover go2_data_lr_* bags.",
    )
    parser.add_argument("--tag", default=DEFAULT_TAG)
    parser.add_argument("--window-ms", type=int, default=300)
    parser.add_argument("--exclude-sec", type=float, default=0.60)
    parser.add_argument("--sampling-hz", type=int, default=200)
    parser.add_argument(
        "--allow-history-crossing",
        action="store_true",
        help="Do not require the full history window to stay inside one label segment.",
    )
    parser.add_argument(
        "--no-downsample-zero-class",
        action="store_true",
        help="Pass --no-downsample-zero-class through to rosbag_parser.py.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Skip reparsing and only report duplicate raw/processed datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print parser commands without running them.",
    )
    return parser.parse_args()


def discover_bags():
    bag_ids = []
    for path in RAW_BAGS_DIR.iterdir():
        if not path.is_dir():
            continue
        match = RAW_BAG_RE.match(path.name)
        if match is not None:
            bag_ids.append(int(match.group(1)))
    return sorted(bag_ids)


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combined_digest(paths):
    digest = hashlib.sha256()
    content_hashes = sorted((path.suffix.lower(), file_digest(path)) for path in paths)
    for suffix, content_hash in content_hashes:
        digest.update(suffix.encode("ascii", errors="ignore"))
        digest.update(content_hash.encode("ascii"))
    return digest.hexdigest()


def duplicate_groups(signatures):
    by_signature = {}
    for bag_id, signature in signatures.items():
        by_signature.setdefault(signature, []).append(bag_id)
    return [bags for bags in by_signature.values() if len(bags) > 1]


def raw_storage_files_for_bag(bag_id):
    bag_dir = RAW_BAGS_DIR / f"go2_data_lr_{bag_id}"
    return [
        path
        for path in bag_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in RAW_STORAGE_EXTS
    ]


def processed_files_for_bag(bag_id, tag):
    paths = [LEFT_RIGHT_DIR / f"{prefix}_lr_{bag_id}{tag}.npy" for prefix in ["X", "y", "t", "seg"]]
    return [path for path in paths if path.exists()]


def report_raw_duplicates(bag_ids):
    signatures = {}
    missing = []
    for bag_id in bag_ids:
        storage_files = raw_storage_files_for_bag(bag_id)
        if not storage_files:
            missing.append(bag_id)
            continue
        signatures[bag_id] = combined_digest(storage_files)

    groups = duplicate_groups(signatures)
    print("\n===== Raw Storage Duplicate Check =====")
    if missing:
        print(f"Bags without .db3/.mcap/.sqlite3 storage files: {missing}")
    if groups:
        print(f"Duplicate raw storage groups: {groups}")
    else:
        print("No exact duplicate raw storage groups found.")
    return groups


def report_processed_duplicates(bag_ids, tag):
    signatures = {}
    missing = []
    for bag_id in bag_ids:
        files = processed_files_for_bag(bag_id, tag)
        if len(files) != 4:
            missing.append(bag_id)
            continue
        signatures[bag_id] = combined_digest(files)

    groups = duplicate_groups(signatures)
    print("\n===== Processed Dataset Duplicate Check =====")
    if missing:
        print(f"Bags missing one or more processed X/y/t/seg files: {missing}")
    if groups:
        print(f"Duplicate processed dataset groups: {groups}")
    else:
        print("No exact duplicate processed dataset groups found.")
    return groups


def parser_command(bag_id, args):
    command = [
        sys.executable,
        str(PARSER),
        "--bag-name",
        f"go2_data_lr_{bag_id}",
        "--keep-pair",
        "34",
        "--window-ms",
        str(args.window_ms),
        "--sampling-hz",
        str(args.sampling_hz),
        "--exclude-sec",
        str(args.exclude_sec),
        "--output-tag",
        args.tag,
    ]
    if not args.allow_history_crossing:
        command.append("--require-full-history-in-segment")
    if args.no_downsample_zero_class:
        command.append("--no-downsample-zero-class")
    return command


def rebuild_bags(bag_ids, args):
    print("\n===== Rebuild Left/Right Datasets =====")
    print(f"bags={bag_ids}")
    print(
        f"tag={args.tag} | window_ms={args.window_ms} | exclude_sec={args.exclude_sec} | "
        f"sampling_hz={args.sampling_hz} | keep_pair=34"
    )
    print(f"require_full_history_in_segment={not args.allow_history_crossing}")

    for bag_id in bag_ids:
        command = parser_command(bag_id, args)
        display = " ".join(command)
        if args.dry_run:
            print(display)
            continue
        print(f"\n--- Rebuilding go2_data_lr_{bag_id} ---")
        print(display)
        subprocess.run(command, cwd=ROOT_DIR, check=True)


def main():
    args = parse_args()
    bag_ids = sorted(args.bags) if args.bags is not None else discover_bags()
    if not bag_ids:
        raise RuntimeError("No left/right raw bags found to rebuild.")

    print(f"Using bag ids: {bag_ids}")
    report_raw_duplicates(bag_ids)
    if not args.check_only:
        rebuild_bags(bag_ids, args)
    report_processed_duplicates(bag_ids, args.tag)


if __name__ == "__main__":
    main()
