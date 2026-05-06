#!/usr/bin/env python3
"""
raw_fig.py — visualise the structure of the processed up/down and left/right datasets.

Figures produced (saved to figs/raw/):
  01_ud_label_dist.png      — per-bag label sample counts (up/down)
  02_lr_label_dist.png      — per-bag label sample counts (left/right)
  03_ud_sample_counts.png   — total + per-class sample counts per bag (UD)
  04_lr_sample_counts.png   — total + per-class sample counts per bag (LR)
  05_ud_feature_stats.png   — per-feature-block mean ± std across all UD bags
  06_lr_feature_stats.png   — per-feature-block mean ± std across all LR bags
  07_ud_zero_heatmap.png    — fraction of timesteps that are exactly zero, per feature col per bag
  08_ud_arm_zero_flag.png   — which bags have zeroed arm features (new-stack bags)
  09_ud_segment_counts.png  — number of label segments per bag
  10_sample_windows.png     — one example window per class (UD bag 2)
  11_ud_feature_corr.png    — cross-feature Pearson correlation matrix (pooled UD data)
  12_label_pie.png          — overall class balance pie charts (UD and LR)

Run:
  uv run --with numpy,matplotlib,scikit-learn python training/raw_fig.py [--tag TAG] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
UD_DIR   = ROOT_DIR / "bag_data" / "processed_data" / "up_down"
LR_DIR   = ROOT_DIR / "bag_data" / "processed_data" / "left_right"

# Feature layout (3-joint arm, tag _w500_e060_hpure)
FEATURE_BLOCKS = {
    "ff":          (0,  4),
    "accel":       (4,  7),
    "q":           (7,  19),
    "dq":          (19, 31),
    "arm_angles":  (31, 34),
    "arm_currents":(34, 37),
}
BLOCK_COLORS = {
    "ff":          "#4e79a7",
    "accel":       "#f28e2b",
    "q":           "#e15759",
    "dq":          "#76b7b2",
    "arm_angles":  "#59a14f",
    "arm_currents":"#edc948",
}
TOTAL_FEATURES = 37

UD_LABEL_NAMES = {0: "rest", 5: "up", 6: "down"}
LR_LABEL_NAMES = {0: "rest", 3: "left", 4: "right"}
UD_LABEL_COLORS = {0: "#aec7e8", 5: "#1f77b4", 6: "#d62728"}
LR_LABEL_COLORS = {0: "#aec7e8", 3: "#2ca02c", 4: "#ff7f0e"}


# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def discover_bags(data_dir: Path, prefix: str, tag: str) -> list[int]:
    pattern = f"X_{prefix}_*{tag}.npy"
    ids = []
    for p in data_dir.glob(pattern):
        parts = p.stem.split("_")
        try:
            ids.append(int(parts[2]))
        except (IndexError, ValueError):
            pass
    return sorted(ids)


def load_bag(data_dir: Path, prefix: str, bag: int, tag: str):
    """Return (X, y, t, seg) for one bag. seg may be None if file missing."""
    X   = np.load(data_dir / f"X_{prefix}_{bag}{tag}.npy")
    y   = np.load(data_dir / f"y_{prefix}_{bag}{tag}.npy")
    t   = np.load(data_dir / f"t_{prefix}_{bag}{tag}.npy")
    seg_path = data_dir / f"seg_{prefix}_{bag}{tag}.npy"
    seg = np.load(seg_path) if seg_path.exists() else None
    return X, y, t, seg


def load_all(data_dir: Path, prefix: str, tag: str, bags: list[int],
             expected_features: int | None = None):
    """Load all bags, return dict bag_id -> (X, y, t, seg).

    If expected_features is given, bags whose X.shape[2] != expected_features
    are skipped with a warning (handles mixed old/new parser outputs).
    """
    out = {}
    for bag in bags:
        try:
            entry = load_bag(data_dir, prefix, bag, tag)
        except FileNotFoundError:
            continue
        X = entry[0]
        if expected_features is not None and X.shape[2] != expected_features:
            print(f"  [skip] bag {bag}: feature dim {X.shape[2]} != {expected_features}")
            continue
        out[bag] = entry
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Figure helpers
# ─────────────────────────────────────────────────────────────────────────────

def save(fig, out_dir: Path, name: str):
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 & 2 — label distribution per bag (stacked bar)
# ─────────────────────────────────────────────────────────────────────────────

def fig_label_dist(bags_data: dict, label_names: dict, label_colors: dict,
                   title: str, out_dir: Path, fname: str):
    bags = sorted(bags_data)
    labels = sorted(label_names)
    counts = {lbl: [] for lbl in labels}
    for bag in bags:
        _, y, _, _ = bags_data[bag]
        for lbl in labels:
            counts[lbl].append(int(np.sum(y == lbl)))

    fig, ax = plt.subplots(figsize=(max(10, len(bags) * 0.55), 5))
    x = np.arange(len(bags))
    bottom = np.zeros(len(bags))
    for lbl in labels:
        vals = np.array(counts[lbl], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label_names[lbl],
               color=label_colors[lbl], edgecolor="white", linewidth=0.4)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bags], fontsize=8)
    ax.set_xlabel("Bag ID")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 & 4 — total sample counts with per-class breakdown (line overlay)
# ─────────────────────────────────────────────────────────────────────────────

def fig_sample_counts(bags_data: dict, label_names: dict, label_colors: dict,
                      title: str, out_dir: Path, fname: str):
    bags = sorted(bags_data)
    labels = sorted(label_names)
    totals = [len(bags_data[b][1]) for b in bags]

    fig, ax = plt.subplots(figsize=(max(10, len(bags) * 0.55), 5))
    x = np.arange(len(bags))
    ax.bar(x, totals, color="#cccccc", edgecolor="white", label="total")
    for lbl in labels:
        if lbl == 0:
            continue
        vals = [int(np.sum(bags_data[b][1] == lbl)) for b in bags]
        ax.plot(x, vals, marker="o", markersize=4,
                label=label_names[lbl], color=label_colors[lbl])

    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bags], fontsize=8)
    ax.set_xlabel("Bag ID")
    ax.set_ylabel("Sample count")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 & 6 — per-feature-block mean ± std (pooled across all bags)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_stats(bags_data: dict, title: str, out_dir: Path, fname: str):
    """Box-plot of mean feature value per sample window, one sub-plot per block."""
    n_blocks = len(FEATURE_BLOCKS)
    fig, axes = plt.subplots(1, n_blocks, figsize=(3 * n_blocks, 5), sharey=False)

    all_X = np.concatenate([bags_data[b][0] for b in sorted(bags_data)], axis=0)
    # mean over time axis → (N, F)
    X_mean = all_X.mean(axis=1)

    for ax, (bname, (s, e)) in zip(axes, FEATURE_BLOCKS.items()):
        data = [X_mean[:, col] for col in range(s, e)]
        positions = list(range(1, len(data) + 1))
        bp = ax.boxplot(data, positions=positions, patch_artist=True,
                        medianprops=dict(color="black", linewidth=1.5),
                        flierprops=dict(marker=".", markersize=2, alpha=0.3),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8))
        for patch in bp["boxes"]:
            patch.set_facecolor(BLOCK_COLORS[bname])
            patch.set_alpha(0.7)
        ax.set_title(bname, fontsize=9)
        ax.set_xlabel("feature index (within block)", fontsize=7)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(s + i) for i in range(e - s)], fontsize=7)
        ax.tick_params(axis="y", labelsize=7)

    axes[0].set_ylabel("mean value per window")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 7 — zero-fraction heatmap per bag × feature column (UD only)
# ─────────────────────────────────────────────────────────────────────────────

def fig_zero_heatmap(bags_data: dict, title: str, out_dir: Path, fname: str):
    bags = sorted(bags_data)
    n_feat = TOTAL_FEATURES
    matrix = np.zeros((len(bags), n_feat))
    for i, bag in enumerate(bags):
        X = bags_data[bag][0]   # (N, T, F)
        # fraction of samples where ALL timesteps are zero for that feature
        matrix[i] = (X == 0).all(axis=1).mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(12, n_feat * 0.35), max(6, len(bags) * 0.35)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(len(bags)))
    ax.set_yticklabels([str(b) for b in bags], fontsize=7)
    ax.set_xlabel("Feature column index")
    ax.set_ylabel("Bag ID")
    ax.set_title(title)

    # block boundary lines and labels
    for bname, (s, e) in FEATURE_BLOCKS.items():
        ax.axvline(s - 0.5, color="white", linewidth=0.8)
        ax.text((s + e) / 2 - 0.5, -0.8, bname, ha="center", va="top",
                fontsize=7, color="#333333",
                transform=ax.get_xaxis_transform())

    plt.colorbar(im, ax=ax, label="zero fraction (all timesteps)")
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 8 — arm-feature zero flag per bag (which bags zeroed arm features)
# ─────────────────────────────────────────────────────────────────────────────

def fig_arm_zero_flag(ud_data: dict, out_dir: Path, fname: str):
    bags = sorted(ud_data)
    arm_cols = list(range(31, 37))
    zero_fracs = []
    for bag in bags:
        X = ud_data[bag][0]
        frac = (X == 0).all(axis=1).mean(axis=0)[arm_cols].mean()
        zero_fracs.append(frac)

    colors = ["#d62728" if z > 0.5 else "#2ca02c" for z in zero_fracs]
    fig, ax = plt.subplots(figsize=(max(10, len(bags) * 0.55), 4))
    x = np.arange(len(bags))
    ax.bar(x, zero_fracs, color=colors, edgecolor="white")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="50% threshold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in bags], fontsize=8)
    ax.set_xlabel("Bag ID")
    ax.set_ylabel("Avg arm-feature zero fraction")
    ax.set_title("Arm feature availability per bag\n(red = arm zeroed, likely new-stack bag with /arm/state missing)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    # annotate bag IDs that are fully zeroed
    for xi, (bag, frac) in enumerate(zip(bags, zero_fracs)):
        if frac > 0.5:
            ax.text(xi, frac + 0.02, str(bag), ha="center", va="bottom", fontsize=7, color="#d62728")
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 9 — segment counts per bag
# ─────────────────────────────────────────────────────────────────────────────

def fig_segment_counts(ud_data: dict, lr_data: dict, out_dir: Path, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, title, color in [
        (axes[0], ud_data, "Up/Down — segments per bag", "#1f77b4"),
        (axes[1], lr_data, "Left/Right — segments per bag", "#2ca02c"),
    ]:
        bags = sorted(data)
        segs = []
        for bag in bags:
            seg = data[bag][3]
            segs.append(len(np.unique(seg)) if seg is not None else 0)
        x = np.arange(len(bags))
        ax.bar(x, segs, color=color, alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in bags], fontsize=8)
        ax.set_xlabel("Bag ID")
        ax.set_ylabel("Unique label segments")
        ax.set_title(title)
        for xi, s in enumerate(segs):
            ax.text(xi, s + 0.3, str(s), ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 10 — example windows per class (UD bag 2)
# ─────────────────────────────────────────────────────────────────────────────

def fig_sample_windows(ud_data: dict, out_dir: Path, fname: str):
    # pick a representative bag with all 3 classes
    for probe_bag in [2, 14, 15]:
        if probe_bag in ud_data:
            X, y, _, _ = ud_data[probe_bag]
            break
    else:
        print("  [skip] no suitable UD bag found for window plot")
        return

    labels_present = [lbl for lbl in [0, 5, 6] if np.any(y == lbl)]
    n_blocks = len(FEATURE_BLOCKS)
    n_labels = len(labels_present)
    t_axis = np.arange(X.shape[1])

    fig, axes = plt.subplots(n_labels, n_blocks,
                             figsize=(3 * n_blocks, 2.5 * n_labels),
                             sharex=True)
    if n_labels == 1:
        axes = axes[np.newaxis, :]

    for row, lbl in enumerate(labels_present):
        idx = np.where(y == lbl)[0]
        sample = X[idx[len(idx) // 2]]   # pick middle sample
        for col, (bname, (s, e)) in enumerate(FEATURE_BLOCKS.items()):
            ax = axes[row, col]
            ax.plot(t_axis, sample[:, s:e], linewidth=0.8)
            if row == 0:
                ax.set_title(bname, fontsize=8)
            if col == 0:
                ax.set_ylabel(UD_LABEL_NAMES.get(lbl, str(lbl)), fontsize=8,
                              color=UD_LABEL_COLORS.get(lbl, "black"))
            ax.tick_params(labelsize=6)
            ax.set_xlabel("timestep" if row == n_labels - 1 else "", fontsize=7)

    fig.suptitle(f"Example windows per class — UD bag {probe_bag}", fontsize=11)
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 11 — cross-feature correlation matrix (pooled UD)
# ─────────────────────────────────────────────────────────────────────────────

def fig_feature_corr(ud_data: dict, out_dir: Path, fname: str):
    all_X = np.concatenate([ud_data[b][0] for b in sorted(ud_data)], axis=0)
    X_mean = all_X.mean(axis=1)   # (N, F) — mean over time
    corr = np.corrcoef(X_mean.T)  # (F, F)

    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson r")

    # block boundary lines
    for bname, (s, e) in FEATURE_BLOCKS.items():
        ax.axhline(s - 0.5, color="black", linewidth=0.6)
        ax.axvline(s - 0.5, color="black", linewidth=0.6)
        ax.text(s + (e - s) / 2 - 0.5, -1.2, bname, ha="center", va="top",
                fontsize=7, transform=ax.get_xaxis_transform())
        ax.text(-1.2, s + (e - s) / 2 - 0.5, bname, ha="right", va="center",
                fontsize=7, transform=ax.get_yaxis_transform())

    ax.set_title("Cross-feature Pearson correlation (UD, window means)", fontsize=11)
    ax.set_xlabel("Feature index")
    ax.set_ylabel("Feature index")
    ax.set_xticks(range(TOTAL_FEATURES))
    ax.set_yticks(range(TOTAL_FEATURES))
    ax.set_xticklabels(range(TOTAL_FEATURES), fontsize=5)
    ax.set_yticklabels(range(TOTAL_FEATURES), fontsize=5)
    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Fig 12 — overall class balance pie charts
# ─────────────────────────────────────────────────────────────────────────────

def fig_label_pie(ud_data: dict, lr_data: dict, out_dir: Path, fname: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, data, label_names, label_colors, title in [
        (axes[0], ud_data, UD_LABEL_NAMES, UD_LABEL_COLORS, "Up/Down — overall class balance"),
        (axes[1], lr_data, LR_LABEL_NAMES, LR_LABEL_COLORS, "Left/Right — overall class balance"),
    ]:
        all_y = np.concatenate([data[b][1] for b in sorted(data)])
        labels_present = sorted(label_names)
        sizes  = [int(np.sum(all_y == lbl)) for lbl in labels_present]
        colors = [label_colors[lbl] for lbl in labels_present]
        names  = [f"{label_names[lbl]}\n({s:,})" for lbl, s in zip(labels_present, sizes)]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=names, colors=colors, autopct="%1.1f%%",
            startangle=90, pctdistance=0.75,
            wedgeprops=dict(edgecolor="white", linewidth=1.2),
        )
        for text in texts:
            text.set_fontsize(text.get_fontsize() + 4)
        for autotext in autotexts:
            autotext.set_fontsize(autotext.get_fontsize() + 4)
        ax.set_title(title, fontsize=16)

    fig.tight_layout()
    save(fig, out_dir, fname)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Plot raw dataset structure figures.")
    parser.add_argument("--tag",     default="_w500_e060_hpure",
                        help="Dataset tag suffix (default: _w500_e060_hpure).")
    parser.add_argument("--out-dir", default="figs/raw",
                        help="Output directory for PNG files (default: figs/raw).")
    parser.add_argument("--ud-bags", nargs="*", type=int, default=None,
                        help="Specific UD bag IDs to include (default: auto-discover).")
    parser.add_argument("--lr-bags", nargs="*", type=int, default=None,
                        help="Specific LR bag IDs to include (default: auto-discover).")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = ROOT_DIR / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = args.tag
    ud_bags = args.ud_bags or discover_bags(UD_DIR, "ud", tag)
    lr_bags = args.lr_bags or discover_bags(LR_DIR, "lr", tag)

    print(f"Tag:     {tag}")
    print(f"UD bags: {ud_bags}")
    print(f"LR bags: {lr_bags}")
    print(f"Output:  {out_dir}\n")

    print("Loading up/down data...")
    ud_data = load_all(UD_DIR, "ud", tag, ud_bags, expected_features=TOTAL_FEATURES)
    print("Loading left/right data...")
    lr_data = load_all(LR_DIR, "lr", tag, lr_bags, expected_features=TOTAL_FEATURES)

    print("\nGenerating figures...")

    fig_label_dist(ud_data, UD_LABEL_NAMES, UD_LABEL_COLORS,
                   f"Up/Down label distribution per bag  [{tag}]",
                   out_dir, "01_ud_label_dist.png")

    fig_label_dist(lr_data, LR_LABEL_NAMES, LR_LABEL_COLORS,
                   f"Left/Right label distribution per bag  [{tag}]",
                   out_dir, "02_lr_label_dist.png")

    fig_sample_counts(ud_data, UD_LABEL_NAMES, UD_LABEL_COLORS,
                      f"Up/Down total sample counts per bag  [{tag}]",
                      out_dir, "03_ud_sample_counts.png")

    fig_sample_counts(lr_data, LR_LABEL_NAMES, LR_LABEL_COLORS,
                      f"Left/Right total sample counts per bag  [{tag}]",
                      out_dir, "04_lr_sample_counts.png")

    fig_feature_stats(ud_data,
                      f"Up/Down feature block distributions (window mean)  [{tag}]",
                      out_dir, "05_ud_feature_stats.png")

    fig_feature_stats(lr_data,
                      f"Left/Right feature block distributions (window mean)  [{tag}]",
                      out_dir, "06_lr_feature_stats.png")

    fig_zero_heatmap(ud_data,
                     f"Up/Down zero-fraction heatmap  [{tag}]",
                     out_dir, "07_ud_zero_heatmap.png")

    fig_arm_zero_flag(ud_data, out_dir, "08_ud_arm_zero_flag.png")

    fig_segment_counts(ud_data, lr_data, out_dir, "09_segment_counts.png")

    fig_sample_windows(ud_data, out_dir, "10_sample_windows.png")

    fig_feature_corr(ud_data, out_dir, "11_ud_feature_corr.png")

    fig_label_pie(ud_data, lr_data, out_dir, "12_label_pie.png")

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
