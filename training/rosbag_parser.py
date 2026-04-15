#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data"
RAW_BAGS_DIR = DATA_DIR / "raw_bag"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

PUSH_EVENT_TOPIC = "/data/push_event"
LOWSTATE_TOPIC = "/lowstate"
ARM_ANGLES_TOPIC = "/arm_angles"
ARM_ANGLE_DIM = 7
ARM_CURRENT_DIM = 7

rng = np.random.default_rng(42)

push_t_ns = []
push_labels_raw = []

lowstate_t_ns = []
arm_angles_t_ns = []

lowstate_ff = []
lowstate_accel = []
lowstate_q = []
lowstate_dq = []
arm_angles = []
arm_currents = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-name", default="go2_data_ud_2")
    parser.add_argument("--keep-pair", default="56", choices=["12", "34", "56"])
    parser.add_argument("--exclude-sec", type=float, default=0.0)
    parser.add_argument("--window-ms", type=int, default=200)
    parser.add_argument("--sampling-hz", type=int, default=200)
    parser.add_argument(
        "--require-full-history-in-segment",
        action="store_true",
        help=(
            "Keep a sample only when its full history window stays inside the current "
            "contiguous label segment."
        ),
    )
    parser.add_argument(
        "--downsample-zero-class",
        dest="downsample_zero_class",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-downsample-zero-class",
        dest="downsample_zero_class",
        action="store_false",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix appended to the dataset filename, e.g. '_w080'.",
    )
    return parser.parse_args()


def resolve_bagpath(bags_dir, bag_name):
    bagpath = bags_dir / bag_name
    if not bagpath.exists():
        available_bags = sorted(path.name for path in bags_dir.iterdir() if path.is_dir())
        raise ValueError(
            f"Bag {bag_name!r} not found under {bags_dir}. "
            f"Available bag folders: {available_bags}"
        )
    # If metadata.yaml lives directly here, this is the bag directory.
    if (bagpath / "metadata.yaml").exists():
        return bagpath
    # Otherwise look one level deeper (common when the recorder wraps bags in a parent folder).
    subdirs = [p for p in bagpath.iterdir() if p.is_dir() and (p / "metadata.yaml").exists()]
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) > 1:
        raise ValueError(
            f"Multiple bag subdirectories found in {bagpath}: {[s.name for s in subdirs]}. "
            "Pass the specific subfolder name instead."
        )
    raise ValueError(f"No metadata.yaml found in {bagpath} or its immediate subdirectories.")


def dataset_suffix_from_bag_name(bag_name):
    prefix = "go2_data_"
    if bag_name.startswith(prefix):
        return bag_name[len(prefix):]
    return bag_name


def processed_subdir_from_bag_name(bag_name):
    suffix = dataset_suffix_from_bag_name(bag_name)
    if suffix.startswith("fb_"):
        return "front_back"
    if suffix.startswith("lr_"):
        return "left_right"
    if suffix.startswith("ud_") or suffix.startswith("air_updown"):
        return "up_down"
    raise ValueError(
        f"Cannot infer processed data folder from bag name {bag_name!r}. "
        "Expected prefixes like go2_data_fb_*, go2_data_lr_*, go2_data_ud_*, or go2_data_air_updown*."
    )

args = parse_args()
bagpath = resolve_bagpath(RAW_BAGS_DIR, args.bag_name)
dataset_suffix = dataset_suffix_from_bag_name(args.bag_name)
processed_subdir = processed_subdir_from_bag_name(args.bag_name)
keep_pair = args.keep_pair
exclude_sec = args.exclude_sec
downsample_zero_class = args.downsample_zero_class
require_full_history_in_segment = args.require_full_history_in_segment
sliding_window_ms = args.window_ms
sampling_hz = args.sampling_hz
dt_s = 1.0 / sampling_hz
num_steps = int(sliding_window_ms / 1000 * sampling_hz)
dt_ns = int(dt_s * 1e9)
exclude_ns = int(exclude_sec * 1e9)

if num_steps <= 0:
    raise ValueError(
        f"window-ms={sliding_window_ms} and sampling-hz={sampling_hz} produced num_steps={num_steps}."
    )


def sample_nearest(topic_t_ns, topic_x, grid_row_ns):
    idx = np.searchsorted(topic_t_ns, grid_row_ns)
    idx = np.clip(idx, 1, len(topic_t_ns) - 1)

    left = idx - 1
    right = idx

    choose_right = np.abs(topic_t_ns[right] - grid_row_ns) < np.abs(topic_t_ns[left] - grid_row_ns)
    nearest_idx = np.where(choose_right, right, left)

    return topic_x[nearest_idx]


def build_label_segments(times_ns, labels):
    """
    Build contiguous segments of constant label.
    Returns list of dicts with keys: label, start_ns, end_ns
    """
    segments = []
    if len(times_ns) == 0:
        return segments

    start_idx = 0
    current_label = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            segments.append({
                "label": int(current_label),
                "start_ns": int(times_ns[start_idx]),
                "end_ns": int(times_ns[i - 1]),
            })
            start_idx = i
            current_label = labels[i]

    segments.append({
        "label": int(current_label),
        "start_ns": int(times_ns[start_idx]),
        "end_ns": int(times_ns[-1]),
    })
    return segments


def build_segment_ids(labels):
    """Assign a contiguous segment id to each label sample."""
    if len(labels) == 0:
        return np.array([], dtype=np.int64)

    segment_ids = np.zeros(len(labels), dtype=np.int64)
    current_segment_id = 0

    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            current_segment_id += 1
        segment_ids[i] = current_segment_id

    return segment_ids


def is_clean_zero_time(t_ns, nonzero_segments, exclude_ns):
    """True if endpoint t_ns stays outside a fixed exclusion buffer around nonzero segments."""
    for seg in nonzero_segments:
        forbid_start = seg["start_ns"] - exclude_ns
        forbid_end = seg["end_ns"] + exclude_ns

        if forbid_start <= t_ns <= forbid_end:
            return False

    return True


def get_topic_connections(reader, topic, required):
    connections = [connection for connection in reader.connections if connection.topic == topic]
    if required and not connections:
        available_topics = sorted({connection.topic for connection in reader.connections})
        raise ValueError(
            f"Required topic {topic!r} not found in {bagpath}. "
            f"Available topics: {available_topics}"
        )
    return connections


def extract_arm_angle_features(msg):
    return [msg.angle_deg[i] for i in range(ARM_ANGLE_DIM)]


def extract_arm_current_features(msg):
    current = getattr(msg, "current", None)
    if current is None:
        return [0.0] * ARM_CURRENT_DIM
    if len(current) < ARM_CURRENT_DIM:
        raise ValueError(
            f"/arm_angles message current field has {len(current)} values; "
            f"expected at least {ARM_CURRENT_DIM}."
        )
    return [current[i] for i in range(ARM_CURRENT_DIM)]


# ----------------
# Parse rosbag
# ----------------
with AnyReader([bagpath]) as reader:
    push_connections = get_topic_connections(reader, PUSH_EVENT_TOPIC, required=True)
    lowstate_connections = get_topic_connections(reader, LOWSTATE_TOPIC, required=True)
    arm_angles_connections = get_topic_connections(reader, ARM_ANGLES_TOPIC, required=False)

    # Read all push-event timestamps and raw labels.
    for connection, timestamp, rawdata in reader.messages(connections=push_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        push_t_ns.append(timestamp)
        push_labels_raw.append(int(msg.label))

    # Read /arm_angles when present.
    for connection, timestamp, rawdata in reader.messages(connections=arm_angles_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        arm_angles_t_ns.append(timestamp)
        arm_angles.append(extract_arm_angle_features(msg))
        arm_currents.append(extract_arm_current_features(msg))

    # Read /lowstate.
    for connection, timestamp, rawdata in reader.messages(connections=lowstate_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)

        lowstate_t_ns.append(timestamp)
        lowstate_ff.append([
            msg.foot_force[0], msg.foot_force[1], msg.foot_force[2], msg.foot_force[3],
        ])
        lowstate_accel.append([
            msg.imu_state.accelerometer[0],
            msg.imu_state.accelerometer[1],
            msg.imu_state.accelerometer[2],
        ])
        lowstate_q.append([
            msg.motor_state[0].q, msg.motor_state[1].q, msg.motor_state[2].q,
            msg.motor_state[3].q, msg.motor_state[4].q, msg.motor_state[5].q,
            msg.motor_state[6].q, msg.motor_state[7].q, msg.motor_state[8].q,
            msg.motor_state[9].q, msg.motor_state[10].q, msg.motor_state[11].q,
        ])
        lowstate_dq.append([
            msg.motor_state[0].dq, msg.motor_state[1].dq, msg.motor_state[2].dq,
            msg.motor_state[3].dq, msg.motor_state[4].dq, msg.motor_state[5].dq,
            msg.motor_state[6].dq, msg.motor_state[7].dq, msg.motor_state[8].dq,
            msg.motor_state[9].dq, msg.motor_state[10].dq, msg.motor_state[11].dq,
        ])

# ----------------
# Convert to arrays
# ----------------
push_t_ns = np.array(push_t_ns, dtype=np.int64)
push_labels_raw = np.array(push_labels_raw, dtype=np.int64)

if len(push_t_ns) == 0:
    raise ValueError(f"No messages found on required topic {PUSH_EVENT_TOPIC!r} in {bagpath}.")

keep_pair_map = {
    "12": (1, 2),
    "34": (3, 4),
    "56": (5, 6),
}
if keep_pair not in keep_pair_map:
    raise ValueError(f"Unsupported keep_pair: {keep_pair}. Use one of {sorted(keep_pair_map)}")

keep_label_1_raw, keep_label_2_raw = keep_pair_map[keep_pair]
kept_labels = [keep_label_1_raw, keep_label_2_raw]

# Remap labels:
# selected raw pair -> keep original label numbers
# everything else -> 0
push_labels = np.zeros_like(push_labels_raw, dtype=np.int64)
push_labels[push_labels_raw == keep_label_1_raw] = keep_label_1_raw
push_labels[push_labels_raw == keep_label_2_raw] = keep_label_2_raw

lowstate_t_ns = np.array(lowstate_t_ns, dtype=np.int64)
arm_angles_t_ns = np.array(arm_angles_t_ns, dtype=np.int64)

if len(lowstate_t_ns) == 0:
    raise ValueError(f"No messages found on required topic {LOWSTATE_TOPIC!r} in {bagpath}.")

lowstate_ff = np.array(lowstate_ff, dtype=np.float32)
lowstate_accel = np.array(lowstate_accel, dtype=np.float32)
lowstate_q = np.array(lowstate_q, dtype=np.float32)
lowstate_dq = np.array(lowstate_dq, dtype=np.float32)
arm_angles = np.array(arm_angles, dtype=np.float32)
arm_currents = np.array(arm_currents, dtype=np.float32)

if len(arm_angles_t_ns) == 0:
    arm_angles_t_ns = lowstate_t_ns.copy()
    arm_angles = np.zeros((len(lowstate_t_ns), ARM_ANGLE_DIM), dtype=np.float32)
    arm_currents = np.zeros((len(lowstate_t_ns), ARM_CURRENT_DIM), dtype=np.float32)

print("raw label counts:", {k: int(np.sum(push_labels_raw == k)) for k in range(7)})
print("processed label counts:", {k: int(np.sum(push_labels == k)) for k in [0] + kept_labels})
print("window_ms:", sliding_window_ms)
print("sampling_hz:", sampling_hz)
print("num_steps:", num_steps)
print("require_full_history_in_segment:", require_full_history_in_segment)

print("lowstate_t_ns shape:", lowstate_t_ns.shape)
print("arm_angles_t_ns shape:", arm_angles_t_ns.shape)
print("lowstate_ff shape:", lowstate_ff.shape)
print("lowstate_accel shape:", lowstate_accel.shape)
print("lowstate_q shape:", lowstate_q.shape)
print("lowstate_dq shape:", lowstate_dq.shape)
print("arm_angles shape:", arm_angles.shape)
print("arm_currents shape:", arm_currents.shape)

# ----------------
# Build segments from remapped label stream
# Exclusion zone only around the selected raw pair
# ----------------
segments = build_label_segments(push_t_ns, push_labels)
push_segment_ids = build_segment_ids(push_labels)
nonzero_segments = [seg for seg in segments if seg["label"] in kept_labels]
segment_start_ns_by_id = np.array([seg["start_ns"] for seg in segments], dtype=np.int64)

# ----------------
# Keep all selected nonzero labels
# Keep only clean class 0
# ----------------
history_ns = (num_steps - 1) * dt_ns

selected_t_ns = []
selected_labels = []
selected_segment_ids = []
dropped_history_crossing = 0

for t_ns, label, segment_id in zip(push_t_ns, push_labels, push_segment_ids):
    window_start_ns = t_ns - history_ns
    if window_start_ns < lowstate_t_ns[0]:
        continue
    if window_start_ns < arm_angles_t_ns[0]:
        continue
    if require_full_history_in_segment and window_start_ns < segment_start_ns_by_id[segment_id]:
        dropped_history_crossing += 1
        continue

    if label == 0:
        if is_clean_zero_time(t_ns, nonzero_segments, exclude_ns):
            selected_t_ns.append(int(t_ns))
            selected_labels.append(0)
            selected_segment_ids.append(int(segment_id))
    else:
        selected_t_ns.append(int(t_ns))
        selected_labels.append(int(label))
        selected_segment_ids.append(int(segment_id))

selected_t_ns = np.array(selected_t_ns, dtype=np.int64)
selected_labels = np.array(selected_labels, dtype=np.int64)
selected_segment_ids = np.array(selected_segment_ids, dtype=np.int64)

if len(selected_t_ns) == 0:
    raise ValueError("No valid samples were selected after applying the history-window filters.")

print("dropped history-crossing samples:", dropped_history_crossing)

# ----------------
# Downsample class 0
# target = largest count among the selected nonzero labels
# ----------------
counts = {k: int(np.sum(selected_labels == k)) for k in [0] + kept_labels}
print("class counts before downsampling:", counts)

target_zero = (
    max(counts[keep_label_1_raw], counts[keep_label_2_raw])
    if (counts[keep_label_1_raw] > 0 or counts[keep_label_2_raw] > 0)
    else counts[0]
)

if downsample_zero_class:
    idx_zero = np.where(selected_labels == 0)[0]
    idx_nonzero = np.where(selected_labels != 0)[0]

    if len(idx_zero) > target_zero:
        idx_zero_keep = rng.choice(idx_zero, size=target_zero, replace=False)
    else:
        idx_zero_keep = idx_zero

    keep_idx = np.sort(np.concatenate([idx_nonzero, idx_zero_keep]))

    selected_t_ns = selected_t_ns[keep_idx]
    selected_labels = selected_labels[keep_idx]
    selected_segment_ids = selected_segment_ids[keep_idx]

counts = {k: int(np.sum(selected_labels == k)) for k in [0] + kept_labels}
print("class counts after downsampling:", counts)

# ----------------
# Create time grids
# ----------------
offsets_ns = np.arange(num_steps - 1, -1, -1) * dt_ns
grid_ns = selected_t_ns[:, None] - offsets_ns[None, :]

# ----------------
# Sample data
# ----------------
X_ff = []
X_accel = []
X_q = []
X_dq = []
X_arm_angles = []
X_arm_currents = []
y = []
valid_t_ns = []
valid_segment_ids = []

for i in range(len(grid_ns)):
    X_ff.append(sample_nearest(lowstate_t_ns, lowstate_ff, grid_ns[i]))
    X_accel.append(sample_nearest(lowstate_t_ns, lowstate_accel, grid_ns[i]))
    X_q.append(sample_nearest(lowstate_t_ns, lowstate_q, grid_ns[i]))
    X_dq.append(sample_nearest(lowstate_t_ns, lowstate_dq, grid_ns[i]))
    X_arm_angles.append(sample_nearest(arm_angles_t_ns, arm_angles, grid_ns[i]))
    X_arm_currents.append(sample_nearest(arm_angles_t_ns, arm_currents, grid_ns[i]))

    y.append(selected_labels[i])
    valid_t_ns.append(selected_t_ns[i])
    valid_segment_ids.append(selected_segment_ids[i])

X_ff = np.stack(X_ff, axis=0)
X_accel = np.stack(X_accel, axis=0)
X_q = np.stack(X_q, axis=0)
X_dq = np.stack(X_dq, axis=0)
X_arm_angles = np.stack(X_arm_angles, axis=0)
X_arm_currents = np.stack(X_arm_currents, axis=0)

y = np.array(y, dtype=np.int64)
valid_t_ns = np.array(valid_t_ns, dtype=np.int64)
valid_segment_ids = np.array(valid_segment_ids, dtype=np.int64)

print("final class counts:", {k: int(np.sum(y == k)) for k in [0] + kept_labels})

feature_blocks = [X_ff, X_accel, X_q, X_dq, X_arm_angles, X_arm_currents]
X = np.concatenate(feature_blocks, axis=2)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Save dataset
out_dir = PROCESSED_DATA_DIR / processed_subdir
out_dir.mkdir(parents=True, exist_ok=True)

output_suffix = args.output_tag.strip()
x_path = out_dir / f'X_{dataset_suffix}{output_suffix}.npy'
y_path = out_dir / f'y_{dataset_suffix}{output_suffix}.npy'
t_path = out_dir / f't_{dataset_suffix}{output_suffix}.npy'
seg_path = out_dir / f'seg_{dataset_suffix}{output_suffix}.npy'

np.save(x_path, X)
np.save(y_path, y)
np.save(t_path, valid_t_ns)
np.save(seg_path, valid_segment_ids)

print("Saved:")
print(x_path)
print(y_path)
print(t_path)
print(seg_path)
