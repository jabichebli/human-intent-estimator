#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "bag_data"
RAW_BAGS_DIR = DATA_DIR / "raw_bag"
PROCESSED_DATA_DIR = DATA_DIR / "processed_data"

PUSH_EVENT_TOPIC = "/data/push_event"
LOWSTATE_TOPIC = "/lowstate"
ARM_ANGLES_TOPIC         = "/arm_angles"        # go2_msgs stack (bags 2–28)
ARM_STATE_TOPIC          = "/arm/state"         # hq_pcot_msgs stack (bags 30+)
ARM_SERVO_FEEDBACK_TOPIC = "/arm/servo_feedback"  # icon_lab_d1_ros2 stack (bag 1)
ARM_SERVO_COMMAND_TOPIC  = "/arm/servo_command"   # icon_lab_d1_ros2 stack (bag 1)
# Joints excluded from arm features:
#   0 — shoulder roll (not informative for push direction)
#   3 — elbow yaw  (not informative for push direction)
#   5 — wrist roll  (not informative for push direction)
#   6 — gripper     (excluded to prevent overfitting)
# Kept joints: 1 (shoulder pitch), 2 (elbow pitch), 4 (wrist pitch) — 3 joints total.
ARM_JOINT_DIM = 7                            # total joints in the arm (0–6)
ARM_EXCLUDED_JOINTS = frozenset({0, 3, 5, 6})
ARM_KEPT_JOINTS = [i for i in range(ARM_JOINT_DIM) if i not in ARM_EXCLUDED_JOINTS]  # [1, 2, 4]
ARM_ANGLE_DIM = len(ARM_KEPT_JOINTS)         # 3
ARM_CURRENT_DIM = len(ARM_KEPT_JOINTS)       # 3

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
arm_velocities = []    # velocity_deg_s from servo_feedback (icon_lab_d1_ros2 only)
last_actions = []      # last commanded angle_deg from servo_command (icon_lab_d1_ros2 only)
last_action_t_ns = [] # timestamps for servo_command (separate topic, may differ from feedback)
_use_servo_topics = False  # set True when icon_lab_d1_ros2 stack is detected


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
    parser.add_argument(
        "--arm-excluded-joints",
        nargs="*",
        type=int,
        default=[0, 3, 5, 6],
        metavar="J",
        help=(
            "Arm joint indices to exclude (0-based, out of 7 total). "
            "Default: 0 3 5 6 (keeps joints 1, 2, 4). "
            "Pass no indices to keep all 7 joints."
        ),
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

# Override arm joint selection from CLI so multiple experiments can coexist.
ARM_EXCLUDED_JOINTS = frozenset(args.arm_excluded_joints)
ARM_KEPT_JOINTS = [i for i in range(ARM_JOINT_DIM) if i not in ARM_EXCLUDED_JOINTS]
ARM_ANGLE_DIM = len(ARM_KEPT_JOINTS)
ARM_CURRENT_DIM = len(ARM_KEPT_JOINTS)
print(f"[Info] Arm kept joints: {ARM_KEPT_JOINTS}  (excluded: {sorted(ARM_EXCLUDED_JOINTS)})")

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
    # Keep only ARM_KEPT_JOINTS = [1, 2, 4]; joints 0, 3, 5, 6 are excluded.
    return [msg.angle_deg[i] for i in ARM_KEPT_JOINTS]


def extract_arm_current_features(msg):
    """For go2_msgs / hq_pcot_msgs stacks: field name is 'current'."""
    current = getattr(msg, "current", None)
    if current is None:
        return [0.0] * ARM_CURRENT_DIM
    if len(current) < ARM_JOINT_DIM:
        raise ValueError(
            f"/arm_angles message current field has {len(current)} values; "
            f"expected at least {ARM_JOINT_DIM}."
        )
    # Keep only ARM_KEPT_JOINTS = [1, 2, 4]; joints 0, 3, 5, 6 are excluded.
    return [current[i] for i in ARM_KEPT_JOINTS]


def extract_servo_current_features(msg):
    """For icon_lab_d1_ros2 stack: field name is 'current_ma' (milliamps)."""
    current_ma = getattr(msg, "current_ma", None)
    if current_ma is None:
        return [0.0] * ARM_CURRENT_DIM
    return [float(current_ma[i]) for i in ARM_KEPT_JOINTS]


def extract_servo_velocity_features(msg):
    """For icon_lab_d1_ros2 stack: velocity_deg_s field in servo_feedback."""
    vel = getattr(msg, "velocity_deg_s", None)
    if vel is None:
        return [0.0] * ARM_ANGLE_DIM
    return [float(vel[i]) for i in ARM_KEPT_JOINTS]


def extract_servo_command_features(msg):
    """For icon_lab_d1_ros2 stack: last commanded angle from servo_command.
    Joint 6 (gripper) may be NaN — replaced with 0.0 to avoid propagating NaN.
    """
    angle_deg = getattr(msg, "angle_deg", None)
    if angle_deg is None:
        return [0.0] * ARM_ANGLE_DIM
    values = []
    for i in ARM_KEPT_JOINTS:
        v = float(angle_deg[i])
        values.append(0.0 if (v != v) else v)  # NaN check: NaN != NaN
    return values


# ----------------
# Parse rosbag
# ----------------
with AnyReader([bagpath]) as reader:
    push_connections = get_topic_connections(reader, PUSH_EVENT_TOPIC, required=True)
    lowstate_connections = get_topic_connections(reader, LOWSTATE_TOPIC, required=True)
    # Resolve which topic carries arm joint data for this bag.
    # Stack 1 — go2_msgs (bags 2–28):          /arm_angles       → angle_deg, current
    # Stack 2 — hq_pcot_msgs (bags 30+):       /arm/state        → angle_deg, current
    # Stack 3 — icon_lab_d1_ros2 (bag 1):      /arm/servo_feedback → angle_deg, current_ma, velocity_deg_s
    #                                           /arm/servo_command  → angle_deg (last commanded)
    arm_angles_connections = get_topic_connections(reader, ARM_ANGLES_TOPIC, required=False)
    servo_feedback_connections = get_topic_connections(reader, ARM_SERVO_FEEDBACK_TOPIC, required=False)
    servo_command_connections = get_topic_connections(reader, ARM_SERVO_COMMAND_TOPIC, required=False)

    if servo_feedback_connections:
        # icon_lab_d1_ros2 stack — use servo_feedback + servo_command
        _arm_topic_used = ARM_SERVO_FEEDBACK_TOPIC
        _use_servo_topics = True
        arm_angles_connections = servo_feedback_connections
        print(f"[Info] Detected icon_lab_d1_ros2 stack; reading arm features from "
              f"{ARM_SERVO_FEEDBACK_TOPIC!r} + {ARM_SERVO_COMMAND_TOPIC!r}")
    elif arm_angles_connections:
        _arm_topic_used = ARM_ANGLES_TOPIC
        _use_servo_topics = False
        print(f"[Info] Reading arm features from {_arm_topic_used!r}")
    else:
        arm_angles_connections = get_topic_connections(reader, ARM_STATE_TOPIC, required=False)
        _arm_topic_used = ARM_STATE_TOPIC if arm_angles_connections else None
        _use_servo_topics = False
        if _arm_topic_used:
            print(f"[Info] Reading arm features from {_arm_topic_used!r}")
        else:
            print("[Warning] No arm-angle topic found; arm features will be zeroed.")

    # Read all push-event timestamps and raw labels.
    for connection, timestamp, rawdata in reader.messages(connections=push_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        push_t_ns.append(timestamp)
        push_labels_raw.append(int(msg.label))

    # Read arm joint data from whichever topic was found.
    _arm_type_warned = False
    for connection, timestamp, rawdata in reader.messages(connections=arm_angles_connections):
        msg = reader.deserialize(rawdata, connection.msgtype)
        if not hasattr(msg, "angle_deg"):
            if not _arm_type_warned:
                print(f"[Warning] Message on {_arm_topic_used!r} has no 'angle_deg' field "
                      f"(type={connection.msgtype}); skipping.")
                _arm_type_warned = True
            continue
        arm_angles_t_ns.append(timestamp)
        arm_angles.append(extract_arm_angle_features(msg))
        if _use_servo_topics:
            # icon_lab_d1_ros2: current field is current_ma (milliamps), velocity is new
            arm_currents.append(extract_servo_current_features(msg))
            arm_velocities.append(extract_servo_velocity_features(msg))
        else:
            arm_currents.append(extract_arm_current_features(msg))

    # Read servo_command for last commanded angles (icon_lab_d1_ros2 stack only).
    if _use_servo_topics and servo_command_connections:
        for connection, timestamp, rawdata in reader.messages(connections=servo_command_connections):
            msg = reader.deserialize(rawdata, connection.msgtype)
            last_action_t_ns.append(timestamp)
            last_actions.append(extract_servo_command_features(msg))

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

# Convert icon_lab_d1_ros2 extra features; fall back to zeros for older stacks.
last_action_t_ns = np.array(last_action_t_ns, dtype=np.int64)
if _use_servo_topics:
    arm_velocities = np.array(arm_velocities, dtype=np.float32)
    if len(last_actions) > 0:
        last_actions = np.array(last_actions, dtype=np.float32)
    else:
        # servo_command topic missing — zero-fill aligned to servo_feedback timestamps
        print("[Warning] No servo_command messages found; last_action features will be zeroed.")
        last_action_t_ns = arm_angles_t_ns.copy()
        last_actions = np.zeros((len(arm_angles_t_ns), ARM_ANGLE_DIM), dtype=np.float32)
else:
    # Older stacks: no velocity/command features — placeholders so sampling code is uniform
    arm_velocities = None
    last_actions = None
    last_action_t_ns = None

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
if _use_servo_topics:
    print("arm_velocities shape:", arm_velocities.shape)
    print("last_actions shape:", last_actions.shape)

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
X_arm_velocities = []
X_last_actions = []
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
    if _use_servo_topics:
        X_arm_velocities.append(sample_nearest(arm_angles_t_ns, arm_velocities, grid_ns[i]))
        X_last_actions.append(sample_nearest(last_action_t_ns, last_actions, grid_ns[i]))

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
if _use_servo_topics:
    X_arm_velocities = np.stack(X_arm_velocities, axis=0)
    X_last_actions = np.stack(X_last_actions, axis=0)
    feature_blocks.extend([X_arm_velocities, X_last_actions])
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

# Write a feature-layout sidecar so the training code always knows slice bounds,
# regardless of how many arm joints were kept at parse time.
_arm_start = 4 + 3 + 12 + 12  # ff + accel + q + dq = 31
feature_layout = {
    "ff":           [0, 4],
    "accel":        [4, 7],
    "q":            [7, 19],
    "dq":           [19, 31],
    "arm_angles":   [_arm_start, _arm_start + ARM_ANGLE_DIM],
    "arm_currents": [_arm_start + ARM_ANGLE_DIM, _arm_start + ARM_ANGLE_DIM + ARM_CURRENT_DIM],
    "arm_kept_joints": ARM_KEPT_JOINTS,
    "total_raw_features": int(X.shape[2]),
}
if _use_servo_topics:
    _vel_start = _arm_start + ARM_ANGLE_DIM + ARM_CURRENT_DIM
    feature_layout["arm_velocities"] = [_vel_start, _vel_start + ARM_ANGLE_DIM]
    feature_layout["last_action"]    = [_vel_start + ARM_ANGLE_DIM, _vel_start + 2 * ARM_ANGLE_DIM]
layout_path = out_dir / f"feature_layout{output_suffix}.json"
with open(layout_path, "w") as _f:
    json.dump(feature_layout, _f, indent=2)

print("Saved:")
print(x_path)
print(y_path)
print(t_path)
print(seg_path)
print(layout_path)
