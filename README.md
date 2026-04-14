# Human Intent Estimator

Train lightweight 1D CNNs that estimate human push intent around a Unitree Go2 from ROS 2 bag data. This repository turns raw MCAP/rosbag recordings into fixed-length NumPy windows, then trains separate 3-class classifiers for front/back, left/right, and up/down intent.

The codebase is small on purpose: the scripts in `training/` are the interface.

## What This Project Does

The pipeline is:

1. Read raw bag data from `bag_data/raw_bag/`
2. Extract synchronized telemetry around `/data/push_event` timestamps
3. Build fixed-history windows from robot state and arm state
4. Remap labels into one neutral class plus one directional pair
5. Train a 1D CNN for that pair
6. Optionally export PyTorch, ONNX, and deployment metadata artifacts

Each trained model predicts one of three classes:

- `0`: neutral / no selected push intent
- directional label A
- directional label B

The three label families used in this repo are:

- front/back: raw labels `1` and `2`
- left/right: raw labels `3` and `4`
- up/down: raw labels `5` and `6`

## Repository Layout

```text
human-intent-estimator/
|- bag_data/
|  |- raw_bag/                 # source rosbag/MCAP recordings
|  \- processed_data/
|     |- front_back/
|     |- left_right/
|     \- up_down/
|- training/
|  |- rosbag_parser.py         # bag -> NumPy window dataset
|  |- train_cnn_common.py      # shared training/eval/export logic
|  |- train_cnn_012.py         # front/back experiment
|  |- train_cnn_034.py         # left/right experiment
|  \- train_cnn_056.py         # up/down experiment
\- README.md
```

## Data and Feature Schema

`training/rosbag_parser.py` builds one sample per selected push-event timestamp. Each sample is a history window ending at that timestamp.

Required topics:

- `/data/push_event`
- `/lowstate`

Optional topic:

- `/arm_angles`

If `/arm_angles` is missing, the parser fills arm-angle and arm-current features with zeros so dataset generation still succeeds.

The full per-timestep feature vector has 45 values:

| Feature block | Width | Source topic |
| --- | ---: | --- |
| foot forces (`ff`) | 4 | `/lowstate` |
| IMU accelerometer (`accel`) | 3 | `/lowstate` |
| joint positions (`q`) | 12 | `/lowstate` |
| joint velocities (`dq`) | 12 | `/lowstate` |
| arm angles (`arm_angles`) | 7 | `/arm_angles` |
| arm currents (`arm_currents`) | 7 | `/arm_angles` |

Saved arrays look like:

- `X_*.npy`: shape `(N, T, 45)`
- `y_*.npy`: shape `(N,)`

Where:

- `N` = number of windows
- `T` = number of timesteps in the history window
- `45` = full feature width before feature selection

For example, a `300 ms` window at `200 Hz` produces `T = 60`.

## Preprocessing Logic

For a chosen label pair:

- the selected raw labels stay in their original raw numbering
- all other labels are remapped to `0`
- class `0` can be downsampled to reduce imbalance
- windows near nonzero events can be excluded from neutral sampling with `--exclude-sec`

The parser also infers the processed output directory from the bag name:

- `go2_data_fb_*` -> `bag_data/processed_data/front_back/`
- `go2_data_lr_*` -> `bag_data/processed_data/left_right/`
- `go2_data_ud_*` or `go2_data_air_updown*` -> `bag_data/processed_data/up_down/`

## Setup

This repo does not currently include a pinned `requirements.txt` or `pyproject.toml`, so install the packages imported by the scripts.

Recommended starting point on Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install numpy matplotlib scikit-learn rosbags torch
pip install onnx onnxscript
```

Notes:

- `torch`, `numpy`, `scikit-learn`, `matplotlib`, and `rosbags` are used directly by the repo.
- `onnx` and `onnxscript` are useful when artifact export is enabled.
- `train_cnn_common.py` will use `mps`, then `cuda`, then `cpu`, whichever is available.

## Generate Processed Datasets

Run the parser from the repo root:

```powershell
python training\rosbag_parser.py --bag-name go2_data_fb_1 --keep-pair 12
python training\rosbag_parser.py --bag-name go2_data_lr_3 --keep-pair 34
python training\rosbag_parser.py --bag-name go2_data_ud_6 --keep-pair 56 --window-ms 300 --output-tag _w300
```

Useful parser flags:

- `--bag-name`: bag folder under `bag_data/raw_bag/`
- `--keep-pair`: one of `12`, `34`, `56`
- `--exclude-sec`: neutral exclusion buffer around nonzero segments
- `--window-ms`: history length in milliseconds
- `--sampling-hz`: resampling frequency for each history window
- `--downsample-zero-class` / `--no-downsample-zero-class`
- `--output-tag`: suffix added to the output filename, such as `_w300`

Example output files:

```text
bag_data/processed_data/up_down/X_ud_6_w300.npy
bag_data/processed_data/up_down/y_ud_6_w300.npy
```

## Train the Models

Each training script is a complete experiment configuration. Edit the `TrainingConfig` in the script you care about, then run it.

```powershell
python training\train_cnn_012.py
python training\train_cnn_034.py
python training\train_cnn_056.py
```

Current experiment defaults:

| Script | Intent pair | Raw labels | Current selected features | Split strategy |
| --- | --- | --- | --- | --- |
| `train_cnn_012.py` | front/back | `(0, 1, 2)` | `arm_angles`, `arm_currents` | derived split from `fb_1` and `fb_2` |
| `train_cnn_034.py` | left/right | `(0, 3, 4)` | `ff`, `accel`, `dq` | train on `lr_1`, `lr_2`; test on `lr_3` |
| `train_cnn_056.py` | up/down | `(0, 5, 6)` | `arm_angles`, `arm_currents`, `ff`, `accel` | train on `ud_2`-`ud_5`; split `ud_6_w300` into val/test |

Training output includes:

- run configuration summary
- per-epoch loss and accuracy
- confusion matrix in raw labels
- classification report in raw labels
- optional learning curves

## Model Architecture

`training/train_cnn_common.py` defines a configurable 1D CNN:

- stacked `Conv1d + BatchNorm1d + ReLU`
- optional `MaxPool1d` after selected layers
- flattened classifier head with dropout
- weighted cross-entropy for class imbalance
- optional validation-driven early stopping
- optional `ReduceLROnPlateau`
- optional delta features and gravity compensation

Input tensors are normalized with training-set statistics and rearranged to `NCT` format before training and export.

## Exported Artifacts

When `export_artifacts=True`, training writes files to:

```text
bag_data/processed_data/<intent_pair>/models/
```

Artifacts:

- `*.pt`: PyTorch checkpoint with model state and preprocessing metadata
- `*.onnx`: exported ONNX model
- `*.deploy.yaml`: deployment metadata, including:
  - selected feature slices
  - raw-label to class-index mapping
  - normalization statistics
  - ONNX input/output names
  - timestep and feature dimensions

## A Typical Workflow

1. Drop a new bag into `bag_data/raw_bag/`
2. Run `training/rosbag_parser.py` with the correct `--keep-pair`
3. Verify the output landed in the expected processed-data folder
4. Update the relevant training script with the new dataset filenames
5. Train the model
6. Turn on artifact export if you want fresh `.pt`, `.onnx`, and `.deploy.yaml` files

## Gotchas

- All train/val/test datasets for a run must have the same timestep count and feature layout.
- The parser keeps selected labels in raw numbering instead of collapsing them to `1` and `2`; the training code remaps them internally.
- `train_cnn_012.py` currently has `export_artifacts=False`, so it will not write fresh model files unless you change that setting.
- There is no packaged library API yet; the scripts are the intended entry points.

## Next Improvements Worth Making

- add a pinned dependency file
- add a small evaluation script for running a saved `.pt` or `.onnx` model on new windows
- store experiment metrics alongside each exported model
- document the upstream meaning of labels `1` through `6` in one place
