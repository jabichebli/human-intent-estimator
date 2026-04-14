# Human Intent Estimator

Train 1D CNNs that estimate push intent around a Unitree Go2 from ROS 2 bag data.

Commands below assume you are in the repo root.

## Quick Start

If you just want to train the current up/down model using the processed `.npy` files already in this repo, run:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnx,onnxscript python training\train_cnn_056.py
```

That is a good default right now. You do **not** need a separate virtualenv if you use `uv run`.

If you already have your own Python environment set up with the dependencies installed, the command is just:

```powershell
python training\train_cnn_056.py
```

## Up/Down Model Selection

The current default in `training\train_cnn_056.py` is the `_w300_e060_hpure` up/down dataset.
It uses:

- a `0.60 s` exclusion buffer around nonzero segments
- a history-purity filter so a `300 ms` window is dropped if its history crosses a label transition
- file-balanced training sampling (`uniform_files`)
- appended per-window delta features on top of the raw features

That choice is intentional:

- `train_cnn_056.py` is mainly for training and exporting the current best up/down model
- its built-in validation/test path uses a derived split inside bag `6`
- final dataset selection should be based on the stricter whole-bag holdout evaluation, not only the derived split

In practice, this means:

- `_w300_e060_hpure` + appended delta features is the strongest recipe tested so far
- the leave-one-bag-out sweep is still the right check for deployment confidence, because the derived split inside one bag is easier than a true unseen-bag test

Use this command for the current swapped whole-bag check:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnxscript python training\eval_up_down_wholebag.py
```

To compare another tag:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnxscript python training\eval_up_down_wholebag.py --tag _w300_e045
```

For a broader deployment-style check across all up/down bags:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnxscript python training\eval_up_down_lobo.py
```

Rule of thumb:

- use `train_cnn_056.py` to train/export the current up/down model
- use `eval_up_down_wholebag.py` to decide which processed up/down dataset is actually best

## Which Script Trains What?

| Script | Labels | Meaning |
| --- | --- | --- |
| `training\train_cnn_012.py` | `0,1,2` | front/back |
| `training\train_cnn_034.py` | `0,3,4` | left/right |
| `training\train_cnn_056.py` | `0,5,6` | up/down |

So if you want:

- front/back: `python training\train_cnn_012.py`
- left/right: `python training\train_cnn_034.py`
- up/down: `python training\train_cnn_056.py`

With `uv`, the equivalent pattern is:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnx,onnxscript python training\train_cnn_034.py
```

If `export_artifacts=False` in the script, you can usually drop `onnx` and `onnxscript`.

## When Do I Need The Parser?

You only need `training\rosbag_parser.py` when you want to create new processed datasets from raw bags.

If you are training with the existing `.npy` files already under `bag_data\processed_data\`, skip this step.

Example for rebuilding the up/down dataset:

```powershell
uv run --with numpy,rosbags python training\rosbag_parser.py --bag-name go2_data_ud_6 --keep-pair 56 --window-ms 300 --output-tag _w300
```

More examples:

```powershell
uv run --with numpy,rosbags python training\rosbag_parser.py --bag-name go2_data_fb_1 --keep-pair 12
uv run --with numpy,rosbags python training\rosbag_parser.py --bag-name go2_data_lr_3 --keep-pair 34
```

Raw bags live in `bag_data\raw_bag\`. Processed datasets are written to:

- `bag_data\processed_data\front_back\`
- `bag_data\processed_data\left_right\`
- `bag_data\processed_data\up_down\`

## What The Training Scripts Expect

Each training script already contains its own `TrainingConfig`, including:

- which `.npy` files to use
- which features to train on
- train/val/test split strategy
- model size and optimization settings
- whether to export `.pt`, `.onnx`, and `.deploy.yaml`

The current `train_cnn_056.py` setup trains on the processed up/down datasets and exports model artifacts.

## Outputs

When artifact export is enabled, files are written under:

```text
bag_data\processed_data\<task>\models\
```

Typical outputs:

- `*.pt`
- `*.onnx`
- `*.deploy.yaml`

## Minimal Workflow

1. If you already have processed `.npy` files, run the training script directly.
2. If you only have raw bags, run `rosbag_parser.py` first.
3. Edit the matching `train_cnn_0xx.py` file if you want different datasets or hyperparameters.
4. Train again.

## Repo Layout

```text
training\rosbag_parser.py    # raw bag -> X_*.npy and y_*.npy
training\train_cnn_common.py # shared training / eval / export code
training\train_cnn_012.py    # front/back model
training\train_cnn_034.py    # left/right model
training\train_cnn_056.py    # up/down model
training\eval_up_down_wholebag.py # stricter swapped whole-bag evaluation for up/down
training\eval_up_down_lobo.py # leave-one-bag-out evaluation for up/down
```
