# Human Intent Estimator

Train supervised intent classifiers from ROS 2 bag data for the Unitree Go2 with a D1 arm.

Commands assume PowerShell from the repo root:

```powershell
cd C:\Projects\human-intent-estimator
```

Use `uv run` for normal CPU runs. You do not need to activate a virtualenv.

## Current Best Recipes

| Task | Labels | Model | Dataset tag | Features | Inference |
| --- | --- | --- | --- | --- | --- |
| left/right | `0,3,4` = rest/left/right | GRU | `_w500_e060_hpure` | `ff accel dq` | argmax |
| up/down | `0,5,6` = rest/up/down | GRU | `_w500_e060_hpure` | `arm_angles arm_currents` | argmax |

Tag meaning:

- `w500`: each sample contains a `500 ms` history window. At `200 Hz`, that is `100` timesteps.
- `e060`: remove rest/zero samples within `0.60 s` of action segments, so rest labels are cleaner.
- `hpure`: keep only samples whose full history window stays inside one contiguous label segment.
- `argmax`: no extra nonzero confidence threshold; predict the class with the largest model output.

## Rebuild And Check Data

Run this when raw bags changed or you want to regenerate processed `.npy` files.

Up/down:

```powershell
uv run --with numpy,rosbags python training\rebuild_up_down_datasets.py --tag _w500_e060_hpure --window-ms 500 --exclude-sec 0.60
```

Left/right:

```powershell
uv run --with numpy,rosbags python training\rebuild_left_right_datasets.py --tag _w500_e060_hpure --window-ms 500 --exclude-sec 0.60
```

Check for exact duplicate raw/processed datasets without reparsing:

```powershell
uv run --with numpy,rosbags python training\rebuild_up_down_datasets.py --tag _w500_e060_hpure --check-only
uv run --with numpy,rosbags python training\rebuild_left_right_datasets.py --tag _w500_e060_hpure --check-only
```

Good output should say:

```text
No exact duplicate raw storage groups found.
No exact duplicate processed dataset groups found.
```

## Quick Evaluation

Use quick LOBO first. It holds out a few bags and is much faster than full leave-one-bag-out.

Up/down GRU quick check:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_up_down_lobo_grouped.py --tag _w500_e060_hpure --model-type gru --heldout-bags 28 13 9 --epochs 35 --nonzero-threshold none --results-csv logs\quick_ud_gru_w500_e060_hpure_argmax.csv
```

Left/right GRU quick check:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_left_right_lobo.py --tag _w500_e060_hpure --model-type gru --heldout-bags 3 8 11 14 --epochs 35 --nonzero-threshold none --results-csv logs\quick_lr_gru_w500_e060_hpure_argmax.csv
```

Optional CNN baseline, using the same parsed data and argmax inference:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_up_down_lobo_grouped.py --tag _w500_e060_hpure --model-type cnn --heldout-bags 28 13 9 --epochs 35 --nonzero-threshold none --results-csv logs\quick_ud_cnn_w500_e060_hpure_argmax.csv
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_left_right_lobo.py --tag _w500_e060_hpure --model-type cnn --heldout-bags 3 8 11 14 --epochs 35 --nonzero-threshold none --results-csv logs\quick_lr_cnn_w500_e060_hpure_argmax.csv
```

Compare CSV summaries:

```powershell
uv run python training\compare_lobo_results.py logs\quick_ud_gru_w500_e060_hpure_argmax.csv logs\quick_ud_cnn_w500_e060_hpure_argmax.csv
uv run python training\compare_lobo_results.py logs\quick_lr_gru_w500_e060_hpure_argmax.csv logs\quick_lr_cnn_w500_e060_hpure_argmax.csv
```

Prefer the model with stronger `avg_macro_f1`, `min_macro_f1`, and `min_worst_class_f1`.

## Full Evaluation

Run this before trusting a model for hardware. Full LOBO is slow because it retrains once per held-out bag.

Up/down full GRU LOBO:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_up_down_lobo_grouped.py --tag _w500_e060_hpure --model-type gru --nonzero-threshold none --results-csv logs\full_ud_gru_w500_e060_hpure_argmax.csv
```

Left/right full GRU LOBO:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_left_right_lobo.py --tag _w500_e060_hpure --model-type gru --nonzero-threshold none --results-csv logs\full_lr_gru_w500_e060_hpure_argmax.csv
```

Optional full CNN baselines:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_up_down_lobo_grouped.py --tag _w500_e060_hpure --model-type cnn --nonzero-threshold none --results-csv logs\full_ud_cnn_w500_e060_hpure_argmax.csv
uv run --with numpy,torch,scikit-learn,matplotlib python training\eval_left_right_lobo.py --tag _w500_e060_hpure --model-type cnn --nonzero-threshold none --results-csv logs\full_lr_cnn_w500_e060_hpure_argmax.csv
```

Compare:

```powershell
uv run python training\compare_lobo_results.py logs\full_ud_gru_w500_e060_hpure_argmax.csv logs\full_ud_cnn_w500_e060_hpure_argmax.csv
uv run python training\compare_lobo_results.py logs\full_lr_gru_w500_e060_hpure_argmax.csv logs\full_lr_cnn_w500_e060_hpure_argmax.csv
```

## Train And Export

These commands train on all selected bags and export `.pt`, `.onnx`, and `.deploy.yaml`.

Up/down export:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnx,onnxscript python training\train_rnn_056.py --tag _w500_e060_hpure --nonzero-threshold none --artifact-stem intent_up_down_gru_w500_e060_hpure_argmax
```

Left/right export:

```powershell
uv run --with numpy,torch,scikit-learn,matplotlib,onnx,onnxscript python training\train_rnn_034.py --tag _w500_e060_hpure --nonzero-threshold none --artifact-stem intent_left_right_gru_w500_e060_hpure_argmax
```

Check exported files:

```powershell
Get-ChildItem bag_data\processed_data\up_down\models\intent_up_down_gru_w500_e060_hpure_argmax*
Get-ChildItem bag_data\processed_data\left_right\models\intent_left_right_gru_w500_e060_hpure_argmax*
```

Inspect deploy metadata:

```powershell
Get-Content bag_data\processed_data\up_down\models\intent_up_down_gru_w500_e060_hpure_argmax.deploy.yaml
Get-Content bag_data\processed_data\left_right\models\intent_left_right_gru_w500_e060_hpure_argmax.deploy.yaml
```

The YAML should include:

- `input_layout: NCT`
- `input_name: input`
- `output_name: logits`
- `model_type: gru`
- `num_timesteps: 100`
- `window_ms: 500`
- `sampling_hz: 200`
- the correct `raw_label_set`
- the selected feature blocks
- normalization statistics
- delta feature metadata
- `nonzero_prediction_threshold: null` for argmax inference

## Script Map

| Script | Purpose |
| --- | --- |
| `training\rosbag_parser.py` | Convert one raw bag into processed `X/y/t/seg` `.npy` files |
| `training\rebuild_up_down_datasets.py` | Rebuild/check all up/down bags |
| `training\rebuild_left_right_datasets.py` | Rebuild/check all left/right bags |
| `training\eval_up_down_lobo_grouped.py` | Up/down grouped leave-one-bag-out evaluation |
| `training\eval_left_right_lobo.py` | Left/right leave-one-bag-out evaluation |
| `training\train_rnn_056.py` | Train/export up/down GRU |
| `training\train_rnn_034.py` | Train/export left/right GRU |
| `training\compare_lobo_results.py` | Print compact summaries from LOBO CSV files |

## Outputs

Processed datasets live under:

```text
bag_data\processed_data\up_down\
bag_data\processed_data\left_right\
```

Exported models live under:

```text
bag_data\processed_data\up_down\models\
bag_data\processed_data\left_right\models\
```

Typical export files:

```text
*.pt
*.onnx
*.onnx.data
*.deploy.yaml
```
