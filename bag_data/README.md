# Data Directory

This folder is intentionally committed as structure only.

Do not commit raw bags, processed `.npy` datasets, model weights, ONNX exports,or deployment YAML files. They are ignored by `.gitignore`.

Expected local layout:

```text
bag_data/
  raw_bag/
    go2_data_ud_*/
    go2_data_lr_*/
    go2_data_fb_*/
  processed_data/
    front_back/
      models/
    left_right/
      models/
    up_down/
      models/
```

Use `training/rosbag_parser.py` to create processed `.npy` datasets from local raw bags. Use the `training/train_cnn_*.py` scripts to create local model artifacts under the matching `models/` folder.
