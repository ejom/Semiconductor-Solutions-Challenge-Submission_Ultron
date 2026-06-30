# IR-Drop Prediction with a Fully Convolutional U-Net

This project builds tensor datasets from the **ML-for-IR-drop benchmark**, trains a **fully convolutional U-Net-style model** for **IR-drop map prediction**, evaluates a saved checkpoint, and generates prediction plots for the hidden split.

The workflow is split across three scripts:

- `build_tensors.py` — converts raw CSV benchmark data into PyTorch tensor files
- `fcn_ir_drop.py` — trains the model or runs evaluation only
- `plot_predictions.py` — generates prediction plots and reports detailed metrics

The model uses 3 spatial input channels and predicts a continuous IR-drop map of the same spatial size. Although the architecture is U-Net-like, the task is **dense regression**, not segmentation.

---

## Repository Structure
.
├── build_tensors.py
├── fcn_ir_drop.py
├── plot_predictions.py
├── requirements.txt
├── checkpoints/
│   └── best_model.pt
├── tensors/
│   ├── fake.pt
│   ├── real.pt
│   └── hidden.pt
└── results/
    └── plots/

`plot_predictions.py` supports `--ckpt`; `fcn_ir_drop.py` uses a fixed `CKPT_PATH` in code by default of `checkpoints/best_model.pt`. 

---

## Data Layout

`build_tensors.py` expects the benchmark root to contain:

```text
Data/ML-for-IR-drop/benchmarks/
├── fake-circuit-data/
├── real-circuit-data/
└── hidden-real-circuit-data/
```

### Fake split layout

```text
fake-circuit-data/
├── current_map1_current.csv
├── current_map1_eff_dist.csv
├── current_map1_pdn_density.csv
├── current_map1_ir_drop.csv
├── current_map2_current.csv
...
```

### Real and hidden split layout

```text
real-circuit-data/
├── testcase1/
│   ├── current_map.csv
│   ├── eff_dist_map.csv
│   ├── pdn_density.csv
│   └── ir_drop_map.csv
├── testcase2/
...
```

`build_tensors.py` computes normalization statistics from the **fake split only**, then applies those statistics to all splits before writing tensor files. 

---

## Installation

### Create a virtual environment and install dependencies

#### Bash (Linux / macOS / WSL)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Inputs and Output

### Input channels

* `current_map`
* `eff_dist`
* `pdn_density`

### Output

* `ir_drop`

The model input shape is `(B, 3, H, W)` and the output shape is `(B, 1, H, W)`. Because the network is fully convolutional, it can handle varying spatial dimensions. 

---

# How to Run

## 1. Build tensor files from raw CSV benchmark data

This converts the raw benchmark CSVs into:

* `tensors/fake.pt`
* `tensors/real.pt`
* `tensors/hidden.pt` 

### Bash command

```bash
python build_tensors.py
```

### With a custom dataset location

```bash
python build_tensors.py --data_root /path/to/benchmarks --out_dir tensors
```

### What this script does

* loads raw CSV benchmark data
* computes channel-wise normalization stats from the fake split
* z-score normalizes all splits
* saves tensor lists because samples may have different `(H, W)` sizes 

---

## 2. Train the model

This trains the U-Net-style FCN and saves the best checkpoint to:

```text
checkpoints/best_model.pt
```

unless you change the path in code. 

### Bash command

```bash
python fcn_ir_drop.py
```

### Train with custom hyperparameters

```bash
python fcn_ir_drop.py --epochs 200 --base_ch 64 --lr 3e-4
```

### What training does

* loads `tensors/fake.pt` as training data
* loads `tensors/real.pt` as validation data
* loads `tensors/hidden.pt` as test data
* trains the model
* saves the checkpoint with the best validation RMSE
* evaluates the best checkpoint at the end on validation and hidden test data 

---

## 3. Run evaluation only

If you already have a saved checkpoint and want to skip training:

### Bash command

```bash
python fcn_ir_drop.py --eval_only
```

This loads the checkpoint and evaluates on:

* `real.pt` as validation
* `hidden.pt` as test 

### Important checkpoint note

By default, `fcn_ir_drop.py` expects:

```text
checkpoints/best_model.pt
```

---

## 4. Generate prediction plots

This script loads a trained checkpoint, runs inference on the hidden split, and saves plots for each sample plus a summary figure. 

### Bash command

```bash
python plot_predictions.py
```

### Plot with a custom checkpoint

```bash
python plot_predictions.py --ckpt checkpoints/best_model.pt
```

### Plot with your checkpoint name

```bash
python plot_predictions.py --ckpt checkpoints/best_model_CE_92.pt
```

### Change the hotspot threshold

```bash
python plot_predictions.py --threshold_pct 5
```

### Change output directory

```bash
python plot_predictions.py --out_dir results/plots
```

### What plotting does

For each hidden sample, it saves a figure containing:

* Ground Truth
* Prediction
* Error map
* Ground Truth vs Prediction scatter plot

It also computes:

### Regression metrics

* RMSE
* MAE
* NRMSE
* R²

### Hotspot metrics

* Precision
* Recall
* F1 Score

Hotspots are defined using the top-K% of ground-truth pixels, with `K=10` by default. 

---

## Full Bash Workflow

### From scratch

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python build_tensors.py
python fcn_ir_drop.py
python plot_predictions.py
```

### If you already have a trained checkpoint

```bash
source .venv/bin/activate

python fcn_ir_drop.py --eval_only
python plot_predictions.py --ckpt checkpoints/best_model.pt
```

## Model Summary

`fcn_ir_drop.py` defines a **4-level U-Net-style fully convolutional network** with:

* encoder blocks
* bottleneck
* decoder blocks with skip connections
* final `1x1` conv head for single-channel regression output 

This is a **dense regression** model, not a segmentation model, because the output is a continuous IR-drop map and the training loss is MSE. 

---

## Notes and Caveats

* Run `build_tensors.py` before training or plotting unless the tensor files already exist.   
* `base_ch` must match the value used when the checkpoint was trained, or weight loading may fail.  
* Because sample sizes vary, the dataloader uses `batch_size=1` with a custom collate function. 
* `plot_predictions.py` denormalizes the predicted IR-drop map back into original units before plotting. 

---

## Outputs

After running the full pipeline, you should have:

### Tensor files

```text
tensors/fake.pt
tensors/real.pt
tensors/hidden.pt
```

### Checkpoint

```text
checkpoints/best_model.pt
```

or your custom checkpoint name.

### Plots

```text
results/plots/
├── hidden_testcase_*.png
└── summary.png
```

---

## Script Summary

### `build_tensors.py`

Builds normalized tensor datasets from raw benchmark CSV files. 

### `fcn_ir_drop.py`

Trains or evaluates the IR-drop prediction model. 

### `plot_predictions.py`

Generates hidden-split prediction visualizations and reports aggregate metrics. 
