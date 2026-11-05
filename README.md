# Network Traffic Classification on WUSTL-IIoT-2021

This project detects and classifies malicious traffic in an Industrial IoT (IIoT) network using the [WUSTL-IIoT-2021](https://www.cse.wustl.edu/~jain/iiot2/index.html) dataset. Raw flow records are turned into fixed-length time windows per source–destination IP pair. Four model families are then compared on a five-class problem: `normal`, `DoS`, `Reconn`, `CommInj` and `Backdoor`.

The models were chosen to build on each other:

1. **XGBoost (tabular):** a gradient-boosted tree baseline trained on engineered features, with class weighting to handle the heavy class imbalance.
2. **LSTM and 1D CNN (sequence):** the tabular model ignores how traffic changes over time, so these models look at the window second by second.
3. **GCN (graph):** sequence models ignore how IP addresses relate to each other, so a graph convolutional network learns from the communication graph between hosts.

Every model also comes with an explainability analysis: SHAP for XGBoost and Integrated Gradients for the neural networks.

## Repository Structure

```
.
├── config/
│   └── model.yaml                  # GCN hyperparameters and global seed
├── src/
│   ├── preprocess/
│   │   ├── preprocess_data.ipynb   # Feature engineering, windowing, split, scaling
│   │   └── data_augmentation.ipynb # Same pipeline plus attack-window augmentation
│   └── model/
│       ├── xgboost.ipynb           # XGBoost baseline, Optuna tuning, SHAP
│       ├── lstm.ipynb              # LSTM training and evaluation
│       ├── lstm_explain.ipynb      # Integrated Gradients for the LSTM
│       ├── lstm_model.pt           # Trained LSTM weights
│       ├── cnn.ipynb               # 1D CNN training, evaluation, Integrated Gradients
│       └── gcn.ipynb               # GCN training, evaluation, Integrated Gradients
├── pyproject.toml
└── uv.lock
```

The `data/`, `models/` and `results/` directories are git-ignored. The notebooks create them locally.

## Setup

The project needs Python 3.11 and uses [uv](https://docs.astral.sh/uv/) to manage dependencies.

```bash
uv sync
```

This installs the runtime dependencies.

## Usage

The notebooks use paths relative to their own directory (for example `../../data/processed`), so run each one from the folder it lives in. Run them in this order:

1. **`src/preprocess/preprocess_data.ipynb`** writes the non-augmented arrays (`X_*.npy`, `y_*.npy`, `pairs_*.npy`) and `feature_cols.json` to `data/processed/`. Later notebooks need `feature_cols.json`.
2. **`src/preprocess/data_augmentation.ipynb`** reruns the same pipeline from the raw CSV, adds augmented attack windows to the training set, and writes `augmented_*.npy` to `data/processed/`.
3. **Model notebooks** in `src/model/` can run in any order. They all train on the augmented arrays by default.

| Notebook | Saved artifact |
| --- | --- |
| `xgboost.ipynb` | `models/xgboost_augmented_aggregated_new.joblib` (plus `_params.json` and `_metrics.json`) |
| `gcn.ipynb` | `models/gcn_augmented_best_v2.pt` |
| `cnn.ipynb` | `src/model/models/cnn_best.pt` |
| `lstm.ipynb` | `src/model/lstm_model.pt` |
