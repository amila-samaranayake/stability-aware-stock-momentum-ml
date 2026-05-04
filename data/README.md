# Stability-Aware Stock Momentum ML — End-to-End Run Guide

## Overview
This guide describes the full project workflow from raw data download to final model comparison notebooks.

The pipeline is organized so that:
- raw and processed data are generated first
- feature datasets are built next
- model experiments are run after that
- diagnostics and final notebooks are run last

This order keeps outputs reproducible and makes it easier to rerun the entire project from scratch.

---

## 1. Environment setup

Create and activate the virtual environment, then install dependencies.

### Windows PowerShell
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Check that the project imports correctly:
```bash
python -c "import src; print('src import ok')"
```

---

## 2. Project structure

Important folders:

- `data/raw/` → raw downloaded market data
- `data/processed/` → processed returns and feature datasets
- `experiments/results/` → experiment outputs from model runs
- `reports/tables/` → final notebook-exported tables
- `reports/figures/` → final notebook-exported plots
- `src/` → pipeline, feature, model, analysis, and utility code

---

## 3. Configure the project

Before running anything, check `src/config.py`.

Important settings include:
- ticker universe
- start and end dates
- train/test split dates
- whether simple or log returns are used
- market ticker
- top portfolio percentage
- feature-source-specific settings
- model hyperparameters

Recommended check:
- verify raw paths
- verify processed output paths
- verify `FEATURE_SOURCE` before model runs

---

## 4. Download raw data

### 4.1 Download adjusted close and OHLCV
Run:
```bash
python -m src.run_download_ohlcv
```

Expected outputs:
- `data/raw/ohlcv_2015_2025.parquet`
- `data/raw/adj_close_2015_2025.parquet`

Optional sanity check:
```bash
python -m src.check_ohlcv
```

---

## 5. Build returns datasets

Run the preprocessing pipeline to generate daily and monthly returns.

```bash
python -m src.run_pipeline
```

Expected outputs include processed returns under `data/processed/returns/`, such as:
- `returns_daily.parquet`
- `returns_monthly.parquet`
- `train_monthly_2015_2024.parquet`
- `test_monthly_2025.parquet`

This step also handles:
- adjusted-close return construction
- monthly compounding
- train/test split creation

---

## 6. Build feature datasets

### 6.1 Build the adjusted-close daily feature dataset
```bash
python -m src.run_features_daily
```

Expected outputs under:
- `data/processed/features_daily/`

Main files:
- `ml_full_daily.parquet`
- `ml_train_daily_2015_2024.parquet`
- `ml_test_daily_2025.parquet`

### 6.2 Build the OHLCV-enhanced daily feature dataset
```bash
python -m src.run_features_daily_ohlcv
```

Expected outputs under:
- `data/processed/features_daily_ohlcv/`

Main files:
- `ml_full_daily_ohlcv.parquet`
- `ml_train_daily_ohlcv_2015_2024.parquet`
- `ml_test_daily_ohlcv_2025.parquet`

### 6.3 Build LSTM sequence dataset
If the LSTM sequence dataset is built with a separate pipeline, run the sequence dataset generation step used in the project.

For example:
```bash
python -m src.run_features_lstm
```

If the actual module name differs in your codebase, use the sequence-dataset generation script that creates:
- `lstm_train_daily_2015_2024.npz`
- `lstm_test_daily_2025.npz`
- related metadata parquet files

---

## 7. Run benchmark strategy

The benchmark is the monthly baseline momentum strategy.

```bash
python -m src.run_baseline
```

Expected output folder:
- `experiments/results/exp01_baseline_monthly/`

Expected outputs include:
- `metrics_train.json`
- `metrics_test_2025.json`
- `metrics_train_with_costs.json`
- `metrics_test_2025_with_costs.json`
- `equity_train.csv`
- `equity_test_2025.csv`

---

## 8. Run Ridge models

### 8.1 Ridge with `daily` features
Set in `src/config.py`:
```python
FEATURE_SOURCE = "daily"
```
Run:
```bash
python -m src.run_linear
```

Expected output folder:
- `experiments/results/exp02_linear_ridge_daily/`

### 8.2 Ridge with `daily_ohlcv` features
Set in `src/config.py`:
```python
FEATURE_SOURCE = "daily_ohlcv"
```
Run:
```bash
python -m src.run_linear
```

Expected output folder:
- `experiments/results/exp02_linear_ridge_daily_ohlcv/`

---

## 9. Run XGBoost models

### 9.1 XGBoost with `daily` features
Set:
```python
FEATURE_SOURCE = "daily"
```
Run:
```bash
python -m src.run_xgboost_rolling
```

Expected output folder:
- `experiments/results/exp03_xgboost_rolling_daily/`

### 9.2 XGBoost with `daily_ohlcv` features
Set:
```python
FEATURE_SOURCE = "daily_ohlcv"
```
Run:
```bash
python -m src.run_xgboost_rolling
```

Expected output folder:
- `experiments/results/exp03_xgboost_rolling_daily_ohlcv/`

---

## 10. Run Random Forest models

### 10.1 Random Forest with `daily` features
Set:
```python
FEATURE_SOURCE = "daily"
```
Run:
```bash
python -m src.run_tree_rolling
```

Expected output folder:
- `experiments/results/exp04_random_forest_rolling_daily/`

### 10.2 Random Forest with `daily_ohlcv` features
Set:
```python
FEATURE_SOURCE = "daily_ohlcv"
```
Run:
```bash
python -m src.run_tree_rolling
```

Expected output folder:
- `experiments/results/exp04_random_forest_rolling_daily_ohlcv/`

---

## 11. Run MLP models

### 11.1 MLP with `daily` features
Set:
```python
FEATURE_SOURCE = "daily"
```
Run:
```bash
python -m src.run_nn_mlp
```

Expected output folder:
- `experiments/results/exp05_nn_mlp_daily/`

### 11.2 MLP with `daily_ohlcv` features
Set:
```python
FEATURE_SOURCE = "daily_ohlcv"
```
Run:
```bash
python -m src.run_mlp
```

Expected output folder:
- `experiments/results/exp05_nn_mlp_daily_ohlcv/`

---

## 12. Run LSTM model

Run the LSTM experiment after the LSTM sequence dataset has been generated.

```bash
python -m src.run_lstm
```

Expected output folder:
- `experiments/results/exp06_lstm_daily/`

Expected outputs include:
- `prediction_metrics.json`
- `metrics_train.json`
- `metrics_test_2025.json`
- `metrics_train_with_costs.json`
- `metrics_test_2025_with_costs.json`
- `equity_train.csv`
- `equity_test_2025.csv`
- `training_history.json`
- `loss_curve.png`
- `pred_vs_actual_train.png`
- `pred_vs_actual_test_2025.png`

---

## 13. Optional tuning steps

These steps are optional and mainly used for research comparison, not necessarily for the final selected model.

### 13.1 Random Forest grid-style tuning
```bash
python -m src.tunings.run_random_forest_tuning
```

### 13.2 Random Forest Optuna tuning
Set `FEATURE_SOURCE` as needed, typically `daily_ohlcv`, then run:
```bash
python -m src.tunings.run_random_forest_optuna
```

### 13.3 XGBoost tuning
```bash
python -m src.tunings.run_xgboost_tuning
```

---

## 14. Optional diagnostics

These scripts are useful for deeper inspection, but they are not required for the main experiment outputs.

### 14.1 Prediction diagnostics
Example:
```bash
python -m src.analysis.prediction_diagnostics --predictions-path experiments/results/exp04_random_forest_rolling_daily_ohlcv/test_predictions.csv --output-dir experiments/results/exp04_random_forest_rolling_daily_ohlcv/diagnostics
```

### 14.2 Prediction postprocessing experiments
Example:
```bash
python -m src.analysis.postprocess_predictions --train-predictions ... --test-predictions ... --output-dir ...
```

These folders are mainly for checking and analysis and do not have to be used in the final notebook pipeline.

---

## 15. Run the final notebooks

Recommended notebook order:

1. `01_data_download_preprocessing_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_baseline_momentum.ipynb`
4. `04_linear_ridge.ipynb`
5. `05_random_forest.ipynb`
6. `06_xgboost.ipynb`
7. `07_neural_models_mlp_lstm.ipynb`
8. `08_final_comparison_and_model_selection.ipynb`

Notebook outputs are exported to:
- `reports/tables/`
- `reports/figures/`

---

## 16. Recommended full rerun order

For a clean rerun from scratch, use this order:

```bash
python -m src.run_download_ohlcv
python -m src.check_ohlcv
python -m src.run_pipeline
python -m src.run_features_daily
python -m src.run_features_daily_ohlcv
python -m src.run_features_lstm
python -m src.run_baseline
```

Then run the models in sequence:

```bash
# Ridge
# set FEATURE_SOURCE = daily
python -m src.run_linear
# set FEATURE_SOURCE = daily_ohlcv
python -m src.run_linear

# XGBoost
# set FEATURE_SOURCE = daily
python -m src.run_xgboost_rolling
# set FEATURE_SOURCE = daily_ohlcv
python -m src.run_xgboost_rolling

# Random Forest
# set FEATURE_SOURCE = daily
python -m src.run_tree_rolling
# set FEATURE_SOURCE = daily_ohlcv
python -m src.run_tree_rolling

# MLP
# set FEATURE_SOURCE = daily
python -m src.run_mlp
# set FEATURE_SOURCE = daily_ohlcv
python -m src.run_mlp

# LSTM
python -m src.run_lstm
```

Then run notebooks in order.

---

## 17. Final selected model

Based on the current project findings, the final selected model is:

**Random Forest rolling + daily_ohlcv features + top 20% equal-weight portfolio**

This guide still includes all model families because the final report and comparison notebooks require the full experiment set.

---

## 18. Notes

- The project uses different prediction-metric JSON structures across model families.
  - Ridge / MLP / LSTM use `train` and `test_2025`
  - RF / XGBoost rolling use `train_static_fit` and `test_2025_rolling_fit`
- Diagnostics folders are optional and mainly used for deeper checking.
- Final polished outputs should be exported from notebooks into `reports/`, not mixed into the experiment folders.

