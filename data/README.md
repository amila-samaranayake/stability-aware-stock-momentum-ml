# Stability-Aware Stock Momentum ML

This project investigates whether machine-learning models can improve a simple momentum-based stock-selection strategy once turnover and transaction costs are included.

Using daily market data for 100 UK-listed equities from 2015 to 2025, the project frames the task as **monthly cross-sectional stock ranking**. At each month-end, models predict next-month stock returns, rank the stock universe, and form an equal-weight portfolio from the top 20% of predicted stocks.

The study compares:
- a simple momentum benchmark
- Ridge regression
- Random Forest (RF)
- XGBoost (XGB)
- multilayer perceptron (MLP)
- long short-term memory network (LSTM)

Two tabular feature sets are evaluated:
- **Daily** price-derived features
- **Daily+OHLCV** features, which add trading-volume and intraday price-structure information

LSTM is evaluated separately on a rolling sequence dataset.

The main finding is that the **momentum benchmark remains very strong once implementation realism is considered**, while **Random Forest with Daily+OHLCV features** is the strongest machine-learning model in out-of-sample portfolio selection.

---

## Repository highlights

- reproducible pipeline from raw data download to final report-ready figures
- multiple model families under one consistent evaluation framework
- train/test comparison with turnover and transaction costs
- portfolio-selection diagnostics beyond forecast error alone
- notebooks for EDA, feature engineering, model comparison, and final reporting outputs

---

## Project structure

Important folders:

- `data/raw/` → raw downloaded market data
- `data/processed/` → processed returns and feature datasets
- `experiments/results/` → experiment outputs from model runs
- `reports/tables/` → final notebook-exported tables
- `reports/figures/` → final notebook-exported plots
- `src/` → pipeline, feature, model, analysis, and utility code

---

## End-to-End Run Guide

This guide describes the full project workflow from raw data download to final model-comparison notebooks.

The pipeline is organised so that:
1. raw and processed data are generated first
2. feature datasets are built next
3. model experiments are run after that
4. diagnostics and final notebooks are run last

This order keeps outputs reproducible and makes it easier to rerun the full project from scratch.

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

## 2. Configure the project

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

Recommended checks:
- verify raw paths
- verify processed output paths
- verify `FEATURE_SOURCE` before each model run
- verify that the split is **train = 2015–2024** and **test = 2025**

> **Note:** before each tabular-model experiment, update `FEATURE_SOURCE` in `src/config.py` to match the intended dataset (`"daily"` or `"daily_ohlcv"`). The output experiment directory is created automatically from this setting.

---

## 3. Download raw data

### 3.1 Download adjusted close and OHLCV

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

## 4. Build returns datasets

Run the preprocessing pipeline to generate daily and monthly returns.

```bash
python -m src.run_pipeline
```

Expected outputs under `data/processed/returns/` include:
- `returns_daily.parquet`
- `returns_monthly.parquet`
- `train_monthly_2015_2024.parquet`
- `test_monthly_2025.parquet`

This step handles:
- adjusted-close return construction
- monthly compounding
- train/test split creation

---

## 5. Build feature datasets

### 5.1 Build the Daily feature dataset

```bash
python -m src.run_features_daily
```

Expected outputs under:
- `data/processed/features_daily/`

Main files:
- `ml_full_daily.parquet`
- `ml_train_daily_2015_2024.parquet`
- `ml_test_daily_2025.parquet`

### 5.2 Build the Daily+OHLCV feature dataset

```bash
python -m src.run_features_daily_ohlcv
```

Expected outputs under:
- `data/processed/features_daily_ohlcv/`

Main files:
- `ml_full_daily_ohlcv.parquet`
- `ml_train_daily_ohlcv_2015_2024.parquet`
- `ml_test_daily_ohlcv_2025.parquet`

### 5.3 Build the LSTM sequence dataset

Run:

```bash
python -m src.run_features_lstm
```

Expected outputs include:
- `data/processed/features_lstm/lstm_train_daily_2015_2024.npz`
- `data/processed/features_lstm/lstm_test_daily_2025.npz`

---

## 6. Run benchmark strategy

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

> If your baseline folder name differs in your local project version, update this README to match the exact folder used in the repository.

---

## 7. Run Ridge models

### 7.1 Ridge with Daily features

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

### 7.2 Ridge with Daily+OHLCV features

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

## 8. Run XGBoost models

### 8.1 XGBoost with Daily features

Set in `src/config.py`:

```python
FEATURE_SOURCE = "daily"
```

Run:

```bash
python -m src.run_xgboost_rolling
```

Expected output folder:
- `experiments/results/exp03_xgboost_rolling_daily/`

### 8.2 XGBoost with Daily+OHLCV features

Set in `src/config.py`:

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

## 9. Run Random Forest models

### 9.1 Random Forest with Daily features

Set in `src/config.py`:

```python
FEATURE_SOURCE = "daily"
```

Run:

```bash
python -m src.run_tree_rolling
```

Expected output folder:
- `experiments/results/exp04_random_forest_rolling_daily/`

### 9.2 Random Forest with Daily+OHLCV features

Set in `src/config.py`:

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

## 10. Run MLP models

### 10.1 MLP with Daily features

Set in `src/config.py`:

```python
FEATURE_SOURCE = "daily"
```

Run:

```bash
python -m src.run_nn_mlp
```

Expected output folder:
- `experiments/results/exp05_nn_mlp_daily/`

### 10.2 MLP with Daily+OHLCV features

Set in `src/config.py`:

```python
FEATURE_SOURCE = "daily_ohlcv"
```

Run:

```bash
python -m src.run_nn_mlp
```

Expected output folder:
- `experiments/results/exp05_nn_mlp_daily_ohlcv/`

---

## 11. Run LSTM model

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

## 12. Optional tuning steps

These steps are optional and mainly used for research comparison rather than the final selected pipeline.

### 12.1 Random Forest grid-style tuning

```bash
python -m src.tunings.run_random_forest_tuning
```

### 12.2 Random Forest Optuna tuning

Set `FEATURE_SOURCE` as needed, typically `daily_ohlcv`, then run:

```bash
python -m src.tunings.run_random_forest_optuna
```

### 12.3 XGBoost tuning

```bash
python -m src.tunings.run_xgboost_tuning
```

---

## 13. Optional diagnostics

These scripts are useful for deeper inspection, but they are not required for the main experiment outputs.

### 13.1 Prediction diagnostics

Example usage:

```bash
python -m src.analysis.prediction_diagnostics --predictions-path experiments/results/exp04_random_forest_rolling_daily_ohlcv/test_predictions.csv --output-dir experiments/results/exp04_random_forest_rolling_daily_ohlcv/diagnostics
```

### 13.2 Prediction postprocessing experiments

Example usage:

```bash
python -m src.analysis.postprocess_predictions --train-predictions ... --test-predictions ... --output-dir ...
```

These folders are mainly for checking and analysis and do not have to be used in the final notebook pipeline.

---

## 14. Run the notebooks

### Core analysis notebooks
1. `01_data_download_preprocessing_eda.ipynb`
2. `02_feature_engineering.ipynb`
3. `03_baseline_momentum.ipynb`
4. `04_linear_ridge.ipynb`
5. `05_random_forest.ipynb`
6. `06_xgboost.ipynb`
7. `07_neural_models_mlp_lstm.ipynb`
8. `08_final_comparison_and_model_selection.ipynb`

### Final reporting notebooks
9. `09_portfolio_selection_diagnostics.ipynb`
10. `10_final_comparison_plots.ipynb`
11. `11_final_report_ready_figures.ipynb`

Notebook outputs are exported to:
- `reports/tables/`
- `reports/figures/`

---

## 15. Recommended full rerun order

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
python -m src.run_nn_mlp
# set FEATURE_SOURCE = daily_ohlcv
python -m src.run_nn_mlp

# LSTM
python -m src.run_lstm
```

Then run notebooks in order.

---

## 16. Expected final outputs

After completing the full pipeline, the main outputs should include:
- processed returns datasets under `data/processed/returns/`
- feature datasets under `data/processed/features_*`
- experiment outputs under `experiments/results/`
- final comparison figures under `reports/figures/`
- final comparison tables under `reports/tables/`

---

## 17. Final selected model

Based on the current project findings, the final selected model is:

**Random Forest rolling + Daily+OHLCV features + top 20% equal-weight portfolio**

This guide still includes all model families because the final report and comparison notebooks require the full experiment set.

---

## 18. Notes

- Different model families use slightly different prediction-metric JSON structures:
  - Ridge / MLP / LSTM use `train` and `test_2025`
  - RF / XGBoost rolling use `train_static_fit` and `test_2025_rolling_fit`
- Diagnostics folders are optional and mainly used for deeper checking.
- Final polished outputs should be exported from notebooks into `reports/`, not mixed into the experiment folders.

---

## 19. Reproducibility note

Some models, especially neural models and tree-based rolling experiments, may show small run-to-run variation depending on random seeds, library versions, and hardware environment. Fixed seeds are used where possible, but exact numeric replication may still vary slightly across systems.

---

## 20. GitHub and code availability

This repository contains the full codebase used for:
- raw market-data download
- preprocessing and return construction
- feature engineering
- model training
- rolling evaluation
- portfolio backtesting
- transaction-cost analysis
- diagnostics
- final figure and table generation

The repository was used as the primary code archive for the dissertation project and is intended to support transparency, reproducibility, and future extension.
