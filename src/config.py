# src/config.py

from src.tickers_ftse100 import FTSE100_TICKERS

# =========================
# DATES
# =========================
START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"

# =========================
# DATA
# =========================
RAW_ADJ_CLOSE_PATH = "data/raw/adj_close_2015_2025.parquet"
TICKERS = FTSE100_TICKERS
MARKET_TICKER = None

USE_LOG_RETURNS = False

# =========================
# FEATURE SOURCE
# =========================
FEATURE_SOURCE = "daily"   # "monthly" or "daily"

# =========================
# REBALANCING / PORTFOLIO
# =========================
REBALANCE_FREQUENCY = "M"
LOOKBACK_MONTHS = 12
TOP_PERCENTAGE = 0.20
EQUAL_WEIGHT = True
TRANSACTION_COST_RATES = [0.0, 0.001, 0.002]

# =========================
# MONTHLY FEATURE SETTINGS
# =========================
FEATURE_WINDOWS = {
    "lag_1m": 1,
    "lag_3m": 3,
    "lag_6m": 6,
    "lag_12m": 12,
    "vol_3m": 3,
}
RSI_WINDOW_MONTHS = 14
TARGET_HORIZON_MONTHS = 1

# =========================
# DAILY FEATURE SETTINGS
# =========================
DAILY_RETURN_WINDOWS = [5, 20, 60, 120, 252]
DAILY_VOL_WINDOWS = [20, 60, 120]
DAILY_MA_PAIRS = [(20, 60), (60, 252)]
DAILY_HIGH_WINDOWS = [252]
DAILY_DRAWDOWN_WINDOWS = [60]
DAILY_BETA_WINDOWS = [60]
DAILY_RSI_WINDOW = 14

# =========================
# SCALING
# =========================
SCALER_TYPE = "robust"

# =========================
# RIDGE
# =========================
RIDGE_ALPHA = 100.0
RIDGE_ALPHA_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
RIDGE_TUNING_SPLITS = 5
RIDGE_SELECTION_METRIC = "topk_hit_rate"

# =========================
# RANDOM FOREST
# =========================
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 4
RF_MIN_SAMPLES_LEAF = 10
RF_MIN_SAMPLES_SPLIT = 20
RF_MAX_FEATURES = "sqrt"
RF_BOOTSTRAP = True


RF_TUNING_SPLITS = 5
RF_SELECTION_METRIC = "combined_score"  # options: "combined_score", "rmse", "directional_accuracy", "spearman", "topk_hit_rate"

RF_N_ESTIMATORS_GRID = [300, 500]
RF_MAX_DEPTH_GRID = [4, 6, 8]
RF_MIN_SAMPLES_LEAF_GRID = [10, 20, 30]
RF_MIN_SAMPLES_SPLIT_GRID = [20, 40]
RF_MAX_FEATURES_GRID = ["sqrt", 0.5]

# =========================
# XGBOOST
# =========================
XGB_N_ESTIMATORS = 300
XGB_MAX_DEPTH = 4
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_REG_ALPHA = 0.0
XGB_REG_LAMBDA = 1.0
XGB_MIN_CHILD_WEIGHT = 5
XGB_GAMMA =0.0

XGB_TUNING_TRIALS = 40
XGB_TUNING_SPLITS = 5
XGB_SELECTION_METRIC = "topk_hit_rate"  # options: "topk_hit_rate", "spearman", "directional_accuracy", "rmse"

# =========================
# MLP
# =========================
NN_HIDDEN_1 = 64
NN_HIDDEN_2 = 32
NN_HIDDEN_3 = 16

NN_LEARNING_RATE = 0.001
NN_EPOCHS = 100
NN_BATCH_SIZE = 128
NN_VALIDATION_FRACTION = 0.15
NN_EARLY_STOPPING_PATIENCE = 10
NN_LR_PATIENCE = 5
NN_L2 = 1e-4
NN_DROPOUT_1 = 0.20
NN_DROPOUT_2 = 0.15
NN_DROPOUT_3 = 0.10

# =========================
# LSTM
# =========================
# LSTM_SEQUENCE_LENGTH = 120
# LSTM_UNITS = 64
# LSTM_DENSE_UNITS = 32
# LSTM_DROPOUT = 0.10
# LSTM_RECURRENT_DROPOUT = 0.00
# LSTM_L2 = 1e-4
# LSTM_LEARNING_RATE = 0.001
# LSTM_EPOCHS = 100
# LSTM_BATCH_SIZE = 64
# LSTM_VALIDATION_FRACTION = 0.15
# LSTM_EARLY_STOPPING_PATIENCE = 10
# LSTM_LR_PATIENCE = 5
# LSTM_NORMALIZE_PER_SEQUENCE = True
# LSTM_SEQUENCE_LENGTH = 60
# LSTM_MARKET_TICKER = None  # set to the benchmark ticker if available

LSTM_SEQUENCE_LENGTH = 120
LSTM_NORMALIZE_PER_SEQUENCE = True
LSTM_MARKET_TICKER = None

LSTM_UNITS = 64
LSTM_DENSE_UNITS = 32
LSTM_DROPOUT = 0.10
LSTM_RECURRENT_DROPOUT = 0.00
LSTM_L2 = 1e-4
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 100
LSTM_BATCH_SIZE = 64
LSTM_VALIDATION_FRACTION = 0.15
LSTM_EARLY_STOPPING_PATIENCE = 10
LSTM_LR_PATIENCE = 5