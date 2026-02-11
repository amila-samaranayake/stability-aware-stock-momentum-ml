# src/config.py

"""
Global configuration for Stability-Aware Stock Momentum ML project.
"""

# =========================
# DATE CONFIGURATION
# =========================

START_DATE = "2015-01-01"
END_DATE = "2025-12-31"

TRAIN_END_DATE = "2024-12-31"
TEST_START_DATE = "2025-01-01"


# =========================
# REBALANCING CONFIGURATION
# =========================

# Monthly rebalancing
REBALANCE_FREQUENCY = "M"  # 'M' = monthly


# =========================
# MOMENTUM STRATEGY SETTINGS
# =========================

# Lookback window in months
LOOKBACK_MONTHS = 12  # can later test 6 or 9 as robustness

# Portfolio selection rule
TOP_PERCENTAGE = 0.20  # top 20% of ranked stocks

EQUAL_WEIGHT = True  # equal-weight portfolio


# =========================
# FEATURE SETTINGS (ML)
# =========================

FEATURE_WINDOWS = {
    "lag_1m": 1,
    "lag_3m": 3,
    "lag_6m": 6,
    "vol_3m": 3,
}

USE_LOG_RETURNS = False  # If True, use log returns instead of simple returns


# =========================
# SCALING
# =========================

SCALER_TYPE = "robust"  # options: "robust", "standard"


# =========================
# MODEL SETTINGS
# =========================

MODEL_LIST = [
    "momentum",
    "linear",
    "tree",
    "neural"
]

# Neural network settings (keep small to avoid overfitting and complexity)
NN_HIDDEN_LAYERS = [32, 16]
NN_DROPOUT = 0.2
NN_EPOCHS = 100
NN_BATCH_SIZE = 64


# =========================
# DATA SETTINGS
# =========================

# Placeholder: will be replaced with actual FTSE 100 ticker list
TICKERS = [
    "VOD.L",   # Vodafone
    "AZN.L",   # AstraZeneca
    "BP.L",    # BP
    "HSBA.L",  # HSBC
    "ULVR.L",  # Unilever
]

