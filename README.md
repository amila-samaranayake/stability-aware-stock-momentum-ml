# Stability-Aware Stock Momentum ML

This project evaluates whether machine learning models provide practical improvements over a simple momentum investment strategy, with a focus on **decision stability** and **portfolio turnover**.

## Research Question
Under what market conditions do machine learning models fail to improve simple momentum-based investment strategies when decision stability and turnover are considered?

## Data
- Source: Yahoo Finance (via `yfinance`)
- Period:
  - Training/validation: 2015–2024
  - Final test: 2025 (hold-out)

## Methods
### Baseline
- Momentum strategy using past returns ranking
- Monthly rebalancing, equal-weight portfolio

### Models (comparison)
- Linear model (Ridge / Logistic Regression)
- Tree-based model (Random Forest / XGBoost)
- Neural network (small MLP with regularization + early stopping)
