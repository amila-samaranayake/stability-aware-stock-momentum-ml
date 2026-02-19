# src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd


def plot_equity_curve(equity_curve: pd.Series, title: str = "Equity Curve"):
    plt.figure(figsize=(10, 5))
    equity_curve.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_drawdown(equity_curve: pd.Series, title: str = "Drawdown"):
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0

    plt.figure(figsize=(10, 5))
    drawdown.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_turnover(turnover_series: pd.Series, title: str = "Turnover"):
    plt.figure(figsize=(10, 5))
    turnover_series.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Turnover")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
