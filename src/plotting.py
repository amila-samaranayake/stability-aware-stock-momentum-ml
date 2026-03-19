# src/plotting.py

import matplotlib.pyplot as plt
import pandas as pd
import os


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

def plot_series(series: pd.Series, title: str, ylabel: str, save_path: str | None = None):
    plt.figure(figsize=(10, 5))
    series.plot()
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
    plt.show()


def compute_drawdown(equity_curve: pd.Series) -> pd.Series:
    running_max = equity_curve.cummax()
    return equity_curve / running_max - 1.0
