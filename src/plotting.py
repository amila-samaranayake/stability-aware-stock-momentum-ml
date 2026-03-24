# src/plotting.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

FIGSIZE = (11, 6)
DPI = 300
TITLE_SIZE = 15
LABEL_SIZE = 12
TICK_SIZE = 10
LEGEND_SIZE = 10
LINEWIDTH = 2.2

# Color-blind-friendly palette
COLORS = {
    "baseline": "#0072B2",   # blue
    "ridge": "#E69F00",      # orange
    "tree": "#009E73",       # green
    "nn": "#CC79A7",         # purple
    "drawdown": "#D55E00",   # red-orange
    "turnover": "#56B4E9",   # light blue
}


def _prepare_series(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    return s


def _apply_common_style(ax, title: str, ylabel: str):
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel("Date", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)

    # Adaptive date formatting based on visible date span
    xmin, xmax = ax.get_xlim()
    x0 = mdates.num2date(xmin)
    x1 = mdates.num2date(xmax)

    span_days = (x1 - x0).days

    if span_days <= 400:
        # About one year or less -> show months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    elif span_days <= 1100:
        # 1–3 years -> show every 3 months
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    else:
        # Long history -> show years
        ax.xaxis.set_major_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(fig, save_path: str | None):
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight")


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str = COLORS["baseline"],
    linestyle: str = "-"
):
    s = _prepare_series(equity_curve)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(s.index, s.values, label=label, color=color, linestyle=linestyle, linewidth=LINEWIDTH)

    _apply_common_style(ax, title, "Portfolio Value")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_drawdown(
    equity_curve: pd.Series,
    title: str = "Drawdown",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str = COLORS["drawdown"],
    linestyle: str = "-"
):
    s = _prepare_series(equity_curve)
    drawdown = s / s.cummax() - 1.0

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(drawdown.index, drawdown.values, label=label, color=color, linestyle=linestyle, linewidth=LINEWIDTH)

    _apply_common_style(ax, title, "Drawdown")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_turnover(
    turnover_series: pd.Series,
    title: str = "Turnover",
    save_path: str | None = None,
    label: str = "Strategy",
    color: str = COLORS["turnover"],
    linestyle: str = "-"
):
    s = _prepare_series(turnover_series)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(s.index, s.values, label=label, color=color, linestyle=linestyle, linewidth=LINEWIDTH)

    _apply_common_style(ax, title, "Turnover")
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.show()


def plot_two_series(
    series_a: pd.Series,
    series_b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    ylabel: str,
    save_path: str | None = None,
    color_a: str = COLORS["baseline"],
    color_b: str = COLORS["ridge"],
    linestyle_a: str = "-",
    linestyle_b: str = "--",
):
    a = _prepare_series(series_a)
    b = _prepare_series(series_b)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(a.index, a.values, label=label_a, color=color_a, linestyle=linestyle_a, linewidth=LINEWIDTH)
    ax.plot(b.index, b.values, label=label_b, color=color_b, linestyle=linestyle_b, linewidth=LINEWIDTH)

    _apply_common_style(ax, title, ylabel)
    ax.legend(frameon=False, fontsize=LEGEND_SIZE, loc="best")

    fig.tight_layout()
    save_figure(fig, save_path)
    plt.show()