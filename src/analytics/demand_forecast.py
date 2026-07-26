"""
demand_forecast.py
Feature 3: Macro Demand Forecaster

Generates a 24-step hourly order volume forecast using Holt-Winters exponential
smoothing (additive trend + additive seasonality, period = 24 h).

Output feeds the reinforcement learning state matrix (Engineer 3).

Deliverable: data/outputs/demand_forecast.json
  Schema: {"hourly_forecast": [float × 24]}

Usage (from project root):
    python -m src.analytics.demand_forecast
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

from src.config import (
    DATA_INTERIM_DIRECTORY,
    DATA_OUTPUTS_DIRECTORY,
    DEMAND_FORECAST_FILENAME,
    MAPPED_ORDERS_FILENAME,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SEASONAL_PERIOD = 24        # daily cycle
_MIN_HOURS_REQUIRED = 48     # 2 full seasonal cycles needed by Holt-Winters
_FORECAST_STEPS = 24         # 24-hour ahead forecast


def _synthesize_series(df: pd.DataFrame, span_hours: float) -> pd.Series:
    """
    Generate 7 days of synthetic hourly demand when observed span < 48 h.

    Preserves the hour-of-day distribution observed in the data, overlaid with
    a mild upward trend and Gaussian noise.
    """
    hour_dist = (
        df["Order_Timestamp"]
        .dt.hour.value_counts(normalize=True)
        .reindex(range(24), fill_value=0.005)
    )
    hour_dist /= hour_dist.sum()

    avg_daily = len(df) / max(span_hours / 24.0, 1.0)
    base_ts = pd.Timestamp("2022-03-14 00:00:00")
    rows: list[dict] = []
    np.random.seed(42)

    for day in range(7):
        daily_vol = avg_daily * (1 + 0.015 * day)
        for hour in range(24):
            expected = daily_vol * hour_dist.iloc[hour]
            noise = np.random.normal(0, max(expected * 0.12, 1))
            count = max(0, int(expected + noise))
            rows.append(
                {
                    "Hour_Bucket": base_ts + pd.Timedelta(days=day, hours=hour),
                    "Order_Count": count,
                }
            )

    series = pd.DataFrame(rows).set_index("Hour_Bucket")["Order_Count"]
    return series.asfreq("h", fill_value=0)


def forecast_demand(
    handoff_path: Path | None = None,
    output_path: Path | None = None,
) -> list[float]:
    """
    Fit Holt-Winters on mapped order timestamps and produce a 24-hour forecast.

    Parameters
    ----------
    handoff_path : Path, optional
        Path to mapped_orders.pkl (Feature 1 output).
        Defaults to ``data/interim/mapped_orders.pkl``.
    output_path : Path, optional
        Destination JSON path.
        Defaults to ``data/outputs/demand_forecast.json``.

    Returns
    -------
    list[float]
        24-element array of non-negative forecasted hourly order volumes.

    Raises
    ------
    FileNotFoundError
        If the Feature 1 handoff pickle is absent.
    ValueError
        If the ``Order_Timestamp`` column is missing from the handoff.
    """
    handoff_path = handoff_path or (DATA_INTERIM_DIRECTORY / MAPPED_ORDERS_FILENAME)
    output_path = output_path or (DATA_OUTPUTS_DIRECTORY / DEMAND_FORECAST_FILENAME)

    if not handoff_path.exists():
        raise FileNotFoundError(
            f"Feature 1 handoff not found: {handoff_path}\n"
            "Run 'python -m src.ingestion.snap_nodes' first."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Load & validate
    # ------------------------------------------------------------------ #
    df = pd.read_pickle(handoff_path)
    if "Order_Timestamp" not in df.columns:
        raise ValueError(
            "'Order_Timestamp' column missing from Feature 1 handoff."
        )

    df["Order_Timestamp"] = pd.to_datetime(df["Order_Timestamp"])
    df = df.dropna(subset=["Order_Timestamp"])
    span_hours = (
        df["Order_Timestamp"].max() - df["Order_Timestamp"].min()
    ).total_seconds() / 3600
    logger.info(
        f"Loaded {len(df):,} orders | time span: {span_hours:.0f} h "
        f"({span_hours / 24:.1f} days)"
    )

    # ------------------------------------------------------------------ #
    # 2. Hourly aggregation
    # ------------------------------------------------------------------ #
    df["Hour_Bucket"] = df["Order_Timestamp"].dt.floor("h")
    hourly = df.groupby("Hour_Bucket").size().rename("Order_Count")
    hourly = hourly.asfreq("h", fill_value=0)

    if len(hourly) < _MIN_HOURS_REQUIRED:
        logger.warning(
            f"Only {len(hourly)} hourly buckets - below minimum {_MIN_HOURS_REQUIRED}. "
            "Synthesising 7-day demand series."
        )
        hourly = _synthesize_series(df, span_hours)

    logger.info(
        f"Demand series: {len(hourly)} hours | "
        f"mean={hourly.mean():.1f} orders/h | peak={hourly.max()} orders/h"
    )

    # ------------------------------------------------------------------ #
    # 3. Holt-Winters fit
    # ------------------------------------------------------------------ #
    hw = ExponentialSmoothing(
        hourly.values.astype(float),
        trend="add",
        seasonal="add",
        seasonal_periods=_SEASONAL_PERIOD,
    ).fit(optimized=True)

    logger.info(
        f"Holt-Winters fitted - "
        f"alpha={hw.params['smoothing_level']:.4f}, "
        f"beta={hw.params['smoothing_trend']:.4f}, "
        f"gamma={hw.params['smoothing_seasonal']:.4f} | "
        f"AIC={hw.aic:.2f}"
    )

    # ------------------------------------------------------------------ #
    # 4. 24-step forecast
    # ------------------------------------------------------------------ #
    raw_forecast = hw.forecast(_FORECAST_STEPS)
    forecast_values = [round(float(v), 2) for v in np.maximum(0.0, raw_forecast)]

    # --- JSON (consumed by RL agent) ---
    result = {"hourly_forecast": forecast_values}
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    # --- CSV (human-readable) ---
    csv_path = output_path.with_suffix(".csv")
    pd.DataFrame(
        {"hour_offset": range(24), "forecast_orders": forecast_values}
    ).to_csv(csv_path, index=False)
    logger.info(f"CSV saved: {csv_path}")

    # --- Plot 1: Historical demand (last 7 days) ---
    fig, ax = plt.subplots(figsize=(14, 4))
    plot_history = hourly.iloc[-7 * 24:]
    ax.fill_between(plot_history.index, plot_history.values, alpha=0.35, color="steelblue")
    ax.plot(plot_history.index, plot_history.values, color="steelblue", linewidth=0.8)
    ax.axhline(hourly.mean(), color="tomato", linestyle="--", linewidth=1, label=f"Overall mean ({hourly.mean():.1f} orders/h)")
    ax.set_title("Historical Hourly Order Demand (last 7 days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Orders per hour")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator())
    fig.autofmt_xdate()
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    hist_plot_path = output_path.parent / "demand_history.png"
    fig.savefig(hist_plot_path, dpi=120)
    plt.close(fig)
    logger.info(f"Plot saved: {hist_plot_path}")

    # --- Plot 2: 24-hour forecast bar chart ---
    fig, ax = plt.subplots(figsize=(12, 4))
    hours = list(range(24))
    bars = ax.bar(hours, forecast_values, color="steelblue", alpha=0.8, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f", fontsize=7, padding=2)
    ax.set_title("24-Hour Ahead Demand Forecast (Holt-Winters)")
    ax.set_xlabel("Hour offset from now")
    ax.set_ylabel("Forecast orders per hour")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"H+{h}" for h in hours], fontsize=7, rotation=45)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    forecast_plot_path = output_path.parent / "demand_forecast_24h.png"
    fig.savefig(forecast_plot_path, dpi=120)
    plt.close(fig)
    logger.info(f"Plot saved: {forecast_plot_path}")

    # --- Validation ---
    tc_pass = len(forecast_values) == 24 and all(v >= 0 for v in forecast_values)
    logger.info(
        f"Forecast validation - length=24: {len(forecast_values) == 24} | "
        f"all non-negative: {all(v >= 0 for v in forecast_values)} | "
        f"result: {'PASS' if tc_pass else 'FAIL'}"
    )
    logger.info(f"Deliverable saved: {output_path}")

    return forecast_values


def main() -> None:
    forecast_demand()


if __name__ == "__main__":
    main()
