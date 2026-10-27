from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .model_service import FORECAST_INFO


@dataclass
class ForecastResult:
    historical: pd.Series
    forecast: pd.Series
    monthly_table: pd.DataFrame
    quarterly_table: pd.DataFrame
    annual_total: float
    average_monthly: float
    peak_month: str
    peak_value: float
    model_name: str
    notes: list[str]


# =============================================================================
# BUILD MONTHLY INCIDENT SERIES
# =============================================================================

def build_monthly_series(
    df: pd.DataFrame,
    date_format: str = "dayfirst",
) -> pd.Series:

    # -------------------------------------------------------------------------
    # Check required column
    # -------------------------------------------------------------------------

    if "Open_Time" not in df.columns:
        raise ValueError(
            "CSV must contain an 'Open_Time' column."
        )

    raw_dates = df["Open_Time"].copy()

    # -------------------------------------------------------------------------
    # Parse Open_Time
    #
    # Your Excel/CSV example:
    #
    # 05-02-2012 13:32:00
    #
    # This means:
    # 05 = day
    # 02 = month
    # 2012 = year
    #
    # So the expected format is:
    # DD-MM-YYYY HH:MM:SS
    # -------------------------------------------------------------------------

    if date_format == "monthfirst":

        # MM-DD-YYYY HH:MM:SS
        parsed = pd.to_datetime(
            raw_dates,
            format="%m-%d-%Y %H:%M:%S",
            errors="coerce",
        )

        # Fallback for rows that don't exactly match the format
        failed = parsed.isna()

        if failed.any():
            parsed.loc[failed] = pd.to_datetime(
                raw_dates.loc[failed],
                errors="coerce",
                dayfirst=False,
            )

    else:

        # DD-MM-YYYY HH:MM:SS
        parsed = pd.to_datetime(
            raw_dates,
            format="%d-%m-%Y %H:%M:%S",
            errors="coerce",
        )

        # Fallback for rows that may have a slightly different
        # representation, such as:
        # 05-02-2012
        # 05/02/2012
        # 05-02-2012 13:32
        failed = parsed.isna()

        if failed.any():
            parsed.loc[failed] = pd.to_datetime(
                raw_dates.loc[failed],
                errors="coerce",
                dayfirst=True,
            )

    # -------------------------------------------------------------------------
    # Validate dates
    # -------------------------------------------------------------------------

    valid = parsed.notna()

    if valid.sum() == 0:
        raise ValueError(
            "No valid Open_Time values could be parsed. "
            "Expected format: DD-MM-YYYY HH:MM:SS"
        )

    # -------------------------------------------------------------------------
    # Count rows that could not be parsed
    # -------------------------------------------------------------------------

    invalid_count = int((~valid).sum())

    # -------------------------------------------------------------------------
    # Convert timestamps to monthly periods
    #
    # Example:
    #
    # 05-02-2012 13:32:00
    #         ↓
    # Period('2012-02', 'M')
    #         ↓
    # 2012-02-01
    #
    # IMPORTANT:
    # pandas 3.x expects how='start' here.
    # Do NOT use:
    #     .dt.to_timestamp("MS")
    # -------------------------------------------------------------------------

    month = (
        parsed.loc[valid]
        .dt.to_period("M")
        .dt.to_timestamp(how="start")
    )

    # -------------------------------------------------------------------------
    # Count incidents per month
    # -------------------------------------------------------------------------

    monthly = (
        month
        .value_counts()
        .sort_index()
        .astype(float)
    )

    if monthly.empty:
        raise ValueError(
            "No monthly incident data could be created from Open_Time."
        )

    # -------------------------------------------------------------------------
    # Create a continuous monthly index
    #
    # If a month has no records, fill it with 0.
    # -------------------------------------------------------------------------

    full_index = pd.date_range(
        start=monthly.index.min(),
        end=monthly.index.max(),
        freq="MS",
    )

    monthly = monthly.reindex(
        full_index,
        fill_value=0.0,
    )

    monthly.index.name = "Open_Time"
    monthly.name = "Incident_Count"

    return monthly


# =============================================================================
# LAG CONFIGURATION
# =============================================================================

def lag_config(
    history: pd.Series,
) -> tuple[list[int], list[int]]:

    # For longer histories use 12-month information as well
    if len(history) >= 36:
        return [1, 2, 3, 6, 12], [3, 6, 12]

    # This matches the configuration used for your
    # 26-month ABC Tech training history.
    return [1, 2, 3, 6], [3, 6]


# =============================================================================
# CREATE FEATURES FOR EXTRA TREES FORECASTING
# =============================================================================

def create_future_features(
    history: pd.Series,
    timestamp: pd.Timestamp,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:

    # -------------------------------------------------------------------------
    # Recover the lag / rolling configuration from the saved model whenever
    # possible.
    # -------------------------------------------------------------------------

    if feature_columns:

        lags = sorted(
            int(col.split("_")[1])
            for col in feature_columns
            if col.startswith("lag_")
        )

        rolling_windows = sorted(
            int(col.rsplit("_", 1)[1])
            for col in feature_columns
            if col.startswith("rolling_mean_")
        )

    else:

        lags, rolling_windows = lag_config(history)

    # -------------------------------------------------------------------------
    # Validate history length
    # -------------------------------------------------------------------------

    if len(history) < max(lags):

        raise ValueError(
            f"Extra Trees forecasting needs at least "
            f"{max(lags)} months of history; "
            f"the uploaded file produced {len(history)} months."
        )

    # -------------------------------------------------------------------------
    # Create one feature row
    # -------------------------------------------------------------------------

    row: dict[str, float] = {}

    # Lag features
    for lag in lags:

        row[f"lag_{lag}"] = float(
            history.iloc[-lag]
        )

    # Rolling features
    for window in rolling_windows:

        values = history.tail(window)

        row[f"rolling_mean_{window}"] = float(
            values.mean()
        )

        if len(values) > 1:

            row[f"rolling_std_{window}"] = float(
                values.std(ddof=1)
            )

        else:

            row[f"rolling_std_{window}"] = 0.0

    # -------------------------------------------------------------------------
    # Calendar features
    # -------------------------------------------------------------------------

    month = timestamp.month

    row["month_sin"] = float(
        np.sin(
            2 * np.pi * month / 12
        )
    )

    row["month_cos"] = float(
        np.cos(
            2 * np.pi * month / 12
        )
    )

    row["quarter"] = float(
        timestamp.quarter
    )

    row["trend"] = float(
        len(history)
    )

    return pd.DataFrame(
        [row]
    )


# =============================================================================
# DRIFT FORECAST
# =============================================================================

def forecast_drift(
    bundle: dict[str, Any],
    history: pd.Series,
    horizon: int,
) -> pd.Series:

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than 0."
        )

    # The saved bundle contains the historical slope.
    slope = float(
        bundle["slope"]
    )

    # Use the uploaded data's latest observed value.
    last_value = float(
        history.iloc[-1]
    )

    # Generate future monthly dates.
    forecast_index = pd.date_range(
        start=(
            history.index[-1]
            + pd.offsets.MonthBegin(1)
        ),
        periods=horizon,
        freq="MS",
    )

    # Forecast:
    # last observed level + slope * horizon
    values = [
        max(
            0.0,
            last_value + i * slope
        )
        for i in range(1, horizon + 1)
    ]

    return pd.Series(
        values,
        index=forecast_index,
        name="Forecast",
    )


# =============================================================================
# EXTRA TREES FORECAST
# =============================================================================

def forecast_extra_trees(
    bundle: dict[str, Any],
    history: pd.Series,
    horizon: int,
) -> pd.Series:

    if horizon <= 0:
        raise ValueError(
            "Forecast horizon must be greater than 0."
        )

    model = bundle["model"]

    feature_columns = bundle.get(
        "feature_columns"
    )

    # Copy history so recursive forecasting does not modify
    # the original historical series.
    working = (
        history
        .copy()
        .astype(float)
    )

    forecasts: list[float] = []
    forecast_dates: list[pd.Timestamp] = []

    # First future month
    next_date = (
        working.index[-1]
        + pd.offsets.MonthBegin(1)
    )

    # -------------------------------------------------------------------------
    # Recursive forecasting
    # -------------------------------------------------------------------------

    for step in range(horizon):

        timestamp = (
            next_date
            + pd.offsets.MonthBegin(step)
        )

        # Create feature row using current history
        X = create_future_features(
            working,
            timestamp,
            feature_columns=feature_columns,
        )

        # Ensure exact feature order expected by the saved model
        if feature_columns:

            X = X.reindex(
                columns=feature_columns
            )

        # Make prediction
        prediction = float(
            model.predict(X)[0]
        )

        # Incident counts cannot be negative
        prediction = max(
            0.0,
            prediction
        )

        forecasts.append(
            prediction
        )

        forecast_dates.append(
            timestamp
        )

        # Add prediction to working history
        # so it can be used for the next recursive prediction.
        working.loc[
            timestamp
        ] = prediction

    return pd.Series(
        forecasts,
        index=forecast_dates,
        name="Forecast",
    )


# =============================================================================
# BUILD FINAL FORECAST RESULT
# =============================================================================

def make_result(
    history: pd.Series,
    forecast: pd.Series,
    model_key: str,
    notes: list[str],
) -> ForecastResult:

    # -------------------------------------------------------------------------
    # Monthly table
    # -------------------------------------------------------------------------

    monthly_table = pd.DataFrame(
        {
            "Month": forecast.index.strftime(
                "%b %Y"
            ),
            "Forecast Incidents": (
                forecast
                .round(0)
                .astype(int)
                .values
            ),
        }
    )

    # -------------------------------------------------------------------------
    # Quarterly aggregation
    # -------------------------------------------------------------------------

    quarterly = forecast.to_frame(
        "Forecast"
    ).copy()

    quarterly["Quarter"] = (
        quarterly.index
        .to_period("Q")
        .astype(str)
    )

    quarterly_table = (
        quarterly
        .groupby(
            "Quarter",
            sort=True
        )["Forecast"]
        .sum()
        .round(0)
        .astype(int)
        .reset_index()
    )

    quarterly_table.columns = [
        "Quarter",
        "Forecast Incidents",
    ]

    # -------------------------------------------------------------------------
    # Peak month
    # -------------------------------------------------------------------------

    peak_idx = forecast.idxmax()

    # -------------------------------------------------------------------------
    # Create final object
    # -------------------------------------------------------------------------

    return ForecastResult(

        historical=history,

        forecast=forecast,

        monthly_table=monthly_table,

        quarterly_table=quarterly_table,

        annual_total=float(
            forecast.sum()
        ),

        average_monthly=float(
            forecast.mean()
        ),

        peak_month=peak_idx.strftime(
            "%B %Y"
        ),

        peak_value=float(
            forecast.max()
        ),

        model_name=FORECAST_INFO[
            model_key
        ]["name"],

        notes=notes,
    )