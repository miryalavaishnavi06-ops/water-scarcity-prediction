"""Build leakage-safe 24-month Phase-I sequences without changing the 6m path.

The active dataset stores predictors on target-month rows as *_lag1 fields:
the row for month M contains predictor observations from M-1. Therefore the
24 rows ending at target month T (row dates T-23 through T) encode exactly
the source months T-24 through T-1. Target-month predictors are excluded.

This module only constructs sequences. It does not split, scale, train, or
write/overwrite project datasets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from phase1_data import (
    CLASS_NAMES,
    COMPLETENESS_COLUMNS,
    FEATURE_COLUMNS,
    N_CLASSES,
    classify_spei,
)


SEQUENCE_LENGTH_24M = 24
REQUIRED_COLUMNS_24M = ["target_date", "city", *COMPLETENESS_COLUMNS, "spei_3"]


def build_sequences_24m(frame: pd.DataFrame):
    """Return 24-month X, current-month SPEI labels, and endpoint metadata.

    For target T, X contains exactly predictor source months T-24 through
    T-1, in chronological order. Since the source table is lagged by one
    month, this is assembled from its 24 consecutive rows dated T-23 through
    T. The final row supplies the target SPEI value only; none of its SPEI
    values (including prior SPEI) are included in X.

    Rows with a calendar gap or any missing feature/groundwater quality field
    in the full 24-month input window are skipped. Groundwater age and well
    count are completeness checks only and are not model input features.

    Returns:
        X: float32 array, shape (n_sequences, 24, 5)
        y_spei: float32 array of target-month SPEI-3 values
        y_class: int64 array using the existing five fixed SPEI thresholds
        metadata: DataFrame with city, target_date, input_start, input_end
    """
    missing = sorted(set(REQUIRED_COLUMNS_24M) - set(frame.columns))
    if missing:
        raise ValueError(f"24-month input frame is missing columns: {missing}")

    data = frame[REQUIRED_COLUMNS_24M].copy()
    data["target_date"] = (
        pd.to_datetime(data["target_date"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    data["city"] = data["city"].astype(str).str.strip()
    if data["target_date"].isna().any():
        raise ValueError("Invalid target_date in 24-month sequence input")
    if data["city"].isna().any() or data["city"].eq("").any():
        raise ValueError("Missing city in 24-month sequence input")
    if data.duplicated(["target_date", "city"]).any():
        raise ValueError("Duplicate city-month rows in 24-month sequence input")

    for column in [*COMPLETENESS_COLUMNS, "spei_3"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    X_rows: list[np.ndarray] = []
    y_values: list[float] = []
    cities: list[str] = []
    target_dates: list[pd.Timestamp] = []
    input_starts: list[pd.Timestamp] = []
    input_ends: list[pd.Timestamp] = []

    for city, city_frame in data.groupby("city", sort=True):
        city_frame = city_frame.sort_values("target_date").reset_index(drop=True)
        for end in range(SEQUENCE_LENGTH_24M - 1, len(city_frame)):
            window = city_frame.iloc[end - SEQUENCE_LENGTH_24M + 1 : end + 1]
            target_date = window.iloc[-1]["target_date"]

            # These are target-row dates T-23..T. Their lagged predictor
            # source dates must be exactly T-24..T-1.
            expected_row_dates = pd.date_range(
                end=target_date, periods=SEQUENCE_LENGTH_24M, freq="MS"
            )
            if not window["target_date"].tolist() == list(expected_row_dates):
                continue
            if window[COMPLETENESS_COLUMNS].isna().any().any():
                continue

            source_dates = (
                window["target_date"] - pd.offsets.MonthBegin(1)
            ).tolist()
            expected_source_dates = pd.date_range(
                end=target_date - pd.offsets.MonthBegin(1),
                periods=SEQUENCE_LENGTH_24M,
                freq="MS",
            )
            if source_dates != list(expected_source_dates):
                raise AssertionError(
                    f"Predictor months do not equal T-24..T-1 for {city}, {target_date:%Y-%m}"
                )
            if expected_source_dates[-1] >= target_date:
                raise AssertionError("A target-month predictor entered the input sequence")

            target_value = float(window.iloc[-1]["spei_3"])
            if not np.isfinite(target_value):
                continue

            X_rows.append(window[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            y_values.append(target_value)
            cities.append(city)
            target_dates.append(target_date)
            input_starts.append(expected_source_dates[0])
            input_ends.append(expected_source_dates[-1])

    if X_rows:
        X = np.stack(X_rows).astype(np.float32, copy=False)
    else:
        X = np.empty(
            (0, SEQUENCE_LENGTH_24M, len(FEATURE_COLUMNS)), dtype=np.float32
        )
    y_spei = np.asarray(y_values, dtype=np.float32)
    y_class = classify_spei(y_spei).astype(np.int64, copy=False)
    metadata = pd.DataFrame(
        {
            "city": cities,
            "target_date": pd.to_datetime(target_dates),
            "input_start": pd.to_datetime(input_starts),
            "input_end": pd.to_datetime(input_ends),
        }
    )

    if X.shape[1:] != (SEQUENCE_LENGTH_24M, len(FEATURE_COLUMNS)):
        raise AssertionError(f"Unexpected 24-month sequence shape: {X.shape}")
    if len(y_spei) != len(y_class) or len(y_class) != len(metadata):
        raise AssertionError("24-month sequence output lengths do not agree")
    if len(metadata) and not (
        (metadata["input_start"] == metadata["target_date"] - pd.DateOffset(months=24)).all()
        and (metadata["input_end"] == metadata["target_date"] - pd.DateOffset(months=1)).all()
    ):
        raise AssertionError("24-month metadata failed the T-24..T-1 interval check")

    return X, y_spei, y_class, metadata


__all__ = [
    "SEQUENCE_LENGTH_24M",
    "FEATURE_COLUMNS",
    "CLASS_NAMES",
    "N_CLASSES",
    "build_sequences_24m",
]
