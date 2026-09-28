"""Phase 1 SPEI-3 sequence and documented severity-label utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "data" / "phase1_final_dataset.csv"
RESULTS = ROOT / "results"

SEQUENCE_LENGTH = 6
FEATURE_COLUMNS = [
    "rainfall_lag1",
    "temp_lag1",
    "soil_moisture_lag1",
    "vhi_lag1",
    "groundwater_depth_lag1",
]
# Age and count are quality metadata in the prepared dataset, not model inputs.
COMPLETENESS_COLUMNS = FEATURE_COLUMNS + [
    "groundwater_age_months", "groundwater_well_count",
]
CLASS_NAMES = [
    "No drought",
    "Light drought",
    "Moderate drought",
    "Severe drought",
    "Extreme drought",
]
N_CLASSES = len(CLASS_NAMES)

# Published five-grade SPEI scale. Boundaries follow the paper's Table 3:
# no drought: SPEI > -0.5; light: -1 < SPEI <= -0.5;
# moderate: -1.5 < SPEI <= -1; severe: -2 < SPEI <= -1.5;
# extreme: SPEI <= -2.
SPEI_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5]
CLASSIFICATION_REFERENCE = (
    "Construction of an Integrated Drought Monitoring Model Based on Deep "
    "Learning Algorithms, Remote Sensing, 2023, 15(3), 667, Table 3; "
    "https://www.mdpi.com/2072-4292/15/3/667"
)


def classify_spei(values) -> np.ndarray:
    """Map continuous SPEI to five documented drought-severity grades."""
    values = np.asarray(values, dtype=float)
    # Bins: (-inf,-2], (-2,-1.5], (-1.5,-1], (-1,-.5], (-.5,+inf)
    return (N_CLASSES - 1 - np.digitize(
        values, bins=SPEI_THRESHOLDS, right=True
    )).astype(np.int64)


def load_final_dataset(path: Path = DATASET_PATH) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["target_date"])
    required = [
        "target_date", "city", *FEATURE_COLUMNS, "groundwater_age_months",
        "groundwater_well_count", "spei_3",
    ]
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"Final dataset missing required columns: {missing}")
    if frame.duplicated(["target_date", "city"]).any():
        raise ValueError("Duplicate city-target-date records in final dataset")
    if frame["spei_3"].isna().any():
        raise ValueError("Missing SPEI-3 target values")
    return frame.sort_values(["city", "target_date"]).reset_index(drop=True)


def build_sequences(frame: pd.DataFrame):
    """Create X[T-6:T-1] and y[T] from the one-month-lagged table.

    Each six-row window has target dates T-5..T; its lagged predictor columns
    therefore correspond exactly to source months T-6..T-1. Rows are retained
    only when all six rows are monthly-consecutive and every feature plus the
    groundwater quality metadata is present throughout the window.
    """
    X_rows, y_cont, y_class, y_previous, cities, dates = [], [], [], [], [], []
    for city, city_frame in frame.groupby("city", sort=True):
        city_frame = city_frame.sort_values("target_date").reset_index(drop=True)
        for end in range(SEQUENCE_LENGTH - 1, len(city_frame)):
            window = city_frame.iloc[end - SEQUENCE_LENGTH + 1:end + 1]
            expected_dates = pd.date_range(
                window.iloc[0]["target_date"], periods=SEQUENCE_LENGTH, freq="MS"
            )
            if list(window["target_date"]) != list(expected_dates):
                continue
            if window[COMPLETENESS_COLUMNS].isna().any().any():
                continue
            target = float(window.iloc[-1]["spei_3"])
            previous_spei = float(window.iloc[-2]["spei_3"])
            X_rows.append(window[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
            y_cont.append(target)
            y_class.append(int(classify_spei([target])[0]))
            y_previous.append(previous_spei)
            cities.append(city)
            dates.append(window.iloc[-1]["target_date"])

    X = np.asarray(X_rows, dtype=np.float32)
    y_cont = np.asarray(y_cont, dtype=np.float32)
    y_class = np.asarray(y_class, dtype=np.int64)
    y_previous = np.asarray(y_previous, dtype=np.float32)
    metadata = pd.DataFrame({
        "city": cities,
        "target_date": pd.to_datetime(dates),
    })
    if X.shape[1:] != (SEQUENCE_LENGTH, len(FEATURE_COLUMNS)):
        raise AssertionError(f"Unexpected sequence tensor shape: {X.shape}")
    return X, y_cont, y_class, y_previous, metadata
