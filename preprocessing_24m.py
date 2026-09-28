"""Canonical reproducible scaler preparation/loading for Phase-I 24m inference.

The saved 24-month model's training procedure fitted StandardScaler on the
flattened 24 time steps from the optimizer-fit sequences only. This module
reconstructs that exact chronological fit partition and provides the same
reshape/transform behavior for inference.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from data_utils_24m import build_sequences_24m
from phase1_data import DATASET_PATH, FEATURE_COLUMNS, RESULTS


ROOT = Path(__file__).resolve().parents[1]
SCALER_ARTIFACT_PATH = RESULTS / "phase1_scaler_24m.pkl"
MODEL_FEATURE_NAMES = [
    "rainfall_mm", "temp_c", "soil_moisture", "vhi", "groundwater_depth_m"
]
ARTIFACT_SCHEMA_VERSION = 1


def chronological_24m_indices(metadata: pd.DataFrame) -> dict[str, np.ndarray]:
    """Recreate the fixed training split used by train_phase1_24m.py.

    All eligible 2015 endpoints are test. From pre-2015 endpoint dates, the
    latest rounded 15% of unique dates are validation; earlier dates are fit.
    """
    if "target_date" not in metadata:
        raise ValueError("Endpoint metadata is missing target_date")
    dates = pd.to_datetime(metadata["target_date"], errors="coerce")
    if dates.isna().any():
        raise ValueError("Invalid target date in sequence endpoint metadata")
    test_idx = np.flatnonzero(dates.dt.year.to_numpy() == 2015)
    development_idx = np.flatnonzero(dates.dt.year.to_numpy() < 2015)
    development_dates = np.array(sorted(pd.unique(dates.iloc[development_idx])))
    validation_date_count = max(1, int(round(0.15 * len(development_dates))))
    validation_dates = development_dates[-validation_date_count:]
    fit_dates = development_dates[:-validation_date_count]
    fit_idx = np.flatnonzero(dates.isin(fit_dates).to_numpy())
    validation_idx = np.flatnonzero(dates.isin(validation_dates).to_numpy())
    result = {"fit": fit_idx, "validation": validation_idx, "test": test_idx}
    if tuple(map(len, (fit_idx, validation_idx, test_idx))) != (30, 6, 36):
        raise ValueError("Current data no longer reproduces audited split sizes 30/6/36")
    if not (dates.iloc[fit_idx].max() < dates.iloc[validation_idx].min()
            and dates.iloc[validation_idx].max() < dates.iloc[test_idx].min()):
        raise ValueError("Reconstructed scaler fit partition is not chronological")
    return result


def fit_phase1_scaler_24m(
    X_raw: np.ndarray, metadata: pd.DataFrame
) -> tuple[StandardScaler, dict]:
    """Fit the exact baseline scaler on the 30 optimizer-fit sequences only."""
    x = np.asarray(X_raw, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (24, len(FEATURE_COLUMNS)):
        raise ValueError(f"Expected sequence tensor (n,24,5); got {x.shape}")
    if len(x) != len(metadata):
        raise ValueError("Sequence tensor and endpoint metadata lengths differ")

    split = chronological_24m_indices(metadata)
    fit_indices = split["fit"]
    # Same operation used in src/train_phase1_24m.py.
    flattened_fit_steps = x[fit_indices].reshape(-1, len(FEATURE_COLUMNS))
    scaler = StandardScaler()
    scaler.fit(flattened_fit_steps)

    fit_meta = metadata.iloc[fit_indices]
    records = []
    for row in fit_meta.itertuples(index=False):
        records.append({
            "city": str(row.city),
            "target_date": pd.Timestamp(row.target_date).strftime("%Y-%m-%d"),
            "input_start": pd.Timestamp(row.input_start).strftime("%Y-%m-%d"),
            "input_end": pd.Timestamp(row.input_end).strftime("%Y-%m-%d"),
        })
    provenance = {
        "sequence_count": int(len(fit_indices)),
        "flattened_timestep_rows": int(flattened_fit_steps.shape[0]),
        "feature_columns_internal_order": list(FEATURE_COLUMNS),
        "feature_names_model_order": list(MODEL_FEATURE_NAMES),
        "target_date_start": pd.Timestamp(fit_meta["target_date"].min()).strftime("%Y-%m-%d"),
        "target_date_end": pd.Timestamp(fit_meta["target_date"].max()).strftime("%Y-%m-%d"),
        "unique_target_dates": sorted(pd.to_datetime(fit_meta["target_date"]).dt.strftime("%Y-%m").unique().tolist()),
        "city_counts": {str(k): int(v) for k, v in fit_meta["city"].value_counts().sort_index().items()},
        "fit_sequences": records,
        "validation_or_test_used_to_fit": False,
    }
    return scaler, provenance


def transform_24m_sequences(scaler: StandardScaler, sequences: np.ndarray) -> np.ndarray:
    """Apply the training-time flattened StandardScaler transform to 24m input."""
    x = np.asarray(sequences, dtype=np.float32)
    if x.ndim not in (2, 3) or x.shape[-2:] != (24, len(FEATURE_COLUMNS)):
        raise ValueError(f"Expected (24,5) or (n,24,5) sequence data; got {x.shape}")
    shape = x.shape
    transformed = scaler.transform(x.reshape(-1, len(FEATURE_COLUMNS)))
    return transformed.reshape(shape).astype(np.float32)


def dataset_sha256(dataset_path: Path = DATASET_PATH) -> str:
    digest = hashlib.sha256()
    with Path(dataset_path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_scaler_artifact(
    dataset_path: Path = DATASET_PATH,
    artifact_path: Path = SCALER_ARTIFACT_PATH,
) -> dict:
    """Recreate, document, and save the original model's fit-only scaler."""
    dataset_path, artifact_path = Path(dataset_path), Path(artifact_path)
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite scaler artifact: {artifact_path}")
    frame = pd.read_csv(dataset_path, parse_dates=["target_date"])
    X_raw, _, _, metadata = build_sequences_24m(frame)
    if X_raw.shape != (72, 24, 5):
        raise ValueError(f"Expected audited 72x24x5 input sequences, found {X_raw.shape}")
    scaler, provenance = fit_phase1_scaler_24m(X_raw, metadata)
    artifact = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "scaler_type": "sklearn.preprocessing.StandardScaler",
        "scaler": scaler,
        "dataset_path_at_creation": str(dataset_path.resolve()),
        "dataset_sha256": dataset_sha256(dataset_path),
        "sequence_length": 24,
        "feature_count": 5,
        "fit_method": "StandardScaler.fit(X_raw[optimizer_fit_indices].reshape(-1, 5))",
        "fit_rows_are_overlapping_sequence_timesteps": True,
        "provenance": provenance,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("xb") as stream:
        pickle.dump(artifact, stream, protocol=pickle.HIGHEST_PROTOCOL)
    return artifact


def load_phase1_scaler_24m(
    artifact_path: Path = SCALER_ARTIFACT_PATH,
    dataset_path: Path | None = DATASET_PATH,
) -> tuple[StandardScaler, dict]:
    """Load and validate the saved scaler and its dataset/feature provenance."""
    with Path(artifact_path).open("rb") as stream:
        artifact = pickle.load(stream)
    if artifact.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("Unsupported 24-month scaler artifact schema")
    provenance = artifact.get("provenance", {})
    if provenance.get("feature_columns_internal_order") != list(FEATURE_COLUMNS):
        raise ValueError("Scaler feature order does not match the 24-month sequence utility")
    if provenance.get("feature_names_model_order") != MODEL_FEATURE_NAMES:
        raise ValueError("Scaler model feature names do not match the Phase-I model order")
    scaler = artifact.get("scaler")
    if not isinstance(scaler, StandardScaler) or int(scaler.n_features_in_) != len(FEATURE_COLUMNS):
        raise ValueError("Artifact does not contain the expected fitted five-feature StandardScaler")
    if provenance.get("sequence_count") != 30 or provenance.get("flattened_timestep_rows") != 720:
        raise ValueError("Scaler provenance does not match the 30-sequence/720-row fit partition")
    if dataset_path is not None and dataset_sha256(Path(dataset_path)) != artifact.get("dataset_sha256"):
        raise ValueError("Dataset checksum differs from the one used to create this scaler artifact")
    return scaler, artifact


def main() -> None:
    artifact = create_scaler_artifact()
    provenance = artifact["provenance"]
    print(f"Saved fit-only scaler: {SCALER_ARTIFACT_PATH}")
    print(f"Optimizer-fit sequences: {provenance['sequence_count']}")
    print(f"Flattened sequence timestep rows: {provenance['flattened_timestep_rows']}")
    print(f"Endpoint targets: {provenance['target_date_start']} through {provenance['target_date_end']}")
    print(f"City counts: {provenance['city_counts']}")


if __name__ == "__main__":
    main()
