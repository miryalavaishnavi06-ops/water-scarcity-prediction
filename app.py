"""Phase 1 frozen 24-month Attention-LSTM inference UI."""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Register the frozen model's custom layer and load the exact Phase 1 scaler artifact.
from model_utils import SumOverTime  # noqa: F401
from preprocessing_24m import load_phase1_scaler_24m

CLASS_NAMES = ["No drought", "Light drought", "Moderate drought", "Severe drought", "Extreme drought"]
FEATURES = ["rainfall_mm", "temp_c", "soil_moisture", "vhi", "groundwater_depth_m"]
REQUIRED_COLUMNS = ["date", *FEATURES]
MODEL_PATH = ROOT / "models" / "phase1_attention_lstm_24m.keras"
SCALER_PATH = ROOT / "models" / "phase1_scaler_24m.pkl"
SAMPLE_CSV_PATH = ROOT / "data" / "DEMO_SYNTHETIC_phase1_24m_input.csv"

st.set_page_config(page_title="Phase 1 Drought Prediction", page_icon="water", layout="wide")
st.markdown("""
<style>
.block-container {padding-top:1.5rem;padding-bottom:2rem;max-width:1120px;}
h1,h2,h3 {color:#173746;}
.hero {padding:1.8rem 2rem;border-radius:18px;color:#fff;
       background:linear-gradient(115deg,#102d40 0%,#155b65 65%,#317a70 100%);}
.hero h1 {font-size:2.15rem;line-height:1.15;margin:.25rem 0 .65rem;color:#fff;}
.hero p {font-size:1rem;max-width:850px;margin:0;color:#e0efed;}
.result-card {padding:1.25rem 1.5rem;border-radius:16px;border:1px solid #bfd8d0;
              background:linear-gradient(120deg,#eff8f4,#e4f2ee);color:#173d4b;}
.result-label {font-size:.8rem;text-transform:uppercase;letter-spacing:.1em;color:#536c72;}
.result-value {font-size:2rem;font-weight:750;margin-top:.3rem;}
.result-month {font-size:1.05rem;font-weight:600;margin-top:.25rem;color:#265b62;}
</style>
""", unsafe_allow_html=True)


def validate_input(df: pd.DataFrame) -> pd.DataFrame:
    """Validate one chronological 24-month input window without reordering it."""
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Upload a CSV table with the required columns.")
    if len(df) != 24:
        raise ValueError(f"Expected exactly 24 monthly rows; found {len(df)}.")
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing) + ".")
    extra = [column for column in df.columns if column not in REQUIRED_COLUMNS]
    if extra:
        raise ValueError("Unexpected columns: " + ", ".join(map(str, extra)) +
                         ". Use only date and the five Phase 1 predictors; do not include target/SPEI columns.")

    out = df[REQUIRED_COLUMNS].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        bad_rows = (np.flatnonzero(out["date"].isna().to_numpy()) + 2).tolist()
        raise ValueError(f"Invalid or missing monthly date in CSV row(s): {bad_rows}.")
    if not out["date"].dt.day.eq(1).all():
        raise ValueError("Dates must be month-start dates in YYYY-MM-01 format.")

    months = out["date"].dt.to_period("M")
    if months.duplicated().any():
        duplicated = months[months.duplicated(keep=False)].astype(str).unique().tolist()
        raise ValueError("Duplicate months found: " + ", ".join(duplicated) + ".")
    if not months.is_monotonic_increasing:
        raise ValueError("Rows must already be in chronological order from oldest to newest.")
    expected = pd.period_range(months.iloc[0], periods=24, freq="M")
    if not months.reset_index(drop=True).equals(pd.Series(expected)):
        break_at = next((i for i in range(1, 24)
                         if months.iloc[i] != months.iloc[i - 1] + 1), None)
        if break_at is not None:
            previous_month = months.iloc[break_at - 1]
            expected_next = previous_month + 1
            actual_next = months.iloc[break_at]
            detail = f"The sequence jumps from {previous_month} to {actual_next}; expected {expected_next}."
        else:
            detail = "The sequence does not span exactly 24 consecutive months."
        raise ValueError("Months must form a continuous 24-month sequence with no gaps. " + detail)

    for column in FEATURES:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if out[column].isna().any():
            raise ValueError(f"Column '{column}' has missing or non-numeric values.")
        if not np.isfinite(out[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{column}' contains non-finite values.")
    return out.reset_index(drop=True)


@st.cache_resource(show_spinner="Loading the frozen Phase 1 model and scaler...")
def load_phase1_assets():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Frozen Phase 1 model not found: {MODEL_PATH}")
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(f"Frozen Phase 1 scaler not found: {SCALER_PATH}")
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    # The frozen scaler embeds its training provenance and feature order. The
    # historical dataset is intentionally not required at inference time.
    scaler, provenance = load_phase1_scaler_24m(SCALER_PATH, dataset_path=None)
    if tuple(model.input_shape[1:]) != (24, 5):
        raise ValueError(f"Expected model input (24, 5), found {model.input_shape}.")
    if int(model.output_shape[-1]) != 5:
        raise ValueError(f"Expected five output classes, found {model.output_shape}.")
    if int(scaler.n_features_in_) != 5:
        raise ValueError(f"Expected scaler with five features, found {scaler.n_features_in_}.")
    if provenance.get("provenance", {}).get("feature_names_model_order") != FEATURES:
        raise ValueError("Scaler feature order does not match the Phase 1 inference input order.")
    return model, scaler


def predict_window(df: pd.DataFrame, model, scaler) -> dict:
    """Transform one validated window and make one frozen-model prediction."""
    x = df[FEATURES].to_numpy(dtype=np.float32)
    x_scaled = scaler.transform(x).astype(np.float32).reshape(1, 24, 5)
    probabilities = np.asarray(model(x_scaled, training=False).numpy(), dtype=float)
    if probabilities.shape != (1, 5):
        raise ValueError(f"Model returned shape {probabilities.shape}; expected (1, 5).")
    probabilities = probabilities[0]
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Model returned invalid class probabilities.")
    total = float(probabilities.sum())
    if not np.isclose(total, 1.0, atol=1e-4):
        raise ValueError(f"Model probabilities sum to {total:.6f}, not 1.")
    target_month = (df["date"].iloc[-1].to_period("M") + 1)
    return {
        "target_month": str(target_month),
        "predicted_class": CLASS_NAMES[int(np.argmax(probabilities))],
        "probabilities": probabilities,
        "input_start": df["date"].iloc[0].strftime("%Y-%m"),
        "input_end": df["date"].iloc[-1].strftime("%Y-%m"),
    }


def render_app():
    st.markdown(
        '<div class="hero"><h1>Phase 1 Drought Prediction</h1>'
        '<p>Upload 24 consecutive months of rainfall, temperature, soil moisture, VHI, and groundwater inputs. '
        'The frozen Phase 1 Attention-LSTM predicts the following month.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("Input requirements")
    st.markdown(
        "- Exactly **24 rows**, oldest to newest, representing **T-24 through T-1**.\n"
        "- Dates are consecutive month starts in `YYYY-MM-01` format.\n"
        "- Use only these five numeric predictors, in model order: rainfall, temperature, soil moisture, VHI, groundwater depth.\n"
        "- Do not include target-month observations or an SPEI/target column."
    )
    st.code(",".join(REQUIRED_COLUMNS) + "\n2024-01-01,...", language="csv")

    if SAMPLE_CSV_PATH.is_file():
        with st.container(border=True):
            st.markdown("**Synthetic demo file**")
            st.caption("The sample values are synthetic and are not observed real-world data. Use them only to demonstrate the upload workflow.")
            st.download_button(
                "Download synthetic 24-month sample CSV",
                data=SAMPLE_CSV_PATH.read_bytes(),
                file_name=SAMPLE_CSV_PATH.name,
                mime="text/csv",
            )

    uploaded = st.file_uploader("Upload a 24-month CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is None:
        return
    try:
        raw = pd.read_csv(uploaded)
        validated = validate_input(raw)
        st.success(
            f"Input validated: {len(validated)} consecutive months, "
            f"{validated['date'].iloc[0]:%Y-%m} through {validated['date'].iloc[-1]:%Y-%m}."
        )
        with st.expander("Review the 24 input rows", expanded=False):
            st.dataframe(validated, hide_index=True, width="stretch")
        model, scaler = load_phase1_assets()
        result = predict_window(validated, model, scaler)
        st.subheader("Prediction")
        st.markdown(
            '<div class="result-card"><div class="result-label">Predicted drought category</div>'
            f'<div class="result-value">{result["predicted_class"]}</div>'
            f'<div class="result-month">Target month: {result["target_month"]}</div></div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Input window: {result['input_start']} through {result['input_end']} (T-24 to T-1)")
        st.markdown("**Model probabilities**")
        probability_table = pd.DataFrame({
            "Class": CLASS_NAMES,
            "Probability": result["probabilities"],
        })
        st.bar_chart(probability_table.set_index("Class"), height=280)
        st.dataframe(
            probability_table.style.format({"Probability": "{:.4f}"}),
            hide_index=True,
            width="stretch",
        )
        st.info(
            "This prediction is generated from the supplied 24-month input data using the Phase 1 trained model. "
            "It is a model output and should not be interpreted as a guaranteed real-world forecast."
        )
    except Exception as exc:
        st.error(f"Prediction could not be generated: {exc}")


render_app()


