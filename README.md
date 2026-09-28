# Water Scarcity Prediction — Phase 1

This Streamlit application runs the frozen Phase 1 Attention-LSTM classifier on one user-supplied 24-month input window. It predicts the drought category for the month immediately after the last input month.

## Input format

Upload a CSV with exactly 24 rows, oldest month first, and these columns:

```csv
date,rainfall_mm,temp_c,soil_moisture,vhi,groundwater_depth_m
2024-01-01,25.0,20.536,0.18,0.38,14.0
```

Dates must be month starts in `YYYY-MM-01` format and form an uninterrupted 24-month period. The app rejects missing, nonnumeric, or non-finite predictor values, duplicate months, extra columns, and invalid date sequences. The target month is derived as the month after the final input month.

The included `data/DEMO_SYNTHETIC_phase1_24m_input.csv` contains synthetic demonstration values. It is only for checking the upload and inference workflow and is not observed real-world data.

## Run locally

Python 3.12 is used with the pinned dependencies in `requirements.txt`.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The app loads only these frozen Phase 1 artifacts:

- `models/phase1_attention_lstm_24m.keras`
- `models/phase1_scaler_24m.pkl`

The model and scaler paths are resolved relative to `app.py`; no user-specific absolute path is required. `src/` contains the custom model layer and scaler validation helpers needed at runtime. The original training dataset and historical evaluation artifacts are not required to run inference.

## Prediction output and limitations

The app displays the predicted category for the derived target month and a probability for each of the five categories: No drought, Light drought, Moderate drought, Severe drought, and Extreme drought. Results depend on the values and locations represented by the supplied input; the app does not verify the source or real-world accuracy of uploaded observations. It is an inference demonstration, not a guaranteed forecast or a substitute for operational drought monitoring.

## Repository contents

The app runtime consists of `app.py`, `requirements.txt`, `models/`, the required helper modules in `src/`, and the synthetic demo CSV. Training scripts, historical reports, and research artifacts are separate from the inference path.

## Prediction disclaimer

This prediction is generated from the supplied 24-month input data using the Phase 1 trained model. It is a model output and should not be interpreted as a guaranteed real-world forecast.
