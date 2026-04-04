"""
Phase 4–5 — Streamlit UI: XGBoost + fuzzy hybrid with validation-calibrated confidence bands.

Run from project root:
  streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import joblib
import streamlit as st

ROOT = Path(__file__).resolve().parent
_mpl = ROOT / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

from weather_dss.artifacts import artifact_bundle_path  # noqa: E402
from weather_dss.data_processing import process_dataset, season_from_month_int  # noqa: E402
from weather_dss.fuzzy_weather import (  # noqa: E402
    FuzzyWeatherSystem,
    build_fuzzy_system_from_dataframe,
    infer_fuzzy_decision,
)
from weather_dss.hybrid_fusion import fuse_hybrid  # noqa: E402
from weather_dss.prediction_utils import (  # noqa: E402
    build_model_input_dataframe,
    predict_with_proba,
)


@st.cache_resource
def load_ml_bundle():
    path = artifact_bundle_path()
    if not path.is_file():
        return None
    return joblib.load(path)


@st.cache_resource
def load_fuzzy_system() -> FuzzyWeatherSystem:
    df = process_dataset()
    return build_fuzzy_system_from_dataframe(df)


def main() -> None:
    st.set_page_config(page_title="Weather DSS", page_icon="🌦️", layout="centered")
    st.title("Hybrid weather prediction + decision support")
    st.caption("XGBoost + fuzzy logic — validation-calibrated confidence + wind-aware fuzzy")

    bundle = load_ml_bundle()
    if bundle is None:
        st.error(
            "Trained model bundle not found. From the project folder run:\n\n"
            "`python scripts/save_model_artifacts.py`\n\n"
            "Then refresh this app."
        )
        return

    conf_lo = bundle.get("confidence_low_threshold")
    conf_hi = bundle.get("confidence_high_threshold")
    if conf_lo is None or conf_hi is None:
        st.error(
            "Saved model is missing calibrated confidence thresholds. Re-export with:\n\n"
            "`python scripts/save_model_artifacts.py`"
        )
        return

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]

    fuzzy_sys = load_fuzzy_system()

    st.subheader("Inputs")
    with st.form("weather_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            month = st.selectbox(
                "Month (for season features)",
                options=list(range(1, 13)),
                format_func=lambda m: f"{m:02d} — {season_from_month_int(m)}",
                index=max(0, min(11, datetime.now().month - 1)),
            )
            temperature = st.number_input("Temperature (°C)", value=22.0, step=0.5, format="%.1f")
            humidity = st.number_input("Humidity (%)", value=55.0, min_value=0.0, max_value=100.0, step=1.0)
            precipitation = st.number_input(
                "Precipitation (mm)", value=0.0, min_value=0.0, step=0.1, format="%.2f"
            )
        with c2:
            pressure = st.number_input("Pressure (mb)", value=1013.0, step=0.5, format="%.1f")
            wind_speed = st.number_input("Wind speed (kph)", value=12.0, min_value=0.0, step=0.5, format="%.1f")
            visibility = st.number_input("Visibility (km)", value=10.0, min_value=0.0, step=0.5, format="%.1f")
        with c3:
            st.markdown("**Hybrid logic** uses ML max-proba vs validation **p25 / p75** bands.")
            st.caption(f"Low ≤ {conf_lo:.3f} · High ≥ {conf_hi:.3f}")
            submitted = st.form_submit_button("Predict weather", type="primary", use_container_width=True)

    if not submitted:
        st.info("Set parameters and click **Predict weather**.")
        return

    X = build_model_input_dataframe(
        temperature=temperature,
        humidity=humidity,
        precipitation=precipitation,
        pressure=pressure,
        wind_speed=wind_speed,
        visibility=visibility,
        feature_columns=feature_columns,
        month=int(month),
    )
    ml_label, ml_conf, _ = predict_with_proba(model, X, label_encoder)
    fuzzy_score, fuzzy_label = infer_fuzzy_decision(
        fuzzy_sys, temperature, humidity, precipitation, wind_speed
    )

    hybrid = fuse_hybrid(
        ml_label,
        ml_conf,
        fuzzy_label,
        fuzzy_score,
        conf_low=float(conf_lo),
        conf_high=float(conf_hi),
        temperature_c=temperature,
        humidity_pct=humidity,
        precipitation_mm=precipitation,
        wind_speed_kph=wind_speed,
        visibility_km=visibility,
    )

    st.divider()
    st.subheader("Results")

    st.markdown("#### Machine learning")
    c_ml1, c_ml2 = st.columns(2)
    c_ml1.metric("Predicted class", str(hybrid.ml_prediction))
    c_ml2.metric("Confidence (max probability)", f"{hybrid.ml_confidence:.1%}")

    st.markdown("#### Fuzzy inference")
    st.markdown(
        f"- **Linguistic label:** `{hybrid.fuzzy_label}`  \n"
        f"- **Crisp score:** `{hybrid.fuzzy_score:.2f}` / 100 *(higher → wetter / more unsettled)*"
    )

    st.markdown("#### Hybrid fusion")
    st.markdown(
        f"- **Mode:** `{hybrid.fusion_mode}`  \n"
        f"- **Bands used:** low ≤ `{hybrid.conf_low_threshold:.3f}`, high ≥ `{hybrid.conf_high_threshold:.3f}`"
    )
    st.success(hybrid.final_decision)

    st.markdown("#### Recommendations")
    for r in hybrid.recommendations:
        st.markdown(f"- {r}")

    with st.expander("Technical details"):
        st.write("Feature columns (training order):", feature_columns)
        st.dataframe(X, use_container_width=True)


if __name__ == "__main__":
    main()
