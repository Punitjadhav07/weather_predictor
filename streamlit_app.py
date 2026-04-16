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
    predict_with_proba_vector,
)


# ── Cached resource loaders ──────────────────────────────────────────────

@st.cache_resource
def load_ml_bundle():
    """Load the trained model bundle (XGBoost + encoder + calibrated thresholds)."""
    path = artifact_bundle_path()
    if not path.is_file():
        return None
    return joblib.load(path)


@st.cache_resource
def load_fuzzy_system() -> FuzzyWeatherSystem:
    """Build the fuzzy control system from the full dataset's quantiles."""
    df = process_dataset()
    return build_fuzzy_system_from_dataframe(df)


# ── UI helpers ────────────────────────────────────────────────────────────

def _confidence_badge(level: str) -> str:
    """Return a coloured emoji badge for the confidence level."""
    return {"Low": "🔴 Low", "Medium": "🟡 Medium", "High": "🟢 High"}.get(
        level, level
    )


def _decision_style(decision: str):
    """Display the final decision with appropriate Streamlit styling."""
    d = decision.lower()
    if "storm" in d:
        st.error(f"### {decision}")
    elif "rain" in d:
        st.warning(f"### {decision}")
    else:
        st.success(f"### {decision}")


# ── Main application ─────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(page_title="Weather DSS", page_icon="🌦️", layout="centered")
    st.title("Hybrid weather prediction + decision support")
    st.caption("XGBoost + fuzzy logic — validation-calibrated confidence + wind-aware fuzzy")

    # ── Load model artifacts ──────────────────────────────────────────
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
    disagreement_high_threshold = bundle.get("disagreement_high_threshold")
    if disagreement_high_threshold is None:
        st.error(
            "Saved model is missing `disagreement_high_threshold`. "
            "Re-export with `python scripts/save_model_artifacts.py`."
        )
        return

    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns: list[str] = bundle["feature_columns"]

    fuzzy_sys = load_fuzzy_system()

    # ── Input form ────────────────────────────────────────────────────
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

    # ── Run prediction pipeline ───────────────────────────────────────
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
    ml_label, ml_conf, _, ml_proba = predict_with_proba_vector(
        model, X, label_encoder
    )
    fuzzy_score, fuzzy_label = infer_fuzzy_decision(
        fuzzy_sys, temperature, humidity, precipitation, wind_speed
    )

    hybrid = fuse_hybrid(
        ml_label,
        ml_conf,
        ml_proba,
        label_encoder,
        fuzzy_label,
        fuzzy_score,
        conf_low=float(conf_lo),
        conf_high=float(conf_hi),
        disagreement_high_threshold=float(disagreement_high_threshold),
        temperature_c=temperature,
        humidity_pct=humidity,
        precipitation_mm=precipitation,
        wind_speed_kph=wind_speed,
        visibility_km=visibility,
    )

    # ── Results display ───────────────────────────────────────────────
    st.divider()

    # 1. Final Decision (large and clear)
    st.subheader("Final Decision")
    _decision_style(hybrid.final_decision)

    # 2. Confidence Level (coloured badge)
    st.markdown("#### Confidence Level")
    st.metric(label="Model confidence band", value=_confidence_badge(hybrid.confidence_level))
    st.caption(
        "Confidence thresholds are derived from validation-set percentiles: "
        "low = 25th pctl, high = 75th pctl of max predicted class probability."
    )

    # 3. Reasoning (2–3 bullets)
    st.markdown("#### Reasoning")
    for r in hybrid.reasoning:
        st.markdown(f"- {r}")

    # 4. Recommendations (priority-ordered)
    st.markdown("#### Recommendations")
    for r in hybrid.recommendations:
        st.markdown(f"- {r}")

    # 5. Technical Details (expandable — hidden by default)
    with st.expander("🔍 Technical Details"):
        st.markdown("**Machine Learning (XGBoost)**")
        ml_pred_display = (
            "Uncertain / Transitional"
            if hybrid.ml_prediction == "other"
            else str(hybrid.ml_prediction)
        )
        st.write("ML prediction:", ml_pred_display)
        st.write("ML coarse category:", hybrid.ml_coarse)
        st.write("ML confidence (max-proba):", f"{hybrid.ml_confidence:.3f}")

        st.markdown("**Fuzzy Logic**")
        st.write("Fuzzy label:", hybrid.fuzzy_label)
        st.write("Fuzzy coarse category:", hybrid.fuzzy_coarse)
        st.write("Fuzzy crisp wet score (0..100):", f"{hybrid.fuzzy_score:.2f}")

        st.markdown("**Disagreement Metric**")
        st.write("ML rain probability:", f"{hybrid.ml_rain_probability:.3f}")
        st.write("Fuzzy wet score (normalised):", f"{hybrid.fuzzy_score / 100:.3f}")
        st.write("Disagreement:", f"{hybrid.disagreement:.3f}")
        st.write(
            "Disagreement high threshold (p75):",
            f"{hybrid.disagreement_high_threshold:.3f}",
        )

        st.markdown("**Confidence Bands**")
        st.write(
            f"Low ≤ {hybrid.conf_low_threshold:.3f}, High ≥ {hybrid.conf_high_threshold:.3f}"
        )

        st.markdown("**Fusion Decision**")
        st.write("Final source:", hybrid.final_source)
        st.write("Fusion mode:", hybrid.fusion_mode)

        st.markdown("**Feature Vector**")
        st.write("Feature columns (training order):", feature_columns)
        st.dataframe(X, use_container_width=True)


if __name__ == "__main__":
    main()
