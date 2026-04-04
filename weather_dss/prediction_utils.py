"""Build a feature row aligned with training columns from UI / API inputs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from weather_dss.data_processing import season_from_month_int


def build_model_input_dataframe(
    *,
    temperature: float,
    humidity: float,
    precipitation: float,
    pressure: float,
    wind_speed: float,
    visibility: float,
    feature_columns: list[str],
    month: int | None = None,
    when: datetime | None = None,
) -> pd.DataFrame:
    """
    Reconstruct the engineered feature vector used in training (Phase 1/2).

    ``month`` (1–12) selects season dummies; if omitted, uses ``when`` or UTC “now”.
    Feature set matches ``process_dataset`` / ``prepare_xy`` (no gust-derived columns).
    """
    if month is not None:
        m = int(np.clip(int(month), 1, 12))
    elif when is not None:
        m = int(when.month)
    else:
        m = int(datetime.now(timezone.utc).month)
    season = season_from_month_int(m)
    temp_humidity = float(temperature) * (float(humidity) / 100.0)
    precip_indicator = 1 if float(precipitation) > 0.01 else 0

    row: dict[str, Any] = {
        "temperature": float(temperature),
        "humidity": float(humidity),
        "precipitation": float(precipitation),
        "pressure": float(pressure),
        "wind_speed": float(wind_speed),
        "visibility": float(visibility),
        "month": float(m),
        "season": season,
        "temp_humidity": float(temp_humidity),
        "precip_indicator": precip_indicator,
    }
    df = pd.DataFrame([row])
    df = pd.get_dummies(df, columns=["season"], dtype=float)
    for c in feature_columns:
        if c not in df.columns:
            df[c] = 0.0
    extra = [c for c in df.columns if c not in feature_columns]
    if extra:
        df = df.drop(columns=extra)
    return df[feature_columns]


def predict_with_proba(model: Any, X: pd.DataFrame, label_encoder: Any) -> tuple[str, float, int]:
    """Return (label, confidence=max proba, class_index)."""
    proba = model.predict_proba(X)[0]
    idx = int(proba.argmax())
    conf = float(proba[idx])
    label = label_encoder.inverse_transform([idx])[0]
    return label, conf, idx
