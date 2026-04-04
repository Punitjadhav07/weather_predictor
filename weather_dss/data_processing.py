"""
Phase 1 — Data processing for the weather dataset.

Loads GlobalWeatherRepository-style CSVs, selects model features, imputes
missing values (no blind full-row drops), and adds engineered columns.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

# Canonical names used downstream (ML, fuzzy, UI)
RAW_TO_CANONICAL = {
    "temperature_celsius": "temperature",
    "humidity": "humidity",
    "pressure_mb": "pressure",
    "wind_kph": "wind_speed",
    "precip_mm": "precipitation",
    "visibility_km": "visibility",
    "condition_text": "weather_condition",
}

# Extra source columns needed for feature engineering
ENGINEERING_SOURCE_COLS = ("last_updated", "gust_kph")


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def default_csv_path() -> Path:
    """Path to the bundled global weather CSV (override in tests or CLI)."""
    return _project_root() / "Dataset" / "GlobalWeatherRepository.csv"


def load_raw_data(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the full repository CSV with pandas."""
    path = Path(csv_path) if csv_path is not None else default_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path, low_memory=False)


def select_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep relevant columns: temperature, humidity, pressure, wind, precip,
    visibility, condition, plus timestamps / gust for engineering.
    """
    need = list(RAW_TO_CANONICAL.keys()) + list(ENGINEERING_SOURCE_COLS)
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing expected columns: {missing}")
    out = df[need].copy()
    out = out.rename(columns=RAW_TO_CANONICAL)
    return out


def season_from_month_int(month: int) -> str:
    """Single-month meteorological season (for UI / inference rows)."""
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    return "unknown"


def _season_from_month(month: pd.Series) -> pd.Series:
    """Meteorological seasons (Dec–Feb winter, etc.)."""
    m = month.astype("Int64")

    def bucket(x) -> str:
        if pd.isna(x):
            return "unknown"
        xi = int(x)
        if xi in (12, 1, 2):
            return "winter"
        if xi in (3, 4, 5):
            return "spring"
        if xi in (6, 7, 8):
            return "summer"
        return "autumn"

    return m.map(bucket)


def normalize_weather_condition(series: pd.Series) -> pd.Series:
    """
    Canonical label form: lowercase, stripped, internal whitespace collapsed.
    Merges variants like 'Partly Cloudy' and 'partly cloudy'.
    """
    s = series.astype(str).str.strip().str.lower()
    s = s.str.replace(r"\s+", " ", regex=True)
    return s


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values instead of dropping most rows.

    - Numeric features: median (robust to outliers).
    - weather_condition: mode, then 'Unknown' if still missing.
    - last_updated: rows without a parseable date keep month/season as unknown
      after engineering (we do not drop them).
    """
    out = df.copy()
    numeric = [
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "precipitation",
        "visibility",
        "gust_kph",
    ]
    for col in numeric:
        if col in out.columns:
            med = out[col].median()
            out[col] = out[col].fillna(med)

    if "weather_condition" in out.columns:
        mode = out["weather_condition"].mode(dropna=True)
        fill_cat = mode.iloc[0] if len(mode) else "Unknown"
        out["weather_condition"] = out["weather_condition"].fillna(fill_cat).astype(str)
        out.loc[out["weather_condition"].str.strip() == "", "weather_condition"] = fill_cat

    return out


def sanitize_physical_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix rare corrupt readings (e.g. pressure in thousands) by replacing with
    the column median so we do not drop entire rows.
    """
    out = df.copy()
    if "pressure" in out.columns:
        med_p = float(out["pressure"].median())
        bad_p = (out["pressure"] < 870) | (out["pressure"] > 1100)
        out.loc[bad_p, "pressure"] = med_p
    if "wind_speed" in out.columns:
        med_w = float(out["wind_speed"].median())
        out.loc[out["wind_speed"] > 200, "wind_speed"] = med_w
    if "gust_kph" in out.columns:
        med_g = float(out["gust_kph"].median())
        out.loc[out["gust_kph"] > 250, "gust_kph"] = med_g
    if "visibility" in out.columns:
        out["visibility"] = out["visibility"].clip(lower=0)
    if "precipitation" in out.columns:
        out["precipitation"] = out["precipitation"].clip(lower=0)
    return out


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add: month, season, temp_humidity, precip_indicator.

    Uses ``wind_speed`` only (no separate gust-derived column) so training and
    UI inference share identical feature semantics.
    """
    out = df.copy()

    if "last_updated" in out.columns:
        dt = pd.to_datetime(out["last_updated"], errors="coerce")
        out["month"] = dt.dt.month
        out["season"] = _season_from_month(out["month"])
    else:
        out["month"] = np.nan
        out["season"] = "unknown"

    out["temp_humidity"] = out["temperature"] * (out["humidity"] / 100.0)

    out["precip_indicator"] = (out["precipitation"] > 0.01).astype(int)

    return out


def drop_redundant_engineering_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns only needed for engineering (optional tidy export)."""
    out = df.drop(columns=["last_updated", "gust_kph"], errors="ignore")
    return out


def process_dataset(
    csv_path: str | Path | None = None,
    *,
    drop_engineering_sources: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: load → select → impute → engineer.

    Returns a DataFrame ready for train/test split in Phase 2.
    """
    raw = load_raw_data(csv_path)
    selected = select_feature_columns(raw)
    filled = handle_missing_values(selected)
    cleaned = sanitize_physical_bounds(filled)
    engineered = engineer_features(cleaned)
    if drop_engineering_sources:
        engineered = drop_redundant_engineering_sources(engineered)
    if "weather_condition" in engineered.columns:
        engineered["weather_condition"] = normalize_weather_condition(
            engineered["weather_condition"]
        )
    return engineered


def processing_summary(df: pd.DataFrame) -> dict:
    """Small stats dict for logging or Streamlit later."""
    return {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_nulls": df.select_dtypes(include=[np.number]).isna().sum().sum(),
        "weather_condition_nunique": df["weather_condition"].nunique()
        if "weather_condition" in df.columns
        else None,
    }
