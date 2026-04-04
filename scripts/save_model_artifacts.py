#!/usr/bin/env python3
"""Train XGBoost (same pipeline as Phase 2) and save model + encoder + feature list for Streamlit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from weather_dss.artifacts import artifact_bundle_path, write_meta  # noqa: E402
from weather_dss.data_processing import process_dataset  # noqa: E402
from weather_dss.ml_xgboost import (  # noqa: E402
    RANDOM_STATE,
    confidence_percentile_thresholds,
    default_xgb_params,
    optional_tune_on_validation,
    prepare_xy,
    stratified_split_60_20_20,
)


def main() -> None:
    print("Loading data and training (60/20/20 stratified, validation tuning)...")
    df = process_dataset()
    X, y, le = prepare_xy(df)
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_60_20_20(
        X, y, random_state=RANDOM_STATE
    )
    base = default_xgb_params(random_state=RANDOM_STATE)
    model, params, val_acc = optional_tune_on_validation(
        X_train, y_train, X_val, y_val, base
    )
    test_acc = float(
        accuracy_score(y_test, model.predict(X_test))
    )
    conf_lo, conf_hi = confidence_percentile_thresholds(model, X_val)
    feat_names = list(X_train.columns)
    bundle = {
        "model": model,
        "label_encoder": le,
        "feature_columns": feat_names,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "confidence_low_threshold": conf_lo,
        "confidence_high_threshold": conf_hi,
    }
    path = artifact_bundle_path()
    joblib.dump(bundle, path)
    write_meta(
        val_acc,
        test_acc,
        n_classes=len(le.classes_),
        confidence_low_threshold=conf_lo,
        confidence_high_threshold=conf_hi,
    )
    print(f"Saved bundle to {path}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy (holdout): {test_acc:.4f}")
    print(f"Confidence bands (val max-proba): low≤p25={conf_lo:.4f}, high≥p75={conf_hi:.4f}")
    print(f"Classes: {len(le.classes_)}, Features: {len(feat_names)}")


if __name__ == "__main__":
    main()
