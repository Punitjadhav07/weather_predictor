#!/usr/bin/env python3
"""Train XGBoost (same pipeline as Phase 2) and save model + encoder + feature list for Streamlit."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

from weather_dss.artifacts import artifact_bundle_path, write_meta  # noqa: E402
from weather_dss.data_processing import process_dataset  # noqa: E402
from weather_dss.fuzzy_weather import (  # noqa: E402
    build_fuzzy_system_from_dataframe,
    infer_fuzzy_decision,
)
from weather_dss.hybrid_fusion import ml_label_to_coarse_category  # noqa: E402
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

    # ── Data-driven disagreement threshold ───────────────────────────
    #
    # disagreement = | ML(rain-like probability) − fuzzy_wet_score / 100 |
    #
    # Computed on the validation set so the threshold adapts to the
    # actual distribution of ML vs fuzzy disagreements seen during
    # training.  The 75th percentile (p75) is saved as the threshold;
    # at inference, disagreement ≥ this value triggers the "uncertain /
    # directional" fusion path when ML confidence is only medium.
    #
    # No hardcoded threshold is used anywhere in the pipeline.
    # ────────────────────────────────────────────────────────────────
    fuzzy_sys = build_fuzzy_system_from_dataframe(df)

    ml_proba_val = model.predict_proba(X_val)
    rain_mask = np.array(
        [ml_label_to_coarse_category(cls) in ("rain-like", "storm-like") for cls in le.classes_],
        dtype=bool,
    )
    ml_rain_prob_val = ml_proba_val[:, rain_mask].sum(axis=1)

    # Subsample validation set for fuzzy inference speed.
    n_val = len(X_val)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n_val, size=min(4000, n_val), replace=False)

    fuzzy_wet_norm_val = []
    for i in idx:
        row = X_val.iloc[int(i)]
        fuzzy_score, _ = infer_fuzzy_decision(
            fuzzy_sys,
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["precipitation"]),
            float(row["wind_speed"]),
        )
        fuzzy_wet_norm_val.append(float(fuzzy_score) / 100.0)

    fuzzy_wet_norm_val = np.asarray(fuzzy_wet_norm_val, dtype=float)
    ml_rain_prob_sub = ml_rain_prob_val[idx]
    disagreement = np.abs(ml_rain_prob_sub - fuzzy_wet_norm_val)
    disagreement_high = float(np.percentile(disagreement, 75))

    # Print distribution stats for traceability
    print(
        f"Disagreement distribution (n={len(disagreement)}): "
        f"mean={float(disagreement.mean()):.4f}, "
        f"p50={float(np.percentile(disagreement, 50)):.4f}, "
        f"p75={disagreement_high:.4f}"
    )

    feat_names = list(X_train.columns)
    bundle = {
        "model": model,
        "label_encoder": le,
        "feature_columns": feat_names,
        "val_accuracy": val_acc,
        "test_accuracy": test_acc,
        "confidence_low_threshold": conf_lo,
        "confidence_high_threshold": conf_hi,
        "disagreement_high_threshold": disagreement_high,
    }
    path = artifact_bundle_path()
    joblib.dump(bundle, path)
    write_meta(
        val_acc,
        test_acc,
        n_classes=len(le.classes_),
        confidence_low_threshold=conf_lo,
        confidence_high_threshold=conf_hi,
        disagreement_high_threshold=disagreement_high,
    )
    print(f"Saved bundle to {path}")
    print(f"Validation accuracy: {val_acc:.4f}")
    print(f"Test accuracy (holdout): {test_acc:.4f}")
    print(
        "Confidence bands (val max-proba): "
        f"low≤p25={conf_lo:.4f}, high≥p75={conf_hi:.4f}"
    )
    print(f"Disagreement high threshold (p75): {disagreement_high:.4f}")
    print(f"Classes: {len(le.classes_)}, Features: {len(feat_names)}")


if __name__ == "__main__":
    main()
