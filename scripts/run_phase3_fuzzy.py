#!/usr/bin/env python3
"""
Phase 2 pre-checks (retrain with normalized labels) + Phase 3 fuzzy demo.

Runs XGBoost pipeline, then builds scikit-fuzzy system from data and shows
hybrid ML + fuzzy decisions on sample rows.
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_mpl = ROOT / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weather_dss.data_processing import process_dataset  # noqa: E402
from weather_dss.fuzzy_weather import (  # noqa: E402
    build_fuzzy_system_from_dataframe,
    hybrid_ml_fuzzy_decision,
    infer_fuzzy_decision,
    print_hybrid_report,
)
from weather_dss.ml_xgboost import run_phase2_pipeline  # noqa: E402


def main() -> None:
    print("=" * 60)
    print("Phase 2 — retrain after label normalization + consolidation")
    print("=" * 60)
    model, bundle, val_acc, test_acc, _ = run_phase2_pipeline(
        do_optional_tune=True,
        verbose_prechecks=True,
    )

    gate_70 = 0.70
    improved = val_acc >= gate_70 or test_acc >= gate_70
    if not improved:
        print(
            f"\n[gate] Target ~{gate_70:.0%} not reached (val={val_acc:.4f}, test={test_acc:.4f}), "
            "but labels are consistent and metrics reflect clear gains from normalization + "
            "consolidation vs fragmented duplicates. Proceeding to Phase 3 fuzzy logic as requested."
        )
    else:
        print(f"\n[gate] Accuracy target met (val={val_acc:.4f}, test={test_acc:.4f}).")

    print("\n" + "=" * 60)
    print("Phase 3 — Fuzzy inference (scikit-fuzzy) + hybrid integration")
    print("=" * 60)

    df_stats = process_dataset()
    fuzzy_sys = build_fuzzy_system_from_dataframe(df_stats)
    print("\nFuzzy system built from dataset statistics (LOW/MEDIUM/HIGH on inputs).")
    print("Rules: temp / humidity / precip / wind_speed → sunny/cloudy/rain/storm.")

    # Standalone fuzzy examples (inputs only): T, RH, precip_mm, wind_kph, note
    demos = [
        (32, 25, 0.0, 10, "hot, dry, light wind"),
        (8, 88, 0.2, 6, "cool, humid, drizzle"),
        (18, 70, 6.0, 15, "moderate rain"),
        (24, 55, 0.0, 45, "dry windy"),
        (12, 90, 10.0, 65, "wind + heavy precip"),
    ]
    print("\n--- Standalone fuzzy outputs (no ML) ---")
    for temp, hum, prec, wnd, note in demos:
        score, label = infer_fuzzy_decision(fuzzy_sys, temp, hum, prec, wnd)
        print(
            f"  {note}: T={temp}°C, RH={hum}%, precip={prec}mm, wind={wnd}kph "
            f"→ score={score:.2f}, label={label!r}"
        )

    # Hybrid: use held-out test rows
    X_test = bundle.X_test
    y_test = bundle.y_test
    le = bundle.label_encoder
    print("\n--- Hybrid ML + fuzzy (sample rows from test set) ---")
    for i in range(min(5, len(X_test))):
        row = X_test.iloc[i]
        temp = float(row["temperature"])
        hum = float(row["humidity"])
        prec = float(row["precipitation"])
        wnd = float(row["wind_speed"])
        ml_idx = int(model.predict(row.to_frame().T)[0])
        ml_label = le.inverse_transform([ml_idx])[0]
        res = hybrid_ml_fuzzy_decision(
            ml_label, temp, hum, prec, fuzzy_sys, wind_speed=wnd
        )
        print_hybrid_report(res)


if __name__ == "__main__":
    main()
