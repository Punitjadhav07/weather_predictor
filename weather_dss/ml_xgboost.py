"""
Phase 2 — XGBoost classifier with stratified 60% / 20% / 20% train/val/test split.

Uses processed data from Phase 1; target encoded with LabelEncoder; no feature scaling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover - platform-specific
    raise ImportError(
        "XGBoost failed to import (often missing OpenMP on macOS). "
        "Try: brew install libomp\n"
        f"Original error: {exc}"
    ) from exc

from weather_dss.data_processing import process_dataset

RANDOM_STATE = 42
# Classes with fewer than this many rows are merged into "other" to reduce
# fragmentation and stabilize learning (after label normalization).
MERGE_SMALL_CLASSES_MIN_COUNT = 75


@dataclass
class SplitBundle:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder
    feature_columns: list[str]


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def outputs_dir() -> Path:
    d = _project_root() / "outputs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def consolidate_small_classes(
    df: pd.DataFrame,
    *,
    column: str = "weather_condition",
    min_count: int = MERGE_SMALL_CLASSES_MIN_COUNT,
    other_label: str = "other",
) -> pd.DataFrame:
    """Merge classes with support < min_count into a single 'other' bucket."""
    out = df.copy()
    counts = out[column].value_counts()
    rare = counts[counts < min_count].index
    if len(rare):
        out.loc[out[column].isin(rare), column] = other_label
        print(
            f"[consolidate] Merged {len(rare)} small classes (<{min_count} samples) "
            f"into {other_label!r}."
        )
    return out


def print_class_distribution(df: pd.DataFrame, *, title: str = "weather_condition") -> None:
    vc = df["weather_condition"].value_counts()
    print(f"\n--- Class distribution ({title}) ---")
    print(f"Distinct labels: {vc.shape[0]}")
    # After normalization, many rows share the same canonical label; check no empty
    bad = df["weather_condition"].astype(str).str.strip().eq("")
    if bad.any():
        print(f"Warning: {bad.sum()} empty labels")
    print(vc.head(25).to_string())
    if len(vc) > 25:
        print(f"... ({len(vc) - 25} more classes)")


def print_top_feature_importances(
    model: XGBClassifier,
    feature_names: list[str],
    top_n: int = 10,
) -> list[tuple[str, float]]:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    rows: list[tuple[str, float]] = [(feature_names[i], float(imp[i])) for i in order]
    print(f"\nTop {top_n} features (importance):")
    for name, v in rows:
        print(f"  {name}: {v:.5f}")
    must = ("precipitation", "humidity", "temperature")
    top_names = {r[0] for r in rows}
    missing = [m for m in must if m not in top_names]
    if missing:
        rank_map = {feature_names[i]: pos + 1 for pos, i in enumerate(np.argsort(imp)[::-1])}
        details = ", ".join(f"{m}→#{rank_map.get(m, '?')}" for m in missing)
        print(
            f"[feature check] Not all of {{precipitation, humidity, temperature}} appear in top {top_n}. "
            f"Global ranks: {details}"
        )
    else:
        print("[feature check] precipitation, humidity, and temperature are all in the top 10.")
    return rows


def print_prediction_samples(
    model: XGBClassifier,
    X: pd.DataFrame,
    y_true: np.ndarray,
    label_encoder: LabelEncoder,
    n: int = 8,
    *,
    random_state: int = RANDOM_STATE,
) -> None:
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    pred = model.predict(X.iloc[idx])
    print(f"\n--- Sample predictions (n={len(idx)}, test/val rows) ---")
    for i, row_i in enumerate(idx):
        t = y_true[row_i]
        p = pred[i]
        actual = label_encoder.inverse_transform([t])[0]
        predicted = label_encoder.inverse_transform([p])[0]
        print(f"  {i + 1}. actual={actual!r}  |  predicted={predicted!r}")


def prepare_xy(
    df: pd.DataFrame,
    *,
    min_class_samples: int = 10,
    merge_small_min_count: int | None = MERGE_SMALL_CLASSES_MIN_COUNT,
) -> tuple[pd.DataFrame, np.ndarray, LabelEncoder]:
    """
    Build numeric feature matrix X and encoded target y.

    Drops rare weather_condition classes so nested stratified splits are valid
    (each split needs enough items per class).
    """
    work = df.copy()
    if merge_small_min_count is not None and merge_small_min_count > 0:
        work = consolidate_small_classes(work, min_count=merge_small_min_count)
    counts = work["weather_condition"].value_counts()
    keep = counts[counts >= min_class_samples].index
    work = work[work["weather_condition"].isin(keep)].reset_index(drop=True)
    dropped = int(len(df) - len(work))
    if dropped:
        print(
            f"[prepare_xy] Removed {dropped} rows (small classes / not in keep set) "
            f"for stable stratified splits (min_class_samples={min_class_samples})."
        )

    y_raw = work["weather_condition"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X = work.drop(columns=["weather_condition"])
    X = pd.get_dummies(X, columns=["season"], dtype=float)

    return X, y, le


def stratified_split_60_20_20(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    60% train, 20% validation, 20% test — stratified on y (weather_condition).
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.4,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def default_xgb_params(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
    }
    base.update(overrides)
    return base


def train_model(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    params: dict[str, Any] | None = None,
) -> XGBClassifier:
    params = default_xgb_params() if params is None else default_xgb_params(**params)
    clf = XGBClassifier(**params)
    clf.fit(X_train, y_train)
    return clf


def evaluate_accuracy(model: XGBClassifier, X: pd.DataFrame, y: np.ndarray) -> float:
    pred = model.predict(X)
    return float(accuracy_score(y, pred))


def confidence_percentile_thresholds(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    *,
    p_low: float = 25.0,
    p_high: float = 75.0,
) -> tuple[float, float]:
    """
    Data-driven fusion bands: percentiles of max predicted class probability on
    the validation set (correct rows only use val data, never test).
    """
    proba = model.predict_proba(X_val)
    max_p = np.asarray(proba.max(axis=1), dtype=float)
    lo = float(np.percentile(max_p, p_low))
    hi = float(np.percentile(max_p, p_high))
    if hi <= lo:
        hi = float(min(1.0, lo + 1e-3))
    return lo, hi


def optional_tune_on_validation(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    base_params: dict[str, Any],
) -> tuple[XGBClassifier, dict[str, Any], float]:
    """
    Train baseline; if a slightly deeper / larger model improves val accuracy, use it.
    Does not touch the test set.
    """
    base_model = train_model(X_train, y_train, params=base_params)
    base_val = evaluate_accuracy(base_model, X_val, y_val)

    candidates = [
        {**base_params, "max_depth": 6},
        {**base_params, "n_estimators": 150},
        {**base_params, "max_depth": 6, "n_estimators": 150},
    ]

    best_model = base_model
    best_params = dict(base_params)
    best_val = base_val

    for cand in candidates:
        m = train_model(X_train, y_train, params=cand)
        v = evaluate_accuracy(m, X_val, y_val)
        if v > best_val:
            best_model, best_params, best_val = m, cand, v

    if best_val > base_val:
        print(
            f"[tune] Validation improved {base_val:.4f} → {best_val:.4f} "
            f"with params: { {k: best_params[k] for k in ('max_depth', 'n_estimators') if k in best_params} }"
        )
    else:
        print(f"[tune] Kept baseline (val={base_val:.4f}); candidates did not improve.")

    return best_model, best_params, best_val


def plot_feature_importance(
    model: XGBClassifier,
    feature_names: list[str],
    out_path: Path,
    top_n: int = 15,
) -> None:
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    names = [feature_names[i] for i in order]
    vals = imp[order]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.25)))
    ax.barh(names[::-1], vals[::-1], color="steelblue")
    ax.set_xlabel("Importance (gain-based, XGBoost)")
    ax.set_title(f"Top {top_n} feature importances")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] Saved feature importance figure: {out_path}")


def print_confusion_matrix_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: LabelEncoder,
    max_classes_print: int = 12,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion matrix shape: {cm.shape} (classes × classes)")
    acc = accuracy_score(y_true, y_pred)
    print(f"Test accuracy (from held-out set): {acc:.4f}")

    # Per-class diagonal / support — quick view without printing 49×49
    support = cm.sum(axis=1)
    correct = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, correct / support, 0.0)
    order = np.argsort(support)[::-1]
    names = label_encoder.classes_
    print(f"\nPer-class recall (top {max_classes_print} by frequency):")
    for idx in order[:max_classes_print]:
        print(
            f"  {names[idx]!r}: recall={recall[idx]:.3f}, support={support[idx]}"
        )

    # Full matrix numeric (may be large)
    print("\nFull confusion matrix (rows=true, cols=pred):")
    np.set_printoptions(threshold=min(cm.size, 400), linewidth=120)
    print(cm)
    np.set_printoptions()


def run_phase2_pipeline(
    csv_path: str | Path | None = None,
    *,
    random_state: int = RANDOM_STATE,
    do_optional_tune: bool = True,
    verbose_prechecks: bool = True,
) -> tuple[XGBClassifier, SplitBundle, float, float, np.ndarray]:
    """
    End-to-end Phase 2: load Phase-1 data → split → train → val → [tune] → test.

    Returns (model, splits, val_accuracy, test_accuracy, confusion_matrix_test).
    """
    df = process_dataset(csv_path)
    if verbose_prechecks:
        print("Labels are normalized in Phase 1 (lowercase, stripped, collapsed spaces).")
        print_class_distribution(df, title="after Phase 1 normalization")

    X, y, le = prepare_xy(df)
    if verbose_prechecks:
        print_class_distribution(
            pd.DataFrame({"weather_condition": le.inverse_transform(y)}),
            title="after consolidation + stratify filter (modeling set)",
        )

    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split_60_20_20(
        X, y, random_state=random_state
    )

    base_params = default_xgb_params(random_state=random_state)

    if do_optional_tune:
        model, used_params, val_acc = optional_tune_on_validation(
            X_train, y_train, X_val, y_val, base_params
        )
    else:
        model = train_model(X_train, y_train, params=base_params)
        used_params = base_params
        val_acc = evaluate_accuracy(model, X_val, y_val)

    print(f"\nValidation accuracy: {val_acc:.4f}")

    y_test_pred = model.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_test_pred))
    print(f"Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix(y_test, y_test_pred)
    print_confusion_matrix_summary(y_test, y_test_pred, le)

    feat_names = list(X_train.columns)
    plot_feature_importance(
        model,
        feat_names,
        outputs_dir() / "feature_importance_xgb.png",
        top_n=min(20, len(feat_names)),
    )

    print_top_feature_importances(model, feat_names, top_n=10)
    imp = model.feature_importances_
    rank = np.argsort(imp)[::-1][:15]
    print("\n(Extended) Top 15 features:")
    for i in rank:
        print(f"  {feat_names[i]}: {imp[i]:.5f}")

    if verbose_prechecks:
        print_prediction_samples(model, X_test, y_test, le, n=8, random_state=random_state)

    bundle = SplitBundle(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        label_encoder=le,
        feature_columns=feat_names,
    )
    return model, bundle, val_acc, test_acc, cm
