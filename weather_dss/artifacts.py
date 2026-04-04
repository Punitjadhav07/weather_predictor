"""Paths and helpers for saved XGBoost + encoder artifacts (Streamlit / batch)."""

from __future__ import annotations

import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    d = project_root() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def artifact_bundle_path() -> Path:
    return models_dir() / "xgboost_bundle.joblib"


def artifact_meta_path() -> Path:
    return models_dir() / "training_meta.json"


def write_meta(
    val_acc: float,
    test_acc: float | None,
    n_classes: int,
    *,
    confidence_low_threshold: float | None = None,
    confidence_high_threshold: float | None = None,
) -> None:
    meta = {
        "validation_accuracy": val_acc,
        "test_accuracy": test_acc,
        "n_classes": n_classes,
        "confidence_low_threshold": confidence_low_threshold,
        "confidence_high_threshold": confidence_high_threshold,
    }
    artifact_meta_path().write_text(json.dumps(meta, indent=2), encoding="utf-8")
