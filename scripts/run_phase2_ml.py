#!/usr/bin/env python3
"""Phase 2: stratified 60/20/20 XGBoost train → validation → optional tune → test."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_mpl = ROOT / ".mplconfig"
_mpl.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl))

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weather_dss.ml_xgboost import run_phase2_pipeline  # noqa: E402


def main() -> None:
    run_phase2_pipeline(do_optional_tune=True)


if __name__ == "__main__":
    main()
