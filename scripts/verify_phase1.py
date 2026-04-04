#!/usr/bin/env python3
"""Run Phase 1 pipeline and print a short report (no ML yet)."""

import sys
from pathlib import Path

# Allow running from repo without installing the package
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weather_dss.data_processing import (  # noqa: E402
    default_csv_path,
    process_dataset,
    processing_summary,
)


def main() -> None:
    path = default_csv_path()
    print(f"Loading: {path}")
    df = process_dataset(path)
    info = processing_summary(df)
    print(f"Rows: {info['rows']}")
    print(f"Columns ({len(info['columns'])}): {info['columns']}")
    print(f"Total numeric NaNs after processing: {info['numeric_nulls']}")
    print(f"Distinct weather_condition: {info['weather_condition_nunique']}")
    print()
    print(df.head(3).to_string())
    print()
    print(df.describe().T.to_string())


if __name__ == "__main__":
    main()
