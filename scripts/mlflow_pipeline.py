#!/usr/bin/env python3
"""MLflow CI/CD pipeline for fashion price prediction model.

Runs: data generation (if needed) -> train -> evaluate -> register/promote.
Usage:
  python scripts/mlflow_pipeline.py [--regenerate-data] [--mlflow-tracking-uri URI]
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], cwd: Path = ROOT) -> None:
    """Run subprocess; exit on non-zero return."""
    result = subprocess.run(cmd, cwd=str(cwd), capture_output=False)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(description="MLflow model training pipeline")
    parser.add_argument("--regenerate-data", action="store_true", help="Regenerate 3000+ samples before training")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking server URI")
    parser.add_argument("--skip-feast", action="store_true", help="Skip Feast materialization")
    args = parser.parse_args()

    if args.mlflow_tracking_uri:
        import os
        os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_tracking_uri

    if args.regenerate_data:
        print("--- Regenerating training data (3200 samples) ---")
        run([sys.executable, str(ROOT / "scripts" / "generate_training_data.py")])

    print("--- Training model (MLflow tracked) ---")
    run([sys.executable, str(ROOT / "scripts" / "train.py")])

    if not args.skip_feast:
        print("--- Refreshing Feast feature store ---")
        run([sys.executable, str(ROOT / "scripts" / "generate_feast_data.py")])
        feat_repo = ROOT / "feature_repo" / "feature_repo"
        run(["feast", "apply"], cwd=feat_repo)
        # Materialize using parquet's event_timestamp range so new products are included
        parquet_path = feat_repo / "data" / "fashion_price_features.parquet"
        if parquet_path.exists():
            pf = pd.read_parquet(parquet_path, columns=["event_timestamp"])
            ts_min = pd.to_datetime(pf["event_timestamp"]).min()
            ts_max = pd.to_datetime(pf["event_timestamp"]).max()
            start_date = ts_min.strftime("%Y-%m-%d")
            end_date = (ts_max + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            end = pd.Timestamp.now()
            start = end - pd.Timedelta(days=90)
            start_date, end_date = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
        run(["feast", "materialize", start_date, end_date], cwd=feat_repo)

    print("--- Pipeline complete ---")


if __name__ == "__main__":
    main()
