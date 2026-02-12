#!/usr/bin/env python3
"""Check model drift against baseline using Evidently AI. Logs to MLflow.

CLI: loads current data, builds features, compares to drift_reference.parquet.
Exit code 1 if drift detected, 0 otherwise. Logs share_of_drifting_columns to MLflow.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from models.drift_monitor import compute_drift, load_baseline
from models.features import build_features
from models.price_change_model import FEATURE_COLS


def main():
    # Load current data and build features (same pipeline as API drift endpoint)
    products = pd.read_csv(ROOT / "data" / "products.csv")
    price_history = pd.read_csv(ROOT / "data" / "price_history.csv")
    if "region" not in products.columns:
        products["region"] = "North"
    if "age_group" not in products.columns:
        products["age_group"] = "25-34"
    try:
        inventory = pd.read_csv(ROOT / "data" / "inventory.csv")
        df = build_features(products, price_history)
        df = df.merge(inventory[["product_id", "inventory_level"]], on="product_id", how="left")
        df["inventory_level"] = df["inventory_level"].fillna(50)
    except FileNotFoundError:
        df = build_features(products, price_history)
        df["inventory_level"] = 50

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)

    report = compute_drift(X)
    print(f"Drift: {'DETECTED' if report['drift_detected'] else 'None'}")
    share = report.get("share_of_drifting_columns")
    n_drifted = report.get("number_drifted_columns")
    print(f"Share of drifting columns: {share}" + (f" ({n_drifted} drifted)" if n_drifted is not None else ""))
    print(report["summary"])

    # Optionally log drift metrics to MLflow for monitoring
    try:
        import mlflow
        mlflow.set_experiment("fashion_price_prediction")
        with mlflow.start_run(run_name="drift_check"):
            mlflow.log_metric("share_of_drifting_columns", report.get("share_of_drifting_columns") or 0)
            mlflow.log_metric("drift_detected", 1 if report["drift_detected"] else 0)
            mlflow.log_param("n_samples", report.get("n_current_samples", 0))
    except Exception:
        pass

    return 1 if report["drift_detected"] else 0


if __name__ == "__main__":
    sys.exit(main())
