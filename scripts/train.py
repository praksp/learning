"""Train the fashion price change prediction model with MLflow tracking.

Pipeline: load products/price_history/inventory -> build_features -> 80/20 split ->
fit GradientBoosting -> save model, category_mappings, drift baseline, metrics.
"""

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*pickle or cloudpickle format.*")
warnings.filterwarnings("ignore", message=".*artifact_path.*deprecated.*")

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.drift_monitor import compute_baseline
from models.features import build_features, fill_missing_features, CATEGORICAL_COLS
from models.price_change_model import PriceChangeModel, FEATURE_COLS, TARGET_COL

MLFLOW_EXPERIMENT = "fashion_price_prediction"


def _build_and_save_category_mappings(df: pd.DataFrame, output_path: Path) -> None:
    """Build category->int mappings from training data for API inference on new products."""
    mappings = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        vals = df[col].dropna().unique().tolist()
        mappings[col] = {v: i for i, v in enumerate(sorted(str(x) for x in vals))}
    with open(output_path, "w") as f:
        json.dump(mappings, f, indent=2)


def main(
    products_path: str = None,
    price_history_path: str = None,
    inventory_path: str = None,
    output_path: str = None,
):
    products_path = products_path or str(ROOT / "data" / "products.csv")
    price_history_path = price_history_path or str(ROOT / "data" / "price_history.csv")
    inventory_path = inventory_path or str(ROOT / "data" / "inventory.csv")
    output_path = output_path or str(ROOT / "artifacts" / "price_change_model.pkl")

    products = pd.read_csv(products_path)
    price_history = pd.read_csv(price_history_path)
    if "region" not in products.columns:
        products["region"] = "North"
    if "age_group" not in products.columns:
        products["age_group"] = "25-34"
    try:
        inventory = pd.read_csv(inventory_path)
    except FileNotFoundError:
        inventory = None

    df = build_features(products, price_history)

    # Ensure artifacts directory exists (needed for CI/fresh clones)
    artifacts_dir = Path(output_path).parent
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save category mappings for API inference
    mappings_path = artifacts_dir / "category_mappings.json"
    _build_and_save_category_mappings(products, mappings_path)

    # Attach inventory snapshot for training if available
    if inventory is not None and "inventory_level" in inventory.columns:
        df = df.merge(inventory[["product_id", "inventory_level"]], on="product_id", how="left")
        if "inventory_level" in df.columns:
            df["inventory_level"] = df["inventory_level"].fillna(df["inventory_level"].median())
    else:
        # Fallback: assume medium inventory if not provided
        df["inventory_level"] = 50

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = fill_missing_features(df[feature_cols], feature_cols).fillna(0)
    y = df[TARGET_COL]

    # 80/20 stratified split to preserve class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = PriceChangeModel()
    params = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1}

    model.fit(X_train, y_train)
    compute_baseline(X_train)  # Save reference for drift monitoring
    y_pred_test = model.model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred_test, zero_division=0))
    accuracy = float(accuracy_score(y_test, y_pred_test))

    model.save(output_path)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="train"):
        mlflow.log_params({**params, "test_size": 0.2})
        mlflow.log_metrics({
            "f1": f1,
            "accuracy": accuracy,
            "n_train": len(X_train),
            "n_test": len(X_test),
        })
        mlflow.sklearn.log_model(model.model, name="model")
        if Path(output_path).exists():
            mlflow.log_artifact(str(output_path), "artifacts")
    metrics_path = Path(output_path).parent / "model_metrics.json"
    metrics = {"f1": f1, "accuracy": accuracy, "n_samples": len(df), "n_train": len(X_train), "n_test": len(X_test)}
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Model trained on {len(X_train)} samples, evaluated on {len(X_test)} holdout samples")
    print(f"Test F1={f1:.4f}, Test Accuracy={accuracy:.4f}")
    return model


if __name__ == "__main__":
    main()
