"""Train the fashion price change prediction model."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.features import build_features
from models.price_change_model import PriceChangeModel, FEATURE_COLS, TARGET_COL


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
    try:
        inventory = pd.read_csv(inventory_path)
    except FileNotFoundError:
        inventory = None

    df = build_features(products, price_history)

    # Attach inventory snapshot for training if available
    if inventory is not None and "inventory_level" in inventory.columns:
        df = df.merge(inventory[["product_id", "inventory_level"]], on="product_id", how="left")
        if "inventory_level" in df.columns:
            df["inventory_level"] = df["inventory_level"].fillna(df["inventory_level"].median())
    else:
        # Fallback: assume medium inventory if not provided
        df["inventory_level"] = 50

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].fillna(0)
    y = df[TARGET_COL]

    model = PriceChangeModel()
    model.fit(X, y)
    model.save(output_path)

    print(f"Model trained on {len(df)} samples, saved to {output_path}")
    return model


if __name__ == "__main__":
    main()
