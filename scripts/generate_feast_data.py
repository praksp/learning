"""Generate Parquet data for Feast feature store from products and price history."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from models.features import build_features


def main():
    products = pd.read_csv(ROOT / "data" / "products.csv")
    price_history = pd.read_csv(ROOT / "data" / "price_history.csv")
    try:
        inventory = pd.read_csv(ROOT / "data" / "inventory.csv")
    except FileNotFoundError:
        inventory = None

    df = build_features(products, price_history)

    # Attach inventory snapshot so it appears in the feature store
    if inventory is not None and "inventory_level" in inventory.columns:
        df = df.merge(inventory[["product_id", "inventory_level"]], on="product_id", how="left")
        if "inventory_level" in df.columns:
            df["inventory_level"] = df["inventory_level"].fillna(df["inventory_level"].median())
    else:
        df["inventory_level"] = 50

    # Feast requires event_timestamp (datetime) and created column
    price_history["date"] = pd.to_datetime(price_history["date"])
    last_dates = price_history.groupby("product_id")["date"].max()
    df["event_timestamp"] = df["product_id"].map(last_dates)
    df["created"] = df["event_timestamp"]

    out_dir = ROOT / "feature_repo" / "feature_repo" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fashion_price_features.parquet"

    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
