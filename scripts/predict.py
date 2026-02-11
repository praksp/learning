"""Generate price-drop predictions for recommendation scoring."""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.features import build_features
from models.price_change_model import PriceChangeModel


def main(
    products_path: str = None,
    price_history_path: str = None,
    model_path: str = None,
    output_path: str = None,
):
    products_path = products_path or str(ROOT / "data" / "products.csv")
    price_history_path = price_history_path or str(ROOT / "data" / "price_history.csv")
    model_path = model_path or str(ROOT / "artifacts" / "price_change_model.pkl")
    output_path = output_path or str(ROOT / "data" / "predictions.csv")

    products = pd.read_csv(products_path)
    price_history = pd.read_csv(price_history_path)

    df = build_features(products, price_history)
    model = PriceChangeModel.load(model_path)

    df["prob_price_drop"] = model.predict_proba(df)["prob_drop"]
    df["predicted_drop"] = model.predict(df)

    # Sort by likelihood of price drop for recommendation (higher = better deal potential)
    df = df.sort_values("prob_price_drop", ascending=False).reset_index(drop=True)

    output_cols = [
        "product_id",
        "name",
        "category",
        "brand",
        "price_last_day",
        "price_change_pct_7d",
        "prob_price_drop",
        "predicted_drop",
    ]
    output_cols = [c for c in output_cols if c in df.columns]
    df[output_cols].to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
    print(df[output_cols].head(10).to_string(index=False))
    return df


if __name__ == "__main__":
    main()
