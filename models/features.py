"""Feature engineering for fashion price change prediction.

Builds numeric features from product metadata and 7-day price history:
- Price stats (min, max, mean, std, change, volatility)
- Count of price drops in window
- Categorical encodings (category, brand, subcategory, color, season, region, age_group)
- Inventory level
"""

import numpy as np
import pandas as pd


# Fallback mappings when training mappings unavailable; used for API inference on new products
DEFAULT_CATEGORY_MAPPINGS = {
    "category": {"Outerwear": 0, "Tops": 1, "Footwear": 2, "Bottoms": 3, "Dresses": 4},
    "brand": {"Zara": 0, "H&M": 1, "Steve Madden": 2, "Levi's": 3, "Uniqlo": 4, "Converse": 5, "North Face": 6, "Mango": 7, "Massimo Dutti": 8, "Nike": 9, "Adidas": 10, "Gap": 11},
    "subcategory": {"Coats": 0, "T-Shirts": 1, "Boots": 2, "Jeans": 3, "Dresses": 4, "Sweaters": 5, "Sneakers": 6, "Jackets": 7, "Tops": 8, "Pants": 9, "Blouses": 10, "Hoodies": 11, "Shorts": 12, "Skirts": 13, "Casual": 14, "Evening": 15, "Maxi": 16, "Midi": 17, "Blazers": 18, "Vests": 19, "Sandals": 20, "Loafers": 21},
    "color": {"Black": 0, "White": 1, "Brown": 2, "Blue": 3, "Pink": 4, "Gray": 5, "Cream": 6, "Navy": 7, "Red": 8, "Green": 9},
    "season": {"Fall": 0, "All": 1, "Spring": 2, "Winter": 3, "Summer": 4},
    "region": {"North": 0, "South": 1, "East": 2, "West": 3, "International": 4},
    "age_group": {"18-24": 0, "25-34": 1, "35-44": 2, "45-54": 3, "55+": 4},
}

CATEGORICAL_COLS = ["category", "brand", "subcategory", "color", "season", "region", "age_group"]


def fill_missing_features(df: pd.DataFrame, feature_cols: list[str], random_state: int = 42) -> pd.DataFrame:
    """Fill NaN feature values with random values in [col_min, col_max].
    Uses column's observed range for plausible imputation; reproducible via random_state.
    """
    rng = np.random.default_rng(random_state)
    out = df.copy()
    for col in feature_cols:
        if col not in out.columns:
            continue
        mask = out[col].isna()
        if not mask.any():
            continue
        valid = out.loc[~mask, col]
        if len(valid) == 0:
            out[col] = out[col].fillna(0)
            continue
        vmin, vmax = valid.min(), valid.max()
        if vmin == vmax:
            out.loc[mask, col] = vmin
        else:
            n_missing = mask.sum()
            out.loc[mask, col] = rng.uniform(vmin, vmax, size=n_missing)
    return out


def _encode_categoricals(df: pd.DataFrame, mappings: dict | None = None) -> pd.DataFrame:
    """Encode categoricals to integer codes. Uses mappings if provided; else DEFAULT_CATEGORY_MAPPINGS
    or pandas category codes. Unmapped values get -1.
    """
    mappings = mappings or {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        cat_map = mappings.get(col) or DEFAULT_CATEGORY_MAPPINGS.get(col)
        if cat_map:
            df[f"{col}_cat"] = df[col].map(cat_map).fillna(-1).astype(int)
        else:
            df[f"{col}_cat"] = df[col].astype("category").cat.codes
    return df


def build_features_single(
    products: pd.DataFrame,
    price_history: pd.DataFrame,
    category_mappings: dict | None = None,
) -> pd.DataFrame:
    """
    Build features for one or more products. Supports optional category mappings
    for consistent encoding when predicting on new products.
    """
    price_history = price_history.copy()
    price_history["date"] = pd.to_datetime(price_history["date"])
    price_history = price_history.sort_values(["product_id", "date"])

    # Aggregate price stats per product over the price history window
    agg = (
        price_history.groupby("product_id")["price_usd"]
        .agg(["min", "max", "mean", "std", "first", "last", "count"])
        .reset_index()
    )
    agg.columns = [
        "product_id", "price_min_7d", "price_max_7d", "price_mean_7d", "price_std_7d",
        "price_first_day", "price_last_day", "price_records_count",
    ]
    agg["price_change_7d"] = agg["price_last_day"] - agg["price_first_day"]
    agg["price_change_pct_7d"] = (agg["price_change_7d"] / agg["price_first_day"] * 100).fillna(0)
    agg["price_volatility_7d"] = agg["price_std_7d"].fillna(0)
    agg["price_dropped"] = (agg["price_change_7d"] < 0).astype(int)

    def count_drops(g):
        diff = g["price_usd"].diff()
        return (diff < 0).sum()

    drops = price_history.groupby("product_id").apply(count_drops, include_groups=False).reset_index()
    drops.columns = ["product_id", "num_price_drops_7d"]
    agg = agg.merge(drops, on="product_id", how="left")

    df = products.merge(agg, on="product_id", how="inner")
    df = _encode_categoricals(df, category_mappings)
    return df


def build_features(products: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from product data and 7-day price history.

    Returns a DataFrame with one row per product and columns suitable for
    price change prediction.
    """
    price_history["date"] = pd.to_datetime(price_history["date"])

    # Aggregate price history per product
    agg = (
        price_history.groupby("product_id")["price_usd"]
        .agg(["min", "max", "mean", "std", "first", "last", "count"])
        .reset_index()
    )
    agg.columns = [
        "product_id",
        "price_min_7d",
        "price_max_7d",
        "price_mean_7d",
        "price_std_7d",
        "price_first_day",
        "price_last_day",
        "price_records_count",
    ]

    # Price change features
    agg["price_change_7d"] = agg["price_last_day"] - agg["price_first_day"]
    agg["price_change_pct_7d"] = (
        agg["price_change_7d"] / agg["price_first_day"] * 100
    ).fillna(0)
    agg["price_volatility_7d"] = agg["price_std_7d"].fillna(0)
    agg["price_dropped"] = (agg["price_change_7d"] < 0).astype(int)

    # Count days where price decreased from previous day
    def count_drops(g):
        diff = g["price_usd"].diff()
        return (diff < 0).sum()

    drops = price_history.groupby("product_id").apply(count_drops, include_groups=False).reset_index()
    drops.columns = ["product_id", "num_price_drops_7d"]
    agg = agg.merge(drops, on="product_id", how="left")

    # Merge with product attributes
    df = products.merge(agg, on="product_id", how="inner")
    df = _encode_categoricals(df)
    return df
