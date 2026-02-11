"""Generate 3000+ training samples with region, age_group, and seasonality distribution."""

import random
import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

CATEGORIES = ["Tops", "Bottoms", "Dresses", "Outerwear", "Footwear"]
SUBCATEGORIES = {
    "Tops": ["T-Shirts", "Sweaters", "Blouses", "Hoodies"],
    "Bottoms": ["Jeans", "Pants", "Shorts", "Skirts"],
    "Dresses": ["Casual", "Evening", "Maxi", "Midi"],
    "Outerwear": ["Coats", "Jackets", "Blazers", "Vests"],
    "Footwear": ["Sneakers", "Boots", "Sandals", "Loafers"],
}
BRANDS = [
    "Zara", "H&M", "Levi's", "Uniqlo", "Converse", "North Face",
    "Mango", "Massimo Dutti", "Steve Madden", "Nike", "Adidas", "Gap",
]
COLORS = ["Black", "White", "Navy", "Gray", "Blue", "Brown", "Pink", "Cream", "Red", "Green"]
REGIONS = ["North", "South", "East", "West", "International"]
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
SEASONS = ["Spring", "Summer", "Fall", "Winter", "All"]

# Seasonal weight: more Fall/Winter outerwear, more Summer dresses, etc.
SEASON_WEIGHTS = {
    "Spring": 0.2, "Summer": 0.2, "Fall": 0.25, "Winter": 0.25, "All": 0.1,
}
REGION_WEIGHTS = {"North": 0.25, "South": 0.2, "East": 0.2, "West": 0.2, "International": 0.15}
AGE_WEIGHTS = {"18-24": 0.15, "25-34": 0.3, "35-44": 0.25, "45-54": 0.2, "55+": 0.1}


def generate_products(n: int = 3200, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for i in range(n):
        pid = f"P{i+1:05d}"
        cat = random.choice(CATEGORIES)
        sub = random.choice(SUBCATEGORIES[cat])
        brand = random.choice(BRANDS)
        color = random.choice(COLORS)
        season = random.choices(SEASONS, weights=[SEASON_WEIGHTS[s] for s in SEASONS])[0]
        region = random.choices(REGIONS, weights=[REGION_WEIGHTS[r] for r in REGIONS])[0]
        age_group = random.choices(AGE_GROUPS, weights=[AGE_WEIGHTS[a] for a in AGE_GROUPS])[0]
        base_price = random.uniform(15, 250)
        original_price = round(base_price + random.uniform(-5, 15), 2)
        original_price = max(9.99, original_price)
        name = f"{brand} {sub} {color}"
        rows.append({
            "product_id": pid,
            "name": name,
            "category": cat,
            "brand": brand,
            "subcategory": sub,
            "color": color,
            "original_price_usd": original_price,
            "season": season,
            "region": region,
            "age_group": age_group,
        })
    return pd.DataFrame(rows)


def generate_price_history(products: pd.DataFrame, base_date: str = "2025-02-01", seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    rows = []
    for _, p in products.iterrows():
        start = pd.to_datetime(base_date)
        prices = [p["original_price_usd"]]
        for d in range(1, 7):
            change = random.uniform(-0.15, 0.05)
            new_price = max(5.0, prices[-1] * (1 + change))
            prices.append(round(new_price, 2))
        for d, pr in enumerate(prices):
            dt = (start + timedelta(days=d)).strftime("%Y-%m-%d")
            rows.append({"product_id": p["product_id"], "date": dt, "price_usd": pr})
    return pd.DataFrame(rows)


def generate_inventory(products: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    return pd.DataFrame([
        {"product_id": p["product_id"], "inventory_level": random.randint(5, 95)}
        for _, p in products.iterrows()
    ])


def main():
    n = 3200
    products = generate_products(n=n)
    price_history = generate_price_history(products)
    inventory = generate_inventory(products)

    data_dir = ROOT / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    products.to_csv(data_dir / "products.csv", index=False)
    price_history.to_csv(data_dir / "price_history.csv", index=False)
    inventory.to_csv(data_dir / "inventory.csv", index=False)

    print(f"Generated {len(products)} products, {len(price_history)} price records, {len(inventory)} inventory rows")
    print("Region distribution:", products["region"].value_counts().to_dict())
    print("Age group distribution:", products["age_group"].value_counts().to_dict())
    print("Season distribution:", products["season"].value_counts().to_dict())


if __name__ == "__main__":
    main()
