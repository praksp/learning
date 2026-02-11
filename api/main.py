"""REST API for fashion price change prediction."""

from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project root
ROOT = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT))

from models.features import build_features_single
from models.price_change_model import PriceChangeModel


# --- Pydantic models ---

class PriceDay(BaseModel):
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    price_usd: float = Field(..., ge=0)


class ProductInput(BaseModel):
    product_id: str = Field(..., description="Unique product identifier")
    name: str
    category: str = Field(..., description="e.g. Tops, Bottoms, Footwear")
    brand: str
    subcategory: str
    color: str
    original_price_usd: float = Field(..., ge=0)
    season: str = Field(..., description="e.g. All, Fall, Spring, Winter")
    region: str = Field("North", description="e.g. North, South, East, West, International")
    age_group: str = Field("25-34", description="e.g. 18-24, 25-34, 35-44, 45-54, 55+")
    inventory_level: int = Field(..., ge=1, le=100, description="Inventory level from 1 (low) to 100 (high)")
    price_history: list[PriceDay] = Field(
        ...,
        min_length=1,
        max_length=14,
        description="1–14 days of price history (7 recommended)",
    )


class PredictionResponse(BaseModel):
    product_id: str
    predicted_drop: int = Field(..., description="0 = no drop, 1 = likely drop")
    prob_price_drop: float = Field(..., ge=0, le=1)
    price_change_pct_7d: float | None
    recommendation: str = Field(..., description="Human-readable recommendation")
    # Inventory-aware dynamic pricing suggestion
    recommended_price: float | None = Field(
        default=None,
        description="Suggested price based on inventory level and price-drop risk",
    )
    recommended_discount_pct: float | None = Field(
        default=None,
        description="Suggested discount percentage versus current price",
    )


# --- Model loader ---

_model: PriceChangeModel | None = None
_category_mappings: dict | None = None


def load_model():
    global _model, _category_mappings
    model_path = ROOT / "artifacts" / "price_change_model.pkl"
    mappings_path = ROOT / "artifacts" / "category_mappings.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run scripts/train.py first.")
    _model = PriceChangeModel.load(model_path)
    if mappings_path.exists():
        import json
        with open(mappings_path) as f:
            _category_mappings = json.load(f)
    else:
        _category_mappings = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    # cleanup if needed


# --- App ---

app = FastAPI(
    title="Fashion Price Prediction API",
    description="Predict whether fashion item prices will drop. Use for recommendation scoring.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/api/model/metrics")
def get_model_metrics():
    """Return model F1 and accuracy from last training run."""
    metrics_path = ROOT / "artifacts" / "model_metrics.json"
    if not metrics_path.exists():
        return {"f1": None, "accuracy": None, "n_samples": None}
    import json
    with open(metrics_path) as f:
        return json.load(f)


def _run_retrain():
    """Retrain model, refresh Feast, reload in-memory model. Returns metrics."""
    import subprocess
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py")],
        cwd=str(ROOT),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "generate_feast_data.py")],
        cwd=str(ROOT),
        capture_output=True,
        check=True,
    )
    feat_repo = ROOT / "feature_repo" / "feature_repo"
    subprocess.run(["feast", "apply"], cwd=str(feat_repo), capture_output=True, check=True)
    subprocess.run(
        ["feast", "materialize", "2025-02-01", "2025-02-09"],
        cwd=str(feat_repo),
        capture_output=True,
        check=True,
    )
    load_model()
    return get_model_metrics()


@app.post("/api/retrain")
def retrain():
    """Retrain model from current data files. Refreshes Feast. Returns metrics."""
    try:
        metrics = _run_retrain()
        return {"status": "ok", **metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/products")
def add_product_and_retrain(product: ProductInput):
    """Add product to data files, retrain model, refresh Feast. Returns metrics."""
    try:
        products_path = ROOT / "data" / "products.csv"
        price_path = ROOT / "data" / "price_history.csv"
        inv_path = ROOT / "data" / "inventory.csv"

        # Append to products
        new_row = pd.DataFrame([{
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "brand": product.brand,
            "subcategory": product.subcategory,
            "color": product.color,
            "original_price_usd": product.original_price_usd,
            "season": product.season,
            "region": getattr(product, "region", "North"),
            "age_group": getattr(product, "age_group", "25-34"),
        }])
        pd.concat([pd.read_csv(products_path), new_row], ignore_index=True).to_csv(products_path, index=False)

        # Append to price_history
        price_rows = pd.DataFrame([
            {"product_id": product.product_id, "date": p.date, "price_usd": p.price_usd}
            for p in product.price_history
        ])
        pd.concat([pd.read_csv(price_path), price_rows], ignore_index=True).to_csv(price_path, index=False)

        # Append to inventory
        inv_row = pd.DataFrame([{"product_id": product.product_id, "inventory_level": product.inventory_level}])
        pd.concat([pd.read_csv(inv_path), inv_row], ignore_index=True).to_csv(inv_path, index=False)

        metrics = _run_retrain()
        return {"status": "ok", "product_id": product.product_id, **metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


FEATURE_NAMES = [
    "price_min_7d", "price_max_7d", "price_mean_7d", "price_std_7d",
    "price_change_7d", "price_change_pct_7d", "price_volatility_7d",
    "num_price_drops_7d", "original_price_usd",
    "category_cat", "brand_cat", "subcategory_cat", "color_cat", "season_cat",
    "region_cat", "age_group_cat", "inventory_level", "price_dropped",
]


@app.get("/api/features")
def list_all_features():
    """List all products with their features from the Feast feature store."""
    try:
        products_df = pd.read_csv(ROOT / "data" / "products.csv")
        product_ids = products_df["product_id"].tolist()
        if not product_ids:
            return {"products": [], "feature_names": []}

        from feast import FeatureStore
        store = FeatureStore(repo_path=str(ROOT / "feature_repo" / "feature_repo"))
        feature_refs = [f"fashion_price_features:{f}" for f in FEATURE_NAMES]
        entity_rows = [{"product_id": pid} for pid in product_ids]
        result = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_dict()

        # Build list of {product_id, ...features}
        rows = []
        for i, pid in enumerate(result["product_id"]):
            row = {"product_id": pid}
            for name in FEATURE_NAMES:
                val = result.get(name)
                if val is not None:
                    v = val[i] if isinstance(val, list) else val
                    row[name] = round(float(v), 4) if isinstance(v, (int, float)) and not isinstance(v, bool) else v
            rows.append(row)
        return {"products": rows, "feature_names": FEATURE_NAMES}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/features/{product_id}")
def get_features(product_id: str):
    """Fetch features for a single product from the Feast feature store."""
    try:
        from feast import FeatureStore
        store = FeatureStore(repo_path=str(ROOT / "feature_repo" / "feature_repo"))
        feature_refs = [f"fashion_price_features:{f}" for f in FEATURE_NAMES]
        result = store.get_online_features(
            features=feature_refs,
            entity_rows=[{"product_id": product_id}],
        ).to_dict()
        return {k: v[0] if v else None for k, v in result.items()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict", response_model=PredictionResponse)
def predict(product: ProductInput):
    """Predict price drop likelihood for a fashion item with its 7-day price history."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Build DataFrames for feature engineering
    products_df = pd.DataFrame([{
        "product_id": product.product_id,
        "name": product.name,
        "category": product.category,
        "brand": product.brand,
        "subcategory": product.subcategory,
        "color": product.color,
        "original_price_usd": product.original_price_usd,
        "season": product.season,
        "region": product.region,
        "age_group": product.age_group,
    }])
    price_df = pd.DataFrame([
        {"product_id": product.product_id, "date": p.date, "price_usd": p.price_usd}
        for p in product.price_history
    ])

    try:
        features_df = build_features_single(products_df, price_df, _category_mappings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Inject current inventory level as a dynamic feature
    features_df["inventory_level"] = product.inventory_level

    proba = _model.predict_proba(features_df)
    pred = _model.predict(features_df)

    prob_drop = float(proba["prob_drop"].iloc[0])
    pred_drop = int(pred.iloc[0])
    change_pct = features_df["price_change_pct_7d"].iloc[0] if "price_change_pct_7d" in features_df.columns else None

    # Simple inventory-aware dynamic pricing heuristic
    # Base on last observed price when available, otherwise original price
    base_price = (
        float(features_df.get("price_last_day", pd.Series([product.original_price_usd])).iloc[0])
        if "price_last_day" in features_df.columns
        else float(product.original_price_usd)
    )
    inv_norm = product.inventory_level / 100.0  # 0–1
    # Higher inventory and higher drop probability → larger discount
    raw_discount = 0.5 * prob_drop + 0.4 * inv_norm
    discount = max(0.0, min(raw_discount, 0.5))  # cap at 50%
    recommended_price = round(base_price * (1 - discount), 2)
    recommended_discount_pct = round(discount * 100, 1)

    if prob_drop >= 0.7:
        rec = "High likelihood of price drop — consider waiting or highlighting for deal seekers"
    elif prob_drop >= 0.4:
        rec = "Moderate chance of price drop — monitor"
    else:
        rec = "Low likelihood of price drop — stable price expected"

    return PredictionResponse(
        product_id=product.product_id,
        predicted_drop=pred_drop,
        prob_price_drop=round(prob_drop, 4),
        price_change_pct_7d=round(change_pct, 2) if change_pct is not None and not pd.isna(change_pct) else None,
        recommendation=rec,
        recommended_price=recommended_price,
        recommended_discount_pct=recommended_discount_pct,
    )
