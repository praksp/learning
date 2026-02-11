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
    }])
    price_df = pd.DataFrame([
        {"product_id": product.product_id, "date": p.date, "price_usd": p.price_usd}
        for p in product.price_history
    ])

    try:
        features_df = build_features_single(products_df, price_df, _category_mappings)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    proba = _model.predict_proba(features_df)
    pred = _model.predict(features_df)

    prob_drop = float(proba["prob_drop"].iloc[0])
    pred_drop = int(pred.iloc[0])
    change_pct = features_df["price_change_pct_7d"].iloc[0] if "price_change_pct_7d" in features_df.columns else None

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
    )
