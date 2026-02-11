# Fashion Price Change Prediction

Predict price changes for fashion items to support recommendation systems (e.g., items likely to drop in price).

## Setup

```bash
pip install -r requirements.txt
npm install --prefix frontend
```

## Project Structure

```
├── api/                    # REST microservice
│   └── main.py
├── data/
│   ├── products.csv
│   ├── price_history.csv
│   └── predictions.csv
├── feature_repo/           # Feast feature store
│   └── feature_repo/
│       ├── feature_definitions.py
│       ├── feature_store.yaml
│       └── data/
│           └── fashion_price_features.parquet
├── models/
│   ├── features.py
│   └── price_change_model.py
├── scripts/
│   ├── train.py
│   ├── predict.py
│   └── generate_feast_data.py
├── frontend/               # React UI
├── artifacts/
│   └── price_change_model.pkl
└── requirements.txt
```

## Usage

**1. Generate training data (3200 samples, region/age_group/seasonality):**

```bash
python scripts/generate_training_data.py
```

**2. Train the model (MLflow-tracked):**

```bash
python scripts/train.py
```

**3. Run full MLflow CI/CD pipeline:**

```bash
python scripts/mlflow_pipeline.py [--regenerate-data] [--mlflow-tracking-uri URI]
```

**2. Start the API:**

```bash
uvicorn api.main:app --reload --port 8000
```

**3. Start the React frontend:**

```bash
cd frontend && npm run dev
```

Open http://localhost:5173. Add product data and 7 days of price history; predictions update in near real-time as you type.

## Feature Store (Feast)

Features are stored in Feast for reuse in training and serving:

```bash
# Regenerate parquet after data changes
python scripts/generate_feast_data.py

# Apply feature definitions
cd feature_repo/feature_repo && feast apply

# Materialize to online store
feast materialize 2025-02-01 2025-02-09
```

## API

- `GET /api/health` — Health check
- `POST /api/predict` — Predict price drop likelihood. Body: product metadata + `price_history` (array of `{date, price_usd}`).
- `GET /api/features/{product_id}` — Fetch features for a product from the feature store

## Model

- **Target**: Binary — will price drop? (derived from 7-day history)
- **Features**: Price stats, product attributes (category, brand, color, season)
- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
