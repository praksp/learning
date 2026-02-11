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
├── models/
│   ├── features.py
│   └── price_change_model.py
├── scripts/
│   ├── train.py
│   └── predict.py
├── frontend/               # React UI
├── artifacts/
│   └── price_change_model.pkl
└── requirements.txt
```

## Usage

**1. Train the model (one-time):**

```bash
python scripts/train.py
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

## API

- `GET /api/health` — Health check
- `POST /api/predict` — Predict price drop likelihood. Body: product metadata + `price_history` (array of `{date, price_usd}`).

## Model

- **Target**: Binary — will price drop? (derived from 7-day history)
- **Features**: Price stats, product attributes (category, brand, color, season)
- **Algorithm**: Gradient Boosting Classifier (scikit-learn)
