"""Price change prediction model for fashion items."""

import pickle
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV


FEATURE_COLS = [
    "price_min_7d",
    "price_max_7d",
    "price_mean_7d",
    "price_std_7d",
    "price_change_7d",
    "price_change_pct_7d",
    "price_volatility_7d",
    "num_price_drops_7d",
    "original_price_usd",
    "category_cat",
    "brand_cat",
    "subcategory_cat",
    "color_cat",
    "season_cat",
    # Dynamic feature provided at prediction time
    "inventory_level",
]

TARGET_COL = "price_dropped"


class PriceChangeModel:
    """Scikit-learn model to predict whether a fashion item's price will drop."""

    def __init__(self, model=None):
        self.model = model or GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.feature_cols = FEATURE_COLS
        self.target_col = TARGET_COL

    def _get_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract feature columns, filling missing with 0."""
        cols = [c for c in self.feature_cols if c in df.columns]
        return df[cols].fillna(0)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model."""
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Predict price drop (0 or 1)."""
        Xf = self._get_features(X)
        return pd.Series(self.model.predict(Xf), index=X.index)

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict probability of price drop."""
        Xf = self._get_features(X)
        proba = self.model.predict_proba(Xf)
        return pd.DataFrame(
            proba,
            index=X.index,
            columns=["prob_no_drop", "prob_drop"],
        )

    def fit_with_grid_search(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[dict] = None,
        cv: int = 3,
    ):
        """Train with cross-validation grid search."""
        param_grid = param_grid or {
            "n_estimators": [50, 100],
            "max_depth": [3, 4],
            "learning_rate": [0.05, 0.1],
        }
        gs = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=cv,
            scoring="f1",
        )
        gs.fit(X, y)
        self.model = gs.best_estimator_
        return self

    def save(self, path: str | Path):
        """Save model and metadata to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_cols": self.feature_cols},
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "PriceChangeModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(model=data["model"])
        obj.feature_cols = data.get("feature_cols", FEATURE_COLS)
        return obj
