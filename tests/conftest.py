"""Pytest fixtures for API tests.

Provides: client (FastAPI TestClient), sample_product_payload (valid predict body).
Requires model artifacts from train.py; some tests may skip if Feast/model missing.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def client():
    """FastAPI test client. Requires model artifacts to exist (run train.py first)."""
    from api.main import app
    return TestClient(app)


@pytest.fixture
def sample_product_payload():
    """Valid product payload for /api/predict."""
    return {
        "product_id": "P_TEST_001",
        "name": "Test Jacket",
        "category": "Outerwear",
        "brand": "Zara",
        "subcategory": "Jackets",
        "color": "Black",
        "original_price_usd": 150.0,
        "season": "Winter",
        "region": "North",
        "age_group": "25-34",
        "inventory_level": 50,
        "price_history": [
            {"date": "2025-02-02", "price_usd": 150},
            {"date": "2025-02-03", "price_usd": 150},
            {"date": "2025-02-04", "price_usd": 140},
            {"date": "2025-02-05", "price_usd": 140},
            {"date": "2025-02-06", "price_usd": 130},
            {"date": "2025-02-07", "price_usd": 130},
            {"date": "2025-02-08", "price_usd": 120},
        ],
    }
