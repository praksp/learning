"""Functional tests for the Fashion Price Prediction API.

Covers: health, predict (valid/invalid payloads), model metrics, features, drift.
Run: pytest tests/ -v. Requires model artifacts (train.py) for predict/features tests.
"""

import pytest


class TestHealth:
    """Tests for /api/health."""

    def test_health_returns_200(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200

    def test_health_has_status_and_model_loaded(self, client):
        r = client.get("/api/health")
        data = r.json()
        assert "status" in data
        assert data["status"] == "ok"
        assert "model_loaded" in data
        assert isinstance(data["model_loaded"], bool)


class TestPredict:
    """Tests for /api/predict."""

    def test_predict_returns_200_with_valid_payload(self, client, sample_product_payload):
        r = client.post("/api/predict", json=sample_product_payload)
        assert r.status_code == 200

    def test_predict_returns_expected_fields(self, client, sample_product_payload):
        r = client.post("/api/predict", json=sample_product_payload)
        data = r.json()
        assert "product_id" in data
        assert data["product_id"] == "P_TEST_001"
        assert "predicted_drop" in data
        assert data["predicted_drop"] in (0, 1)
        assert "prob_price_drop" in data
        assert 0 <= data["prob_price_drop"] <= 1
        assert "recommendation" in data
        assert "recommended_price" in data
        assert "recommended_discount_pct" in data

    def test_predict_rejects_missing_required_field(self, client, sample_product_payload):
        del sample_product_payload["price_history"]
        r = client.post("/api/predict", json=sample_product_payload)
        assert r.status_code == 422

    def test_predict_rejects_invalid_inventory_level(self, client, sample_product_payload):
        sample_product_payload["inventory_level"] = 150  # > 100
        r = client.post("/api/predict", json=sample_product_payload)
        assert r.status_code == 422

    def test_predict_rejects_negative_price(self, client, sample_product_payload):
        sample_product_payload["price_history"][0]["price_usd"] = -1
        r = client.post("/api/predict", json=sample_product_payload)
        assert r.status_code == 422


class TestModelMetrics:
    """Tests for /api/model/metrics."""

    def test_metrics_returns_200(self, client):
        r = client.get("/api/model/metrics")
        assert r.status_code == 200

    def test_metrics_has_f1_accuracy_n_samples(self, client):
        r = client.get("/api/model/metrics")
        data = r.json()
        assert "f1" in data
        assert "accuracy" in data
        assert "n_samples" in data
        if data["f1"] is not None:
            assert 0 <= data["f1"] <= 1
        if data["accuracy"] is not None:
            assert 0 <= data["accuracy"] <= 1


class TestFeatures:
    """Tests for /api/features."""

    def test_features_list_returns_200(self, client):
        r = client.get("/api/features")
        assert r.status_code == 200

    def test_features_list_has_products_and_feature_names(self, client):
        r = client.get("/api/features")
        data = r.json()
        assert "products" in data
        assert "feature_names" in data
        assert isinstance(data["products"], list)
        assert isinstance(data["feature_names"], list)

    def test_features_single_returns_200(self, client):
        r = client.get("/api/features/P00001")
        assert r.status_code == 200

    def test_features_single_has_feature_values(self, client):
        r = client.get("/api/features/P00001")
        data = r.json()
        assert "product_id" in data
        assert data["product_id"] == "P00001"


class TestDrift:
    """Tests for /api/drift (drift observability)."""

    def test_drift_returns_200(self, client):
        r = client.get("/api/drift")
        assert r.status_code == 200

    def test_drift_has_expected_structure(self, client):
        r = client.get("/api/drift")
        data = r.json()
        assert "drift_detected" in data
        assert "summary" in data
        assert isinstance(data["drift_detected"], bool)
        assert "share_of_drifting_columns" in data
        assert "feature_drifts" in data
