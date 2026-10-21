"""
tests/test_server.py
--------------------
Pytest suite verifying the FastAPI server and static UI file serving.
"""

import io
import sys
from pathlib import Path
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from server import app

client = TestClient(app)


def test_static_ui_served():
    """Verify that index.html, style.css, and app.js are served properly."""
    res = client.get("/")
    assert res.status_code == 200
    assert "Used Phone Price Estimator" in res.text
    assert "style.css" in res.text
    assert "app.js" in res.text

    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    assert "Inter" in res_css.text

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    assert "BRAND_CATALOGUE" in res_js.text


def test_catalogue_endpoint():
    """Verify GET /api/catalogue returns all expected phone brands."""
    res = client.get("/api/catalogue")
    assert res.status_code == 200
    data = res.json()
    assert "Apple" in data
    assert "Samsung" in data
    assert "OnePlus" in data


def test_metrics_endpoint():
    """Verify GET /api/metrics returns test metrics and optimal model."""
    res = client.get("/api/metrics")
    assert res.status_code == 200
    data = res.json()
    assert "best_model" in data
    assert "test_metrics" in data


def test_predict_endpoint():
    """Verify POST /api/predict calculates price, range, and empirical factors."""
    payload = {
        "brand": "Apple",
        "model": "iPhone 13",
        "release_year": 2021,
        "age_years": 4.0,
        "storage_gb": 128,
        "ram_gb": 4,
        "battery_health_pct": 85.0,
        "screen_condition": "minor_scratches",
        "body_condition": "good",
        "has_box_bill": 1,
        "warranty_months_left": 0,
        "original_price_inr": 79900,
        "market_demand_index": 5.5,
    }
    res = client.post("/api/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "predicted_price" in data
    assert data["predicted_price"] > 0
    assert data["low_price"] <= data["predicted_price"] <= data["high_price"]
    assert "factors" in data


def test_batch_predict_endpoint():
    """Verify POST /api/batch-predict accepts CSV and returns predictions."""
    csv_content = (
        "brand,model,release_year,storage_gb,ram_gb,battery_health_pct,screen_condition,"
        "body_condition,has_box_bill,warranty_months_left,original_price_inr,market_demand_index\n"
        "Apple,iPhone 13,2021,128,4,86.0,minor_scratches,good,1,0,79900,7.0\n"
        "Samsung,Galaxy S22,2022,128,8,89.0,flawless,like_new,1,3,72999,6.5\n"
    )
    res = client.post(
        "/api/batch-predict",
        files={"file": ("test.csv", io.BytesIO(csv_content.encode("utf-8")), "text/csv")},
    )
    assert res.status_code == 200
    records = res.json().get("records", [])
    assert len(records) == 2
    assert "predicted_price_inr" in records[0]
    assert records[0]["predicted_price_inr"] > 0
