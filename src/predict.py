"""
predict.py
----------
Load the saved best-model pipeline and predict resale price from a
plain Python dict of phone specifications.

Usage
-----
    from src.predict import load_model, predict_price

    model = load_model()
    price = predict_price(model, {
        "brand": "Apple",
        "model": "iPhone 13",
        "release_year": 2021,
        "age_years": 3,
        "storage_gb": 128,
        "ram_gb": 4,
        "battery_health_pct": 87,
        "screen_condition": "minor_scratches",
        "body_condition": "good",
        "has_box_bill": 1,
        "warranty_months_left": 0,
        "original_price_inr": 79900,
        "market_demand_index": 7.5,
    })
    print(price)   # e.g. 42300
"""

import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ALL_FEATURES, MODEL_PATH


def load_model(path: Path = MODEL_PATH):
    """Load the serialised sklearn pipeline from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. "
            "Please run `python src/train.py` first."
        )
    return joblib.load(path)


def predict_price(model, spec: dict[str, Any]) -> int:
    """
    Return the predicted resale price (INR) for one phone.

    Parameters
    ----------
    model : fitted sklearn Pipeline (TransformedTargetRegressor)
    spec  : dict with keys matching ALL_FEATURES in config.py

    Returns
    -------
    Predicted price rounded to the nearest 100.
    """
    row = pd.DataFrame([spec])[ALL_FEATURES]
    raw = model.predict(row)[0]
    return int(round(float(raw) / 100) * 100)


def predict_price_range(
    model, spec: dict[str, Any], mape_pct: float = 15.0
) -> tuple[int, int]:
    """
    Return (low, high) price range using +/-mape_pct/2 around the point estimate.
    mape_pct is the test-set MAPE of the model (loaded from metrics.json).
    """
    point = predict_price(model, spec)
    half  = mape_pct / 2 / 100
    low   = int(round(point * (1 - half) / 100) * 100)
    high  = int(round(point * (1 + half) / 100) * 100)
    return low, high


if __name__ == "__main__":
    # Quick sanity check
    model = load_model()
    spec = {
        "brand":                "Apple",
        "model":                "iPhone 13",
        "release_year":         2021,
        "age_years":            3,
        "storage_gb":           128,
        "ram_gb":               4,
        "battery_health_pct":   87.0,
        "screen_condition":     "minor_scratches",
        "body_condition":       "good",
        "has_box_bill":         1,
        "warranty_months_left": 0,
        "original_price_inr":   79900,
        "market_demand_index":  7.5,
    }
    price = predict_price(model, spec)
    print(f"Predicted price: Rs{price:,}")
