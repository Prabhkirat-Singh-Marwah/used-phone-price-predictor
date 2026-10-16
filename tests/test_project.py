"""
tests/test_project.py
---------------------
Pytest tests covering:
  1. Data generator output schema (correct columns, row count, dtypes).
  2. No target leakage -- price_inr must not appear in ALL_FEATURES.
  3. predict.py returns a positive integer for a valid spec.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

# Allow imports from src/
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import ALL_FEATURES, TARGET, N_ROWS, RAW_DATA_PATH, MODEL_PATH
from data_generator import generate


# -- Fixtures ---------------------------------------------------------------

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Generate a small dataset (300 rows) for fast tests."""
    return generate(n_rows=300, save=False)


# -- Test 1: Schema ---------------------------------------------------------

EXPECTED_COLUMNS = {
    "brand", "model", "release_year", "age_years", "storage_gb", "ram_gb",
    "battery_health_pct", "screen_condition", "body_condition", "has_box_bill",
    "warranty_months_left", "original_price_inr", "market_demand_index", "price_inr",
}

def test_data_schema_columns(raw_df: pd.DataFrame) -> None:
    """All expected columns must be present."""
    assert EXPECTED_COLUMNS.issubset(set(raw_df.columns)), (
        f"Missing columns: {EXPECTED_COLUMNS - set(raw_df.columns)}"
    )


def test_data_row_count(raw_df: pd.DataFrame) -> None:
    """Row count must be at least n_rows (duplicates are injected, so may be slightly more)."""
    assert len(raw_df) >= 300


def test_data_dtypes(raw_df: pd.DataFrame) -> None:
    """Numeric columns must be numeric; string columns must be object."""
    numeric_cols = ["age_years", "storage_gb", "ram_gb", "battery_health_pct",
                    "warranty_months_left", "original_price_inr",
                    "market_demand_index", "price_inr", "release_year"]
    for col in numeric_cols:
        assert pd.api.types.is_numeric_dtype(raw_df[col]), (
            f"{col} should be numeric, got {raw_df[col].dtype}"
        )

    string_cols = ["brand", "model", "screen_condition", "body_condition"]
    for col in string_cols:
        assert raw_df[col].dtype == object, (
            f"{col} should be object/str dtype, got {raw_df[col].dtype}"
        )


def test_price_is_positive(raw_df: pd.DataFrame) -> None:
    """Non-outlier prices should all be positive."""
    # After dropping injected absurd outliers (which may be > 5M) we check floor
    reasonable = raw_df[raw_df["price_inr"] < 5_000_000]["price_inr"]
    assert (reasonable > 0).all(), "Some non-outlier prices are non-positive"


def test_has_box_bill_binary(raw_df: pd.DataFrame) -> None:
    """has_box_bill must only contain 0 or 1."""
    assert set(raw_df["has_box_bill"].dropna().unique()).issubset({0, 1})


def test_warranty_range(raw_df: pd.DataFrame) -> None:
    """Warranty months must be in [0, 12] for non-outlier rows."""
    assert raw_df["warranty_months_left"].between(0, 12).all()


# -- Test 2: No target leakage ---------------------------------------------

def test_no_target_leakage() -> None:
    """
    TARGET ('price_inr') must not appear in ALL_FEATURES.
    price_retention_ratio (derived from price) must also be absent.
    """
    assert TARGET not in ALL_FEATURES, (
        f"Target '{TARGET}' found in ALL_FEATURES -- this is data leakage!"
    )
    leaky = [f for f in ALL_FEATURES if "price_retention" in f]
    assert not leaky, (
        f"Leaky feature(s) in ALL_FEATURES: {leaky}"
    )


def test_release_year_not_in_features() -> None:
    """
    release_year should not be a direct feature because age_years already
    encodes the same information and is more interpretable.
    """
    assert "release_year" not in ALL_FEATURES, (
        "release_year is in ALL_FEATURES -- use age_years instead to avoid "
        "indirect leakage and redundancy."
    )


# -- Test 3: predict.py returns a positive number --------------------------

@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet. Run `python src/train.py` first.",
)
def test_predict_returns_positive() -> None:
    """predict_price() must return a positive integer for a valid spec."""
    from predict import load_model, predict_price

    model = load_model()
    spec = {
        "brand":                "Apple",
        "model":                "iPhone 13",
        "release_year":         2021,
        "age_years":            3.0,
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
    assert isinstance(price, int), f"Expected int, got {type(price)}"
    assert price > 0, f"Predicted price {price} is not positive"


@pytest.mark.skipif(
    not MODEL_PATH.exists(),
    reason="Model not trained yet. Run `python src/train.py` first.",
)
def test_predict_budget_phone_lower_than_flagship() -> None:
    """A budget Redmi should predict lower than an Apple iPhone 15."""
    from predict import load_model, predict_price

    model = load_model()
    flagship_spec = {
        "brand":                "Apple",
        "model":                "iPhone 15",
        "release_year":         2023,
        "age_years":            1.0,
        "storage_gb":           256,
        "ram_gb":               6,
        "battery_health_pct":   95.0,
        "screen_condition":     "flawless",
        "body_condition":       "like_new",
        "has_box_bill":         1,
        "warranty_months_left": 10,
        "original_price_inr":   79900,
        "market_demand_index":  9.0,
    }
    budget_spec = {
        "brand":                "Xiaomi",
        "model":                "Redmi 12C",
        "release_year":         2023,
        "age_years":            1.0,
        "storage_gb":           64,
        "ram_gb":               4,
        "battery_health_pct":   85.0,
        "screen_condition":     "minor_scratches",
        "body_condition":       "good",
        "has_box_bill":         0,
        "warranty_months_left": 0,
        "original_price_inr":   10999,
        "market_demand_index":  4.0,
    }
    flagship_price = predict_price(model, flagship_spec)
    budget_price   = predict_price(model, budget_spec)
    assert flagship_price > budget_price, (
        f"Flagship {flagship_price} should be > budget {budget_price}"
    )
