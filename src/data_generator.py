"""
data_generator.py
-----------------
Generates ~6 000 rows of synthetic (but realistic) used-phone sales data
and saves it as data/raw/used_phones_raw.csv.

Pricing assumptions encoded here:
  1. Exponential depreciation with age, rate controlled per brand.
  2. Battery health multiplier with a sharp penalty below 80 %.
  3. Screen / body condition multiplicative discounts.
  4. Box + bill and warranty premium.
  5. Market demand index adds up to +/-8 % noise.
  6. Log-normal multiplicative noise (sigma ~ 0.06) mimics real variance.

Mess injected deliberately for the preprocessing exercise:
  - 3 % missing values in battery_health_pct, storage_gb, body_condition.
  - 1.5 % duplicate rows.
  - 1 % outlier rows (absurd prices or battery > 100).
  - Inconsistent casing / whitespace in brand and condition strings.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a script from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    BRAND_CATALOGUE,
    CURRENT_YEAR,
    DUPLICATE_RATE,
    MISSING_RATE,
    N_ROWS,
    OUTLIER_RATE,
    RANDOM_SEED,
    RAW_DATA_PATH,
)

rng = np.random.default_rng(RANDOM_SEED)
random.seed(RANDOM_SEED)


# -- Helpers ----------------------------------------------------------------

def _battery_multiplier(health: float) -> float:
    """Nonlinear battery health multiplier; sharp drop below 80 %."""
    if health >= 90:
        return 1.0
    if health >= 80:
        return 0.95 + (health - 80) * 0.005   # 0.95 - 1.00
    if health >= 70:
        return 0.80 + (health - 70) * 0.015   # 0.80 - 0.95
    return 0.65 + (health - 60) * 0.015       # 0.65 - 0.80


def _demand_index(brand: str, release_year: int, age: float) -> float:
    """Derive market demand index (1-10) from brand tier + age, plus noise."""
    brand_base = {
        "Apple": 9.0, "Samsung": 7.5, "OnePlus": 7.0,
        "Google": 6.5, "Xiaomi": 6.0, "Vivo": 5.5,
        "Oppo": 5.5, "Realme": 5.0,
    }.get(brand, 5.0)
    age_penalty = min(age * 0.5, 4.0)
    raw = brand_base - age_penalty + rng.normal(0, 0.8)
    return float(np.clip(raw, 1.0, 10.0))


def _compute_price(
    original: int,
    age: float,
    retention: float,
    battery: float,
    screen: str,
    body: str,
    has_box: int,
    warranty: int,
    demand: float,
) -> int:
    """Return a realistic resale price in INR (rounded to nearest 100)."""
    # 1. Exponential depreciation
    depreciated = original * (retention ** age)

    # 2. Battery multiplier
    price = depreciated * _battery_multiplier(battery)

    # 3. Screen condition
    screen_mult = {"flawless": 1.0, "minor_scratches": 0.94, "cracked": 0.70}
    price *= screen_mult.get(screen, 1.0)

    # 4. Body condition
    body_mult = {"like_new": 1.0, "good": 0.92, "fair": 0.82, "poor": 0.68}
    price *= body_mult.get(body, 1.0)

    # 5. Box / bill premium
    price *= (1.03 if has_box else 1.0)

    # 6. Warranty premium (0.5 % per month)
    price *= (1 + warranty * 0.005)

    # 7. Demand index shifts price +/-8 % linearly (index centred at 5.5)
    price *= (1 + (demand - 5.5) / 5.5 * 0.08)

    # 8. Log-normal noise
    price *= rng.lognormal(0, 0.06)

    # Enforce a floor (never less than 5 % of launch price)
    price = max(price, original * 0.05)

    return int(round(price / 100) * 100)


# -- Row sampler ------------------------------------------------------------

def _sample_row() -> dict:
    """Draw one row from the catalogue at random."""
    brand = random.choice(list(BRAND_CATALOGUE.keys()))
    info  = BRAND_CATALOGUE[brand]
    model_tuple = random.choice(info["models"])
    name, rel_year, launch_price, storages, rams = model_tuple
    retention = info["retention"]

    age = float(CURRENT_YEAR - rel_year)
    # Phones in catalogue aged 0-6 years; add a small jitter for mid-year sales
    age = max(0.0, age + rng.uniform(-0.5, 0.5))

    storage  = int(random.choice(storages))
    ram      = int(random.choice(rams))

    # Battery health: skewed higher for newer phones
    batt_mean = max(65, 95 - age * 4)
    battery   = float(np.clip(rng.normal(batt_mean, 5), 60, 100))

    screen   = random.choices(
        ["flawless", "minor_scratches", "cracked"],
        weights=[0.45, 0.40, 0.15],
    )[0]
    body     = random.choices(
        ["like_new", "good", "fair", "poor"],
        weights=[0.25, 0.40, 0.25, 0.10],
    )[0]
    has_box  = int(rng.binomial(1, 0.45))
    warranty = int(np.clip(rng.integers(0, 13), 0, 12))

    demand = _demand_index(brand, rel_year, age)

    price = _compute_price(
        launch_price, age, retention, battery,
        screen, body, has_box, warranty, demand,
    )

    return {
        "brand":                 brand,
        "model":                 name,
        "release_year":          rel_year,
        "age_years":             round(age, 2),
        "storage_gb":            storage,
        "ram_gb":                ram,
        "battery_health_pct":    round(battery, 1),
        "screen_condition":      screen,
        "body_condition":        body,
        "has_box_bill":          has_box,
        "warranty_months_left":  warranty,
        "original_price_inr":    launch_price,
        "market_demand_index":   round(demand, 2),
        "price_inr":             price,
    }


# -- Mess injection helpers -------------------------------------------------

def _inject_missing(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Randomly blank out a fraction of values in selected columns."""
    cols = ["battery_health_pct", "storage_gb", "body_condition"]
    for col in cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def _inject_duplicates(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Randomly duplicate a fraction of rows."""
    n_dup = int(len(df) * rate)
    dup_idx = rng.choice(df.index, size=n_dup, replace=False)
    duplicates = df.loc[dup_idx].copy()
    return pd.concat([df, duplicates], ignore_index=True)


def _inject_outliers(df: pd.DataFrame, rate: float) -> pd.DataFrame:
    """Inject obviously wrong values (for outlier-detection exercise)."""
    n_out = int(len(df) * rate)
    price_idx   = rng.choice(df.index, size=n_out // 2, replace=False)
    battery_idx = rng.choice(df.index, size=n_out // 2, replace=False)
    df.loc[price_idx,   "price_inr"]         = rng.integers(5_000_000, 9_000_000, n_out // 2)
    df.loc[battery_idx, "battery_health_pct"] = rng.uniform(105, 150, n_out // 2)
    return df


def _inject_casing_mess(df: pd.DataFrame) -> pd.DataFrame:
    """Randomly uppercase / add whitespace to string columns."""
    str_cols = ["brand", "screen_condition", "body_condition"]
    transforms = [str.upper, str.lower, str.title,
                  lambda s: "  " + s, lambda s: s + " "]
    for col in str_cols:
        mask = rng.random(len(df)) < 0.05
        df.loc[mask, col] = df.loc[mask, col].apply(
            lambda v: random.choice(transforms)(str(v)) if isinstance(v, str) else v
        )
    return df


# -- Main generator ---------------------------------------------------------

def generate(n_rows: int = N_ROWS, save: bool = True) -> pd.DataFrame:
    """Generate the synthetic dataset and optionally save it."""
    rows = [_sample_row() for _ in range(n_rows)]
    df   = pd.DataFrame(rows)

    df = _inject_missing(df, MISSING_RATE)
    df = _inject_duplicates(df, DUPLICATE_RATE)
    df = _inject_outliers(df, OUTLIER_RATE)
    df = _inject_casing_mess(df)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    if save:
        RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"[data_generator] Saved {len(df):,} rows -> {RAW_DATA_PATH}")

    return df


if __name__ == "__main__":
    generate()
