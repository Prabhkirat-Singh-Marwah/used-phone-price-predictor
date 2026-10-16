"""
config.py
---------
Central configuration: all paths, constants, feature lists, and the random
seed. Import from here everywhere - never hardcode paths or magic numbers
in other modules.
"""

from pathlib import Path

# -- Reproducibility --------------------------------------------------------
RANDOM_SEED: int = 42

# -- Current year used for age calculation ---------------------------------
CURRENT_YEAR: int = 2025

# -- Project root (two levels up from this file: src/ -> phone-price-predictor/)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# -- Data paths -------------------------------------------------------------
DATA_DIR:       Path = PROJECT_ROOT / "data"
RAW_DATA_PATH:  Path = DATA_DIR / "raw" / "used_phones_raw.csv"
PROCESSED_DIR:  Path = DATA_DIR / "processed"
TRAIN_PATH:     Path = PROCESSED_DIR / "train.csv"
TEST_PATH:      Path = PROCESSED_DIR / "test.csv"

# -- Model artefacts --------------------------------------------------------
MODELS_DIR:     Path = PROJECT_ROOT / "models"
MODEL_PATH:     Path = MODELS_DIR / "best_model.joblib"
METRICS_PATH:   Path = MODELS_DIR / "metrics.json"

# -- Report / figure paths --------------------------------------------------
REPORTS_DIR:    Path = PROJECT_ROOT / "reports"
FIGURES_DIR:    Path = REPORTS_DIR / "figures"

# -- Data generation settings -----------------------------------------------
N_ROWS: int = 6_000
MISSING_RATE:    float = 0.03   # 3 % missing in selected columns
DUPLICATE_RATE:  float = 0.015  # 1.5 % duplicate rows
OUTLIER_RATE:    float = 0.01   # 1 % outlier rows

# -- ML settings ------------------------------------------------------------
TEST_SIZE:   float = 0.20
CV_FOLDS:    int   = 5

# -- Feature lists (must stay in sync with data_generator output) -----------
NUMERIC_FEATURES: list[str] = [
    "age_years",
    "storage_gb",
    "ram_gb",
    "battery_health_pct",
    "warranty_months_left",
    "original_price_inr",
    "market_demand_index",
]

ORDINAL_FEATURES: list[str] = ["screen_condition", "body_condition"]

NOMINAL_FEATURES: list[str] = ["brand", "model"]

ALL_FEATURES: list[str] = NUMERIC_FEATURES + ORDINAL_FEATURES + NOMINAL_FEATURES

TARGET: str = "price_inr"

# -- Ordinal category orders (low -> high quality) --------------------------
SCREEN_CONDITION_ORDER: list[str] = ["cracked", "minor_scratches", "flawless"]
BODY_CONDITION_ORDER:   list[str] = ["poor", "fair", "good", "like_new"]

# -- Brand catalogue used in data_generator --------------------------------
BRAND_CATALOGUE: dict = {
    "Apple": {
        "models": [
            ("iPhone 11",       2019, 64900,   [64, 128, 256],     [4]),
            ("iPhone 12",       2020, 79900,   [64, 128, 256],     [4]),
            ("iPhone 13",       2021, 79900,   [128, 256, 512],    [4]),
            ("iPhone 14",       2022, 79900,   [128, 256, 512],    [6]),
            ("iPhone 15",       2023, 79900,   [128, 256, 512],    [6]),
            ("iPhone SE 2022",  2022, 43900,   [64, 128, 256],     [3]),
        ],
        "retention": 0.82,
    },
    "Samsung": {
        "models": [
            ("Galaxy S21",      2021, 69999,   [128, 256],         [8]),
            ("Galaxy S22",      2022, 72999,   [128, 256],         [8]),
            ("Galaxy S23",      2023, 74999,   [128, 256],         [8]),
            ("Galaxy A54",      2023, 38999,   [128, 256],         [8]),
            ("Galaxy M34",      2023, 18999,   [128],              [8]),
            ("Galaxy A14",      2022, 13999,   [64, 128],          [4]),
        ],
        "retention": 0.72,
    },
    "OnePlus": {
        "models": [
            ("OnePlus 11",      2023, 56999,   [128, 256],         [8, 16]),
            ("OnePlus 10 Pro",  2022, 66999,   [128, 256],         [8, 12]),
            ("OnePlus Nord 3",  2023, 33999,   [128, 256],         [8, 12]),
            ("OnePlus Nord CE3",2023, 26999,   [128],              [8]),
        ],
        "retention": 0.68,
    },
    "Xiaomi": {
        "models": [
            ("Redmi Note 12",   2023, 17999,   [64, 128],          [4, 6]),
            ("Redmi Note 11",   2022, 14999,   [64, 128],          [4, 6]),
            ("Xiaomi 13 Pro",   2023, 79999,   [256],              [12]),
            ("Redmi 12C",       2023, 10999,   [64, 128],          [4]),
            ("POCO X5 Pro",     2023, 22999,   [128, 256],         [6, 8]),
        ],
        "retention": 0.58,
    },
    "Realme": {
        "models": [
            ("Realme 11 Pro",   2023, 24999,   [128, 256],         [8]),
            ("Realme C55",      2023, 13999,   [64, 128],          [6, 8]),
            ("Realme GT 5",     2023, 34999,   [128, 256],         [8, 12]),
        ],
        "retention": 0.55,
    },
    "Vivo": {
        "models": [
            ("Vivo V27",        2023, 34999,   [128, 256],         [8]),
            ("Vivo Y56",        2023, 17999,   [128],              [8]),
            ("Vivo X90 Pro",    2023, 84999,   [256],              [12]),
        ],
        "retention": 0.60,
    },
    "Oppo": {
        "models": [
            ("Oppo Reno 10",    2023, 35999,   [128, 256],         [8]),
            ("Oppo A78",        2023, 22999,   [128],              [8]),
        ],
        "retention": 0.58,
    },
    "Google": {
        "models": [
            ("Pixel 7",         2022, 59999,   [128, 256],         [8]),
            ("Pixel 7a",        2023, 53999,   [128],              [8]),
            ("Pixel 6a",        2022, 43999,   [128],              [6]),
        ],
        "retention": 0.70,
    },
}
