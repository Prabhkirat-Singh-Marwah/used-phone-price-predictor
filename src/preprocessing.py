"""
preprocessing.py
----------------
Builds a reusable sklearn ColumnTransformer pipeline that:
  - Standardises strings (strip + title-case).
  - Caps battery_health_pct at 100 (outlier rule).
  - Imputes missing values (median for numeric, most-frequent for categorical).
  - Ordinally encodes screen_condition and body_condition.
  - One-hot encodes brand and model (handle_unknown='ignore').
  - Standard-scales all numeric features.

The pipeline is fit ONLY on training data; the same fitted object is used for
the test set and the live app, ensuring zero train/serve mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OrdinalEncoder,
    OneHotEncoder,
    StandardScaler,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ALL_FEATURES,
    BODY_CONDITION_ORDER,
    CURRENT_YEAR,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DATA_PATH,
    SCREEN_CONDITION_ORDER,
    TARGET,
    TEST_PATH,
    TEST_SIZE,
    TRAIN_PATH,
)


# -- String cleaning (done before the sklearn pipeline) --------------------

def clean_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace and title-case all object columns."""
    df = df.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.title()
    # Re-map title-cased ordinal values to their canonical lowercase forms
    screen_map = {"Flawless": "flawless", "Minor_Scratches": "minor_scratches",
                  "Cracked": "cracked", "Minor Scratches": "minor_scratches"}
    body_map   = {"Like_New": "like_new", "Good": "good", "Fair": "fair",
                  "Poor": "poor", "Like New": "like_new"}
    df["screen_condition"] = df["screen_condition"].replace(screen_map)
    df["body_condition"]   = df["body_condition"].replace(body_map)
    return df


# -- Outlier handling -------------------------------------------------------

def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap or drop outliers using IQR rules.

    Strategy (justified):
    - battery_health_pct > 100: physically impossible -> cap to 100.
    - price_inr: IQR fence with k=3 (permissive) -> drop rows outside fence.
      Phones with absurd prices (injected outliers) are removed here.
    """
    df = df.copy()

    # Battery cap
    df["battery_health_pct"] = df["battery_health_pct"].clip(upper=100)

    # Price IQR drop (k=3 = very permissive, only removes the most egregious)
    q1 = df[TARGET].quantile(0.25)
    q3 = df[TARGET].quantile(0.75)
    iqr = q3 - q1
    lower = max(1_000, q1 - 3 * iqr)   # never drop legitimate budget phones
    upper = q3 + 3 * iqr
    before = len(df)
    df = df[(df[TARGET] >= lower) & (df[TARGET] <= upper)]
    print(f"[preprocessing] Outlier removal: {before - len(df)} rows dropped "
          f"(IQR k=3, price outside [{lower:,.0f}, {upper:,.0f}])")
    return df


# -- sklearn pipeline builder -----------------------------------------------

def build_feature_pipeline() -> ColumnTransformer:
    """
    Return an unfitted ColumnTransformer.

    Sub-pipelines:
    - numeric:  SimpleImputer(median) -> StandardScaler
    - ordinal:  SimpleImputer(most_frequent) -> OrdinalEncoder(explicit order)
    - nominal:  SimpleImputer(most_frequent) -> OneHotEncoder(handle_unknown='ignore')
    """

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    ordinal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[SCREEN_CONDITION_ORDER, BODY_CONDITION_ORDER],
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )),
    ])

    nominal_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipe, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipe, ORDINAL_FEATURES),
            ("nominal", nominal_pipe, NOMINAL_FEATURES),
        ],
        remainder="drop",
    )
    return transformer


# -- Load + prepare dataset -------------------------------------------------

def load_and_prepare(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load raw CSV, clean strings, drop duplicates, remove outliers."""
    df = pd.read_csv(path)
    print(f"[preprocessing] Loaded {len(df):,} rows, {df.shape[1]} columns")

    df = clean_strings(df)
    before = len(df)
    df = df.drop_duplicates()
    print(f"[preprocessing] Dropped {before - len(df)} duplicate rows")

    df = remove_outliers(df)

    # Compute age_years from release_year (in case it was re-derived needed)
    df["age_years"] = CURRENT_YEAR - df["release_year"]

    # Drop rows with missing target
    df = df.dropna(subset=[TARGET])

    return df.reset_index(drop=True)


# -- Train / test split + save ----------------------------------------------

def split_and_save(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified-ish split (shuffle only; price is continuous)."""
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH,  index=False)
    print(f"[preprocessing] Train: {len(train_df):,}  Test: {len(test_df):,}")
    return train_df, test_df


if __name__ == "__main__":
    df = load_and_prepare()
    split_and_save(df)
