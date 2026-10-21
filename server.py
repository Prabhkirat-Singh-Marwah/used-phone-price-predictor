"""
server.py
---------
FastAPI application serving the Used Phone Price Estimator.
Serves static UI files (HTML, CSS, JS) from the ui/ directory,
exposes REST endpoints for prediction, batch valuation, model retraining,
and performance metrics.
"""

import io
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add src to Python path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from config import (
    ALL_FEATURES,
    BRAND_CATALOGUE,
    CURRENT_YEAR,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
)
from predict import load_model, predict_price, predict_price_range
from preprocessing import clean_strings
from train import run_training_pipeline

app = FastAPI(title="Used Phone Price Estimator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory cache for model and metrics
MODEL = None
METRICS_DATA = {}


def load_app_model():
    global MODEL, METRICS_DATA
    try:
        if MODEL_PATH.exists():
            MODEL = load_model(MODEL_PATH)
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                METRICS_DATA = json.load(f)
    except Exception as e:
        print(f"Warning loading model/metrics: {e}")


load_app_model()


def fmt_inr(n: int) -> str:
    s = str(abs(n))
    if len(s) <= 3:
        return f"Rs {s}"
    last3 = s[-3:]
    rest = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    return "Rs " + ",".join(reversed(parts)) + "," + last3


def compute_factor_effects(model, base_spec: dict, base_price: int) -> list[dict]:
    factors = []

    def delta(patch: dict) -> int:
        return predict_price(model, {**base_spec, **patch}) - base_price

    batt = base_spec["battery_health_pct"]
    if batt < 90:
        d = delta({"battery_health_pct": 95.0})
        factors.append({
            "label": f"Battery degradation ({batt:.0f}%)",
            "effect": f"-{fmt_inr(abs(d))} vs 95% battery",
        })

    sc = base_spec["screen_condition"]
    if sc != "flawless":
        d = delta({"screen_condition": "flawless"})
        factors.append({
            "label": f"Screen ({sc.replace('_', ' ')})",
            "effect": f"-{fmt_inr(abs(d))} vs flawless",
        })

    bc = base_spec["body_condition"]
    if bc != "like_new":
        d = delta({"body_condition": "like_new"})
        factors.append({
            "label": f"Body wear ({bc.replace('_', ' ')})",
            "effect": f"-{fmt_inr(abs(d))} vs like new",
        })

    if base_spec["has_box_bill"] == 0:
        d = delta({"has_box_bill": 1})
        factors.append({
            "label": "Missing box or bill",
            "effect": f"-{fmt_inr(abs(d))} vs original box+bill",
        })

    if base_spec["warranty_months_left"] < 6:
        d = delta({"warranty_months_left": 12})
        factors.append({
            "label": f"Active warranty ({base_spec['warranty_months_left']} mos)",
            "effect": f"+{fmt_inr(abs(d))} if 12 mos warranty",
        })

    return factors[:4]


class PredictRequest(BaseModel):
    brand: str
    model: str
    release_year: int
    age_years: float
    storage_gb: int
    ram_gb: int
    battery_health_pct: float
    screen_condition: str
    body_condition: str
    has_box_bill: int
    warranty_months_left: int
    original_price_inr: int
    market_demand_index: float


@app.get("/api/catalogue")
def get_catalogue():
    return BRAND_CATALOGUE


@app.get("/api/metrics")
def get_metrics():
    global METRICS_DATA
    if not METRICS_DATA and METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            METRICS_DATA = json.load(f)
    return METRICS_DATA


@app.post("/api/predict")
def predict_single(req: PredictRequest):
    global MODEL, METRICS_DATA
    if MODEL is None:
        load_app_model()
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model file not available. Train the model first.")

    spec = req.model_dump()
    bm = METRICS_DATA.get("best_model", "SVR")
    best_mape = METRICS_DATA.get("test_metrics", {}).get(bm, {}).get("MAPE", 11.5)

    price = predict_price(MODEL, spec)
    low, high = predict_price_range(MODEL, spec, best_mape)
    factors = compute_factor_effects(MODEL, spec, price)

    return {
        "predicted_price": price,
        "low_price": low,
        "high_price": high,
        "factors": factors,
        "mape": best_mape,
    }


@app.post("/api/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    global MODEL, METRICS_DATA
    if MODEL is None:
        load_app_model()
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model file not available.")

    content = await file.read()
    try:
        user_df = pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    df_prep = user_df.copy()

    if "age_years" not in df_prep.columns:
        if "release_year" in df_prep.columns:
            df_prep["age_years"] = CURRENT_YEAR - pd.to_numeric(
                df_prep["release_year"], errors="coerce"
            ).fillna(CURRENT_YEAR - 2)
        else:
            df_prep["age_years"] = 2.0

    if "market_demand_index" not in df_prep.columns:
        df_prep["market_demand_index"] = 5.5
    if "has_box_bill" not in df_prep.columns:
        df_prep["has_box_bill"] = 1
    if "warranty_months_left" not in df_prep.columns:
        df_prep["warranty_months_left"] = 0
    if "screen_condition" not in df_prep.columns:
        df_prep["screen_condition"] = "minor_scratches"
    if "body_condition" not in df_prep.columns:
        df_prep["body_condition"] = "good"

    required_cols = ["brand", "model", "original_price_inr"]
    missing = [c for c in required_cols if c not in df_prep.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns in CSV: {', '.join(missing)}")

    df_prep = clean_strings(df_prep)

    for f in ALL_FEATURES:
        if f not in df_prep.columns:
            df_prep[f] = np.nan

    bm = METRICS_DATA.get("best_model", "SVR")
    best_mape = METRICS_DATA.get("test_metrics", {}).get(bm, {}).get("MAPE", 11.5)

    raw_preds = MODEL.predict(df_prep[ALL_FEATURES])
    preds = np.round(raw_preds / 100) * 100
    half_mape = (best_mape / 2) / 100
    low_preds = np.round(preds * (1 - half_mape) / 100) * 100
    high_preds = np.round(preds * (1 + half_mape) / 100) * 100

    result_df = user_df.copy()
    result_df["predicted_price_inr"] = preds.astype(int)
    result_df["price_range_low_inr"] = low_preds.astype(int)
    result_df["price_range_high_inr"] = high_preds.astype(int)

    records = result_df.fillna("").to_dict(orient="records")
    return {"records": records}


@app.post("/api/retrain")
async def retrain_model(file: Optional[UploadFile] = File(None)):
    target_df = None
    if file is not None:
        content = await file.read()
        try:
            target_df = pd.read_csv(io.BytesIO(content))
            if "price_inr" not in target_df.columns:
                raise HTTPException(status_code=400, detail="Missing required target column 'price_inr'")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {e}")

    try:
        best_name, all_metrics = run_training_pipeline(raw_df=target_df, save_plots=True)
        load_app_model()
        return {
            "best_model": best_name,
            "test_metrics": all_metrics.get("test_metrics", {}),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training pipeline error: {e}")


# Static files mount
if FIGURES_DIR.exists():
    app.mount("/figures", StaticFiles(directory=str(FIGURES_DIR)), name="figures")

UI_DIR = ROOT_DIR / "ui"
if UI_DIR.exists():
    app.mount("/", StaticFiles(directory=str(UI_DIR), html=True), name="ui")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
