"""
app.py
------
Streamlit web app for the Used Phone Price Predictor.

Layout: wide mode, two-column form + result card, performance tabs below.
Design: Nuanced monochrome palette with subtle slate-gray tones and elevated cards.
Features: single-device interactive estimation & batch CSV upload estimation.
"""

import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# -- Path setup -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Used Phone Price Estimator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- CSS injection: Nuanced Slate & Gray Tone Palette -----------------------
# Design Tokens:
#   Canvas Background:        #f8fafc (Slate 50 - gentle, calm light gray)
#   Surface / Card:           #ffffff (Pure White)
#   Elevated Surface:         #f1f5f9 (Slate 100)
#   Subtle Border:            #e2e8f0 (Slate 200 - crisp 1px separation)
#   Input Border:             #cbd5e1 (Slate 300)
#   Primary Text / Headers:   #0f172a (Slate 900 - high contrast deep charcoal)
#   Body / Labels:            #334155 (Slate 700 - clear, sharp readability)
#   Secondary / Muted:        #64748b (Slate 500 - compliant contrast ratio)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- Base & Canvas ---- */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    color: #0f172a;
}
.stApp {
    background-color: #f8fafc;
}

/* ---- Header Bar ---- */
.header-bar {
    background: #0f172a;
    color: #ffffff;
    padding: 18px 28px;
    margin: -1rem -1rem 1.8rem -1rem;
    border-radius: 0 0 6px 6px;
    box-shadow: 0 2px 4px rgba(15, 23, 42, 0.08);
}
.header-title {
    font-size: 1.15rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: #ffffff;
    margin: 0;
}
.header-sub {
    font-size: 0.78rem;
    font-weight: 400;
    color: #94a3b8;
    margin-top: 3px;
}

/* ---- Panel Container (Left Form) ---- */
.form-panel {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
}

/* ---- Section Headers ---- */
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #475569;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 6px;
    margin-top: 18px;
    margin-bottom: 12px;
}
.section-label:first-of-type {
    margin-top: 0;
}

/* ---- Result Valuation Card (Right Column) ---- */
.result-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 26px 28px;
    box-shadow: 0 4px 6px -1px rgba(15, 23, 42, 0.05), 0 2px 4px -2px rgba(15, 23, 42, 0.05);
}
.price-badge-label {
    font-size: 0.70rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 6px;
}
.price-main {
    font-size: 2.5rem;
    font-weight: 700;
    color: #0f172a;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.price-range-pill {
    display: inline-block;
    background: #f1f5f9;
    border: 1px solid #e2e8f0;
    color: #334155;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.82rem;
    font-weight: 500;
    font-variant-numeric: tabular-nums;
    margin-top: 10px;
    margin-bottom: 18px;
}
.card-divider {
    border: none;
    border-top: 1px solid #f1f5f9;
    margin: 16px 0;
}
.factors-header {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 10px;
}

/* ---- Factor Rows (Key Drivers) ---- */
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 9px 12px;
    background: #f8fafc;
    border: 1px solid #f1f5f9;
    border-radius: 6px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.factor-key {
    color: #334155;
    font-weight: 500;
}
.factor-val {
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: #0f172a;
    background: #ffffff;
    padding: 2px 8px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
}

/* ---- Spec Summary Caption ---- */
.spec-summary {
    font-size: 0.78rem;
    color: #64748b;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 10px 14px;
    margin-top: 14px;
    line-height: 1.6;
}

/* ---- Explicit Form Controls Styling & Contrast Guarantee ---- */
/* Labels across all widgets */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSelectSlider"] label,
div[data-testid="stFileUploader"] label {
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 4px !important;
}

/* Checkbox specific fix: Guaranteed dark text on light canvas */
div[data-testid="stCheckbox"] {
    background-color: transparent !important;
    padding: 4px 0 !important;
}
div[data-testid="stCheckbox"] * {
    color: #0f172a !important;
}
div[data-testid="stCheckbox"] label span p {
    color: #0f172a !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
}
div[data-testid="stCheckbox"] div[role="checkbox"] {
    border: 1.5px solid #94a3b8 !important;
    background-color: #ffffff !important;
    border-radius: 4px !important;
}
div[data-testid="stCheckbox"] div[role="checkbox"][aria-checked="true"] {
    background-color: #0f172a !important;
    border-color: #0f172a !important;
}

/* Radio buttons options */
div[data-testid="stRadio"] div[role="radiogroup"] label {
    color: #1e293b !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] p {
    color: #1e293b !important;
}

/* Sliders */
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #64748b !important;
    font-size: 0.74rem !important;
    font-weight: 500 !important;
}

/* Primary and secondary buttons */
div[data-testid="stButton"] button[kind="primary"],
div[data-testid="stDownloadButton"] button {
    background-color: #0f172a !important;
    color: #ffffff !important;
    border: 1px solid #0f172a !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.02em !important;
    padding: 9px 18px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08) !important;
    transition: all 0.15s ease !important;
}
div[data-testid="stButton"] button[kind="primary"]:hover,
div[data-testid="stDownloadButton"] button:hover {
    background-color: #1e293b !important;
    border-color: #1e293b !important;
}

/* Tabs styling */
div[data-testid="stTabs"] {
    margin-top: 24px;
}
div[data-testid="stTabs"] button {
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: #64748b !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
    background-color: transparent !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #0f172a !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #0f172a !important;
}

/* Dataframe header & rows */
div[data-testid="stDataFrame"] {
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background-color: #ffffff;
}
div[data-testid="stDataFrame"] th {
    background-color: #f1f5f9 !important;
    color: #0f172a !important;
    font-weight: 600 !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] section {
    background-color: #ffffff !important;
    border: 1.5px dashed #cbd5e1 !important;
    border-radius: 8px !important;
    padding: 18px !important;
}
div[data-testid="stFileUploader"] section:hover {
    border-color: #64748b !important;
}
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] small {
    color: #475569 !important;
}

/* Metric summary widgets (Batch Predict) */
div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04) !important;
}
div[data-testid="stMetricLabel"] {
    color: #64748b !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
div[data-testid="stMetricValue"] {
    color: #0f172a !important;
    font-size: 1.35rem !important;
    font-weight: 700 !important;
    font-variant-numeric: tabular-nums;
}

/* Separator */
.tab-sep {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 32px 0 0 0;
}
</style>
""", unsafe_allow_html=True)

# -- Header Bar -------------------------------------------------------------
st.markdown("""
<div class="header-bar">
    <div class="header-title">Used Phone Price Estimator &mdash; India Market</div>
    <div class="header-sub">Statistical valuation model for second-hand smartphones based on specs, condition, and market demand</div>
</div>
""", unsafe_allow_html=True)


# -- Model loader -----------------------------------------------------------
@st.cache_resource(show_spinner="Loading valuation model ...")
def get_model():
    """Load the saved best-model pipeline. Cached for the session."""
    return load_model(MODEL_PATH)


@st.cache_data(show_spinner=False)
def get_metrics() -> dict:
    """Load test-set metrics from JSON."""
    if not METRICS_PATH.exists():
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


def model_available() -> bool:
    return MODEL_PATH.exists()


# -- Brand / model catalogue helpers ----------------------------------------
ALL_BRANDS = sorted(BRAND_CATALOGUE.keys())

def models_for_brand(brand: str) -> list[str]:
    return [m[0] for m in BRAND_CATALOGUE[brand]["models"]]

def spec_for_model(brand: str, model_name: str) -> dict:
    for m in BRAND_CATALOGUE[brand]["models"]:
        if m[0] == model_name:
            return {
                "release_year": m[1], "original_price_inr": m[2],
                "storages": m[3], "rams": m[4],
            }
    return {}


# -- Indian number formatter ------------------------------------------------
def fmt_inr(n: int) -> str:
    """Format an integer in Indian numbering style, e.g. Rs 1,23,456."""
    s = str(abs(n))
    if len(s) <= 3:
        return f"Rs {s}"
    last3 = s[-3:]
    rest  = s[:-3]
    parts = []
    while len(rest) > 2:
        parts.append(rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.append(rest)
    grouped = ",".join(reversed(parts)) + "," + last3
    return f"Rs {grouped}"


# -- Factor explanation helper ----------------------------------------------
def compute_factor_effects(model, base_spec: dict, base_price: int) -> list[dict]:
    """
    Re-predict with each factor nudged to its baseline reference.
    Reports the actual empirical delta -- zero invented text.
    """
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


# -- Main Layout: Two Column Form & Valuation Card --------------------------
left_col, right_col = st.columns([5, 5], gap="large")

with left_col:
    # Device Specifications
    st.markdown('<div class="section-label">Device Specifications</div>', unsafe_allow_html=True)
    brand = st.selectbox("Brand", ALL_BRANDS, index=ALL_BRANDS.index("Apple"))
    available_models = models_for_brand(brand)
    model_name = st.selectbox("Model", available_models)

    spec_info       = spec_for_model(brand, model_name)
    release_year    = spec_info.get("release_year", 2021)
    original_price  = spec_info.get("original_price_inr", 50000)
    storage_options = spec_info.get("storages", [128])
    ram_options     = spec_info.get("rams", [4])

    c1, c2 = st.columns(2)
    with c1:
        storage = st.selectbox("Internal storage (GB)", storage_options)
    with c2:
        ram = st.selectbox("RAM (GB)", ram_options)

    # Physical Condition
    st.markdown('<div class="section-label">Physical Condition</div>', unsafe_allow_html=True)
    battery = st.slider("Battery health percentage (%)", min_value=55, max_value=100, value=85)

    screen_cond = st.radio(
        "Screen condition",
        options=["flawless", "minor_scratches", "cracked"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )
    body_cond = st.radio(
        "Body condition",
        options=["like_new", "good", "fair", "poor"],
        format_func=lambda x: x.replace("_", " ").title(),
        horizontal=True,
    )

    c3, c4 = st.columns(2)
    with c3:
        has_box = st.checkbox("Has original box + bill", value=True)
    with c4:
        warranty = st.number_input(
            "Warranty remaining (months)", min_value=0, max_value=12, value=0, step=1,
        )

    # Market Factors
    st.markdown('<div class="section-label">Market Dynamics</div>', unsafe_allow_html=True)
    demand_label = st.select_slider(
        "Buyer demand level", options=["Low", "Medium", "High"], value="Medium",
    )
    demand_map   = {"Low": 3.5, "Medium": 5.5, "High": 7.5}
    demand_index = demand_map[demand_label]

    st.button("Calculate estimated resale price", type="primary", use_container_width=True)


with right_col:
    if not model_available():
        st.warning(
            "**Model not found.** Run `python src/train.py` from the project "
            "root, then refresh this page.",
            icon=None,
        )
    else:
        model        = get_model()
        metrics_data = get_metrics()
        best_mape    = 11.5
        if metrics_data and "test_metrics" in metrics_data:
            bm        = metrics_data.get("best_model", "SVR")
            best_mape = metrics_data["test_metrics"].get(bm, {}).get("MAPE", 11.5)

        age_years = float(CURRENT_YEAR - release_year)
        spec = {
            "brand":                brand,
            "model":                model_name,
            "release_year":         release_year,
            "age_years":            age_years,
            "storage_gb":           int(storage),
            "ram_gb":               int(ram),
            "battery_health_pct":   float(battery),
            "screen_condition":     screen_cond,
            "body_condition":       body_cond,
            "has_box_bill":         int(has_box),
            "warranty_months_left": int(warranty),
            "original_price_inr":   original_price,
            "market_demand_index":  demand_index,
        }

        try:
            price     = predict_price(model, spec)
            low, high = predict_price_range(model, spec, best_mape)
            factors   = compute_factor_effects(model, spec, price)

            # Valuation Result Card
            st.markdown(f"""
            <div class="result-card">
              <div class="price-badge-label">Estimated Resale Valuation</div>
              <div class="price-main">{fmt_inr(price)}</div>
              <div class="price-range-pill">
                Expected range: <strong>{fmt_inr(low)} &ndash; {fmt_inr(high)}</strong>
                &nbsp;&middot;&nbsp; Model MAPE ~{best_mape:.1f}%
              </div>
              <div class="card-divider"></div>
              <div class="factors-header">Key Valuation Drivers</div>
            """, unsafe_allow_html=True)

            if factors:
                for f in factors:
                    st.markdown(f"""
                    <div class="factor-row">
                        <span class="factor-key">{f['label']}</span>
                        <span class="factor-val">{f['effect']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div style="font-size:0.83rem;color:#475569;padding:6px 0;">'
                    'Device is in top condition with minimal depreciation penalties.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # Specification summary banner
            st.markdown(f"""
            <div class="spec-summary">
              <strong>{brand} {model_name}</strong> &middot; {storage} GB Storage &middot; {ram} GB RAM<br>
              Age: ~{age_years:.0f} yr &middot; Original Launch: {fmt_inr(original_price)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Valuation error: {e}")


# -- Bottom Tabs Section ----------------------------------------------------
st.markdown('<hr class="tab-sep">', unsafe_allow_html=True)

tab_batch, tab_retrain, tab_perf, tab_about = st.tabs([
    "Batch valuation (CSV upload)",
    "Model retraining",
    "Model performance",
    "Dataset documentation",
])

# -- Tab 1: Batch Prediction ------------------------------------------------
with tab_batch:
    st.markdown("""
    Upload a batch CSV file containing multiple phone listings to generate resale predictions at scale.
    The model computes individual valuations, empirical ranges, and portfolio totals.
    """)

    sample_template_df = pd.DataFrame([
        {
            "brand": "Apple",
            "model": "iPhone 13",
            "release_year": 2021,
            "storage_gb": 128,
            "ram_gb": 4,
            "battery_health_pct": 86.0,
            "screen_condition": "minor_scratches",
            "body_condition": "good",
            "has_box_bill": 1,
            "warranty_months_left": 0,
            "original_price_inr": 79900,
            "market_demand_index": 7.0,
        },
        {
            "brand": "Samsung",
            "model": "Galaxy S22",
            "release_year": 2022,
            "storage_gb": 128,
            "ram_gb": 8,
            "battery_health_pct": 89.0,
            "screen_condition": "flawless",
            "body_condition": "like_new",
            "has_box_bill": 1,
            "warranty_months_left": 3,
            "original_price_inr": 72999,
            "market_demand_index": 6.5,
        },
        {
            "brand": "Xiaomi",
            "model": "Redmi Note 12",
            "release_year": 2023,
            "storage_gb": 128,
            "ram_gb": 6,
            "battery_health_pct": 92.0,
            "screen_condition": "minor_scratches",
            "body_condition": "fair",
            "has_box_bill": 0,
            "warranty_months_left": 0,
            "original_price_inr": 17999,
            "market_demand_index": 5.5,
        },
    ])

    c_top1, c_top2 = st.columns([3, 1])
    with c_top2:
        template_csv = sample_template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download sample CSV template",
            data=template_csv,
            file_name="phone_sample_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    uploaded_file = st.file_uploader(
        "Upload your device spreadsheet (CSV)",
        type=["csv"],
        help="Upload CSV with smartphone specifications",
    )

    if uploaded_file is not None:
        if not model_available():
            st.error("Model file not found. Please train the model first.")
        else:
            try:
                user_df = pd.read_csv(uploaded_file)
                st.markdown(f"**Loaded {len(user_df):,} records.** Preview:")
                st.dataframe(user_df.head(4), use_container_width=True)

                df_prep = user_df.copy()

                # Derive age_years if only release_year is given
                if "age_years" not in df_prep.columns:
                    if "release_year" in df_prep.columns:
                        df_prep["age_years"] = CURRENT_YEAR - pd.to_numeric(df_prep["release_year"], errors="coerce").fillna(CURRENT_YEAR - 2)
                    else:
                        df_prep["age_years"] = 2.0

                # Fill sensible defaults for optional fields if omitted
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
                missing_core = [col for col in required_cols if col not in df_prep.columns]

                if missing_core:
                    st.error(f"Missing required columns in CSV: {', '.join(missing_core)}")
                else:
                    df_prep = clean_strings(df_prep)

                    for f in ALL_FEATURES:
                        if f not in df_prep.columns:
                            df_prep[f] = np.nan

                    model = get_model()
                    metrics_data = get_metrics()
                    mape_val = 11.5
                    if metrics_data and "test_metrics" in metrics_data:
                        bm = metrics_data.get("best_model", "SVR")
                        mape_val = metrics_data["test_metrics"].get(bm, {}).get("MAPE", 11.5)

                    raw_preds = model.predict(df_prep[ALL_FEATURES])
                    preds = np.round(raw_preds / 100) * 100
                    half_mape = (mape_val / 2) / 100
                    low_preds = np.round(preds * (1 - half_mape) / 100) * 100
                    high_preds = np.round(preds * (1 + half_mape) / 100) * 100

                    result_df = user_df.copy()
                    result_df["predicted_price_inr"] = preds.astype(int)
                    result_df["price_range_low_inr"] = low_preds.astype(int)
                    result_df["price_range_high_inr"] = high_preds.astype(int)

                    st.markdown('<div class="section-label">Batch Valuation Summary</div>', unsafe_allow_html=True)
                    m_col1, m_col2, m_col3 = st.columns(3)
                    with m_col1:
                        st.metric("Total devices", f"{len(result_df):,}")
                    with m_col2:
                        avg_price = int(result_df["predicted_price_inr"].mean())
                        st.metric("Average estimated price", fmt_inr(avg_price))
                    with m_col3:
                        total_val = int(result_df["predicted_price_inr"].sum())
                        st.metric("Total portfolio value", fmt_inr(total_val))

                    st.markdown('<div class="section-label">Valuation Output Table</div>', unsafe_allow_html=True)
                    st.dataframe(result_df, use_container_width=True)

                    res_csv = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download valuations spreadsheet (CSV)",
                        data=res_csv,
                        file_name="predicted_phone_resale_prices.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error(f"Error processing CSV: {e}")

# -- Tab 2: Model Retraining ------------------------------------------------
with tab_retrain:
    st.markdown("""
    Train and evaluate all 5 machine learning models (Linear Regression, KNN, Decision Tree, SVR, and Random Forest) using 5-fold cross-validation.
    
    * **Custom Dataset:** Upload your own historical sales records (CSV must contain phone features plus the target column `price_inr`).
    * **Baseline Dataset:** If no file is provided, the pipeline will train on the standard 6,000-device Indian market dataset.
    """)

    c_tr1, c_tr2 = st.columns([3, 1])
    with c_tr2:
        sample_train_df = pd.DataFrame([
            {
                "brand": "Apple",
                "model": "iPhone 13",
                "release_year": 2021,
                "storage_gb": 128,
                "ram_gb": 4,
                "battery_health_pct": 86.0,
                "screen_condition": "minor_scratches",
                "body_condition": "good",
                "has_box_bill": 1,
                "warranty_months_left": 0,
                "original_price_inr": 79900,
                "market_demand_index": 7.0,
                "price_inr": 41200,
            },
            {
                "brand": "Samsung",
                "model": "Galaxy S22",
                "release_year": 2022,
                "storage_gb": 128,
                "ram_gb": 8,
                "battery_health_pct": 89.0,
                "screen_condition": "flawless",
                "body_condition": "like_new",
                "has_box_bill": 1,
                "warranty_months_left": 3,
                "original_price_inr": 72999,
                "market_demand_index": 6.5,
                "price_inr": 34800,
            },
        ])
        sample_tr_bytes = sample_train_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download training CSV template",
            data=sample_tr_bytes,
            file_name="training_data_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    retrain_file = st.file_uploader(
        "Upload custom training CSV (optional)",
        type=["csv"],
        key="retrain_csv_uploader",
        help="Upload CSV with smartphone specs and actual prices in 'price_inr'",
    )

    custom_train_df = None
    has_valid_custom_data = False
    if retrain_file is not None:
        try:
            custom_train_df = pd.read_csv(retrain_file)
            st.markdown(f"**Loaded custom dataset:** {len(custom_train_df):,} records. Preview:")
            st.dataframe(custom_train_df.head(3), use_container_width=True)

            if "price_inr" not in custom_train_df.columns:
                st.error("Missing target column 'price_inr' in the uploaded file. Training requires historical sale prices. To estimate prices on phones without known prices, use the 'Batch valuation' tab.")
                has_valid_custom_data = False
            else:
                has_valid_custom_data = True
        except Exception as e:
            st.error(f"Error reading file: {e}")
            has_valid_custom_data = False

    if has_valid_custom_data:
        st.info(f"Target: Train models on your uploaded dataset ({len(custom_train_df):,} devices).")
    else:
        st.info("Target: Train models on the baseline 6,000-device simulation dataset.")

    btn_label = "Retrain models on custom dataset" if has_valid_custom_data else "Train models on baseline dataset"
    start_training = st.button(btn_label, type="primary", use_container_width=True)

    if start_training:
        status_box = st.status("Executing training pipeline ...", expanded=True)
        def progress_tracker(message: str):
            status_box.write(message)

        try:
            target_df = custom_train_df if has_valid_custom_data else None
            best_model_name, all_metrics = run_training_pipeline(
                raw_df=target_df,
                save_plots=True,
                progress_cb=progress_tracker,
            )
            # Clear caches so the webapp immediately reflects the freshly trained model
            st.cache_resource.clear()
            st.cache_data.clear()

            status_box.update(
                label=f"Training complete! Optimal model: {best_model_name}",
                state="complete",
                expanded=False,
            )

            st.success(f"Model successfully updated and deployed: **{best_model_name}**.")

            tm = all_metrics.get("test_metrics", {})
            if tm:
                best_m = tm.get(best_model_name, {})
                c_m1, c_m2, c_m3, c_m4 = st.columns(4)
                with c_m1:
                    st.metric("Best Model", best_model_name)
                with c_m2:
                    st.metric("Test R2", f"{best_m.get('R2', 0):.4f}")
                with c_m3:
                    st.metric("Test RMSE", fmt_inr(int(best_m.get('RMSE', 0))))
                with c_m4:
                    st.metric("Test MAPE", f"{best_m.get('MAPE', 0):.1f}%")

                st.markdown('<div class="section-label">All Models Comparison Table</div>', unsafe_allow_html=True)
                df_res = pd.DataFrame(tm).T.sort_values("RMSE")
                df_res.index.name = "Model"
                df_res = df_res.rename(columns={
                    "MAE": "MAE (Rs)", "RMSE": "RMSE (Rs)",
                    "MSE": "MSE", "R2": "R2", "MAPE": "MAPE (%)",
                })
                st.dataframe(
                    df_res.style.format({
                        "MAE (Rs)":  "{:,.0f}",
                        "RMSE (Rs)": "{:,.0f}",
                        "MSE":       "{:,.0f}",
                        "R2":        "{:.4f}",
                        "MAPE (%)":  "{:.1f}",
                    }).highlight_min(subset=["RMSE (Rs)"], color="#f1f5f9"),
                    use_container_width=True,
                )

        except Exception as e:
            status_box.update(label="Training failed", state="error", expanded=True)
            st.error(f"Error during training pipeline: {e}")

# -- Tab 3: Model Performance -----------------------------------------------
with tab_perf:
    if not model_available():
        st.info("Train the model first to inspect cross-validation and test metrics.")
    else:
        metrics_data = get_metrics()
        if not metrics_data:
            st.info("No metrics file found. Run `python src/train.py`.")
        else:
            bm = metrics_data.get("best_model", "--")
            st.markdown(f"**Selected Optimal Model:** `{bm}`")

            tm = metrics_data.get("test_metrics", {})
            if tm:
                df_m = pd.DataFrame(tm).T.sort_values("RMSE")
                df_m.index.name = "Model"
                df_m = df_m.rename(columns={
                    "MAE": "MAE (Rs)", "RMSE": "RMSE (Rs)",
                    "MSE": "MSE", "R2": "R2", "MAPE": "MAPE (%)",
                })
                st.dataframe(
                    df_m.style.format({
                        "MAE (Rs)":  "{:,.0f}",
                        "RMSE (Rs)": "{:,.0f}",
                        "MSE":       "{:,.0f}",
                        "R2":        "{:.4f}",
                        "MAPE (%)":  "{:.1f}",
                    }).highlight_min(subset=["RMSE (Rs)"], color="#f1f5f9"),
                    use_container_width=True,
                )

            c_l, c_r = st.columns(2)
            with c_l:
                avp = FIGURES_DIR / "actual_vs_predicted.png"
                if avp.exists():
                    st.image(str(avp), caption="Actual vs Predicted Prices", use_container_width=True)
            with c_r:
                fi = FIGURES_DIR / "feature_importances.png"
                if fi.exists():
                    st.image(str(fi), caption="Feature Importances (Random Forest)", use_container_width=True)

            mc = FIGURES_DIR / "model_comparison.png"
            if mc.exists():
                st.image(str(mc), caption="Model RMSE Comparison Across All 5 Families", use_container_width=True)

# -- Tab 3: Dataset Documentation -------------------------------------------
with tab_about:
    st.markdown("""
### Dataset Architecture

All baseline training records are generated using a controlled econometric simulation (`src/data_generator.py`).
The data models real pricing mechanics observed in the Indian second-hand phone ecosystem (2019-2025).

#### Synthetic Formulation & Assumptions

| Variable / Driver | Implementation Dynamics |
|---|---|
| **Depreciation Curve** | Exponential decay; brand-specific retention (Apple ~82%/yr, Samsung ~72%/yr, Xiaomi/Realme ~55%/yr) |
| **Battery Health Penalty** | Nonlinear degradation; sharp cliff below 80% health |
| **Screen Condition** | Flawless (1.0x), Minor Scratches (0.94x), Cracked screen (0.70x) |
| **Body Condition** | Like new (1.0x), Good (0.92x), Fair (0.82x), Poor (0.68x) |
| **Documentation & Warranty** | Original box & bill adds +3% premium; warranty adds +0.5% per active month |
| **Market Demand Index** | Continuous index (1-10) derived from brand prestige, model age, and organic volume |
| **Price Dispersion** | Multiplicative log-normal noise (sigma ~0.06) simulates seller variation |
| **Deliberate Data Noise** | ~3% missing values, ~1.5% duplicates, ~1% outliers, inconsistent casing for practical cleaning |

#### Boundaries & Scope
- **Geographic Focus:** Republic of India smartphone market (INR denominations).
- **Catalogue:** 30 high-volume models across 8 major manufacturers.
""")
