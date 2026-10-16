"""
app.py
------
Streamlit web app for the Used Phone Price Predictor.

Layout: wide mode, two-column form + result card, performance tabs below.
Design: strict black / white / gray palette. No color accents.
"""

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

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Used Phone Price Estimator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -- CSS injection ----------------------------------------------------------
# Palette (black / white / grays only):
#   #000000  header bg, price, thick borders
#   #111111  primary text
#   #333333  body text, labels, section headings
#   #555555  secondary / caption text
#   #888888  muted (used sparingly)
#   #cccccc  light dividers
#   #f0f0f0  row hover, subtle fills
#   #ffffff  card bg, app bg
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

/* ---- Reset / base ---- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #111111;
}
.stApp { background-color: #ffffff; }

/* ---- Header bar ---- */
.header-bar {
    background: #000000;
    color: #ffffff;
    padding: 16px 28px 14px 28px;
    margin: -1rem -1rem 2rem -1rem;
    font-size: 1.05rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.header-sub {
    font-size: 0.74rem;
    font-weight: 400;
    color: #999999;
    margin-top: 4px;
}

/* ---- Section labels ---- */
.section-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #111111;
    border-bottom: 2px solid #111111;
    padding-bottom: 5px;
    margin-top: 24px;
    margin-bottom: 10px;
}

/* ---- Result card ---- */
.result-card {
    background: #ffffff;
    border: 1.5px solid #111111;
    border-radius: 3px;
    padding: 26px 30px 22px 30px;
    margin-top: 2px;
}
.price-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #555555;
    margin-bottom: 8px;
}
.price-main {
    font-size: 2.5rem;
    font-weight: 600;
    color: #000000;
    font-variant-numeric: tabular-nums;
    letter-spacing: -0.02em;
    line-height: 1.05;
}
.price-range {
    font-size: 0.82rem;
    color: #555555;
    margin-top: 7px;
    font-variant-numeric: tabular-nums;
}
.divider {
    border: none;
    border-top: 1px solid #cccccc;
    margin: 18px 0 14px 0;
}
.factors-label {
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #333333;
    margin-bottom: 8px;
}

/* ---- Factor rows ---- */
.factor-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
    font-size: 0.82rem;
}
.factor-row:last-child { border-bottom: none; }
.factor-key { color: #333333; }
.factor-val { font-weight: 500; font-variant-numeric: tabular-nums; color: #000000; }

/* ---- Spec summary ---- */
.spec-summary {
    font-size: 0.74rem;
    color: #555555;
    margin-top: 14px;
    line-height: 1.9;
}

/* ---- Streamlit widget label overrides ---- */
/* Force all form labels to be dark, readable */
label, .stRadio label, .stCheckbox label,
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stRadio"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label,
div[data-testid="stSelectSlider"] label {
    font-size: 0.81rem !important;
    font-weight: 500 !important;
    color: #111111 !important;
}
/* Radio option labels */
div[data-testid="stRadio"] p {
    color: #111111 !important;
    font-size: 0.81rem;
}
/* Slider value display */
div[data-testid="stSlider"] [data-testid="stTickBarMin"],
div[data-testid="stSlider"] [data-testid="stTickBarMax"] {
    color: #555555;
    font-size: 0.75rem;
}
/* Selectbox text */
div[data-testid="stSelectbox"] div[data-baseweb="select"] {
    border-color: #cccccc;
}

/* ---- Primary button ---- */
div[data-testid="stButton"] button[kind="primary"] {
    background: #000000 !important;
    color: #ffffff !important;
    border: none;
    border-radius: 3px;
    font-weight: 500;
    font-size: 0.84rem;
    letter-spacing: 0.04em;
}
div[data-testid="stButton"] button[kind="primary"]:hover {
    background: #222222 !important;
}

/* ---- Tabs ---- */
div[data-testid="stTabs"] button {
    font-size: 0.81rem;
    color: #333333;
    font-weight: 500;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #000000;
    border-bottom-color: #000000 !important;
}

/* ---- Dataframe ---- */
div[data-testid="stDataFrame"] th {
    background-color: #f0f0f0 !important;
    color: #111111 !important;
    font-weight: 600;
}

/* ---- Separator before tabs ---- */
.tab-sep {
    border: none;
    border-top: 1.5px solid #111111;
    margin: 32px 0 0 0;
}

/* ---- Warning / info box text ---- */
div[data-testid="stAlert"] {
    color: #111111;
}
</style>
""", unsafe_allow_html=True)

# -- Header -----------------------------------------------------------------
st.markdown("""
<div class="header-bar">
    Used Phone Price Estimator &mdash; India Market
    <div class="header-sub">Estimates second-hand smartphone resale value in INR</div>
</div>
""", unsafe_allow_html=True)


# -- Model loader -----------------------------------------------------------
@st.cache_resource(show_spinner="Loading model ...")
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
    Re-predict with each factor nudged to its best value.
    Reports the actual delta -- no invented text.
    """
    factors = []

    def delta(patch: dict) -> int:
        return predict_price(model, {**base_spec, **patch}) - base_price

    batt = base_spec["battery_health_pct"]
    if batt < 90:
        d = delta({"battery_health_pct": 95.0})
        factors.append({
            "label": f"Battery at {batt:.0f}%",
            "effect": f"-{fmt_inr(abs(d))} vs 95% battery",
        })

    sc = base_spec["screen_condition"]
    if sc != "flawless":
        d = delta({"screen_condition": "flawless"})
        factors.append({
            "label": f"Screen: {sc.replace('_', ' ')}",
            "effect": f"-{fmt_inr(abs(d))} vs flawless",
        })

    bc = base_spec["body_condition"]
    if bc != "like_new":
        d = delta({"body_condition": "like_new"})
        factors.append({
            "label": f"Body: {bc.replace('_', ' ')}",
            "effect": f"-{fmt_inr(abs(d))} vs like new",
        })

    if base_spec["has_box_bill"] == 0:
        d = delta({"has_box_bill": 1})
        factors.append({
            "label": "No box / bill",
            "effect": f"-{fmt_inr(abs(d))} vs with box+bill",
        })

    if base_spec["warranty_months_left"] < 6:
        d = delta({"warranty_months_left": 12})
        factors.append({
            "label": f"{base_spec['warranty_months_left']} months warranty",
            "effect": f"+{fmt_inr(abs(d))} if 12 months left",
        })

    return factors[:4]


# -- Layout: two columns ----------------------------------------------------
left_col, right_col = st.columns([4, 5], gap="large")

with left_col:

    # Device ----------------------------------------------------------------
    st.markdown('<div class="section-label">Device</div>', unsafe_allow_html=True)
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
        storage = st.selectbox("Storage (GB)", storage_options)
    with c2:
        ram = st.selectbox("RAM (GB)", ram_options)

    # Condition -------------------------------------------------------------
    st.markdown('<div class="section-label">Condition</div>', unsafe_allow_html=True)
    battery = st.slider("Battery health (%)", min_value=55, max_value=100, value=85)

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
            "Warranty months left", min_value=0, max_value=12, value=0, step=1,
        )

    # Market ----------------------------------------------------------------
    st.markdown('<div class="section-label">Market</div>', unsafe_allow_html=True)
    demand_label = st.select_slider(
        "Demand level", options=["Low", "Medium", "High"], value="Medium",
    )
    demand_map   = {"Low": 3.5, "Medium": 5.5, "High": 7.5}
    demand_index = demand_map[demand_label]

    st.button("Estimate price", type="primary", use_container_width=True)


# -- Right column: result ---------------------------------------------------
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
        best_mape    = 15.0
        if metrics_data and "test_metrics" in metrics_data:
            bm        = metrics_data.get("best_model", "SVR")
            best_mape = metrics_data["test_metrics"].get(bm, {}).get("MAPE", 15.0)

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

            # Price card
            st.markdown(f"""
            <div class="result-card">
              <div class="price-label">Estimated resale price</div>
              <div class="price-main">{fmt_inr(price)}</div>
              <div class="price-range">
                Likely range &nbsp; {fmt_inr(low)} &ndash; {fmt_inr(high)}
                &nbsp; &middot; &nbsp; Model MAPE ~{best_mape:.1f}%
              </div>
              <div class="divider"></div>
              <div class="factors-label">What moved this price</div>
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
                    '<div style="font-size:0.82rem;color:#333333;">'
                    'Phone is in excellent condition &mdash; minimal deductions.</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

            # Spec summary line below card
            st.markdown(f"""
            <div class="spec-summary">
              {brand} {model_name} &middot; {storage} GB / {ram} GB RAM
              &middot; {age_years:.0f} yr old &middot; Launch {fmt_inr(original_price)}
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# -- Tabs -------------------------------------------------------------------
st.markdown('<hr class="tab-sep">', unsafe_allow_html=True)

tab_perf, tab_about = st.tabs(["Model performance", "About the data"])

with tab_perf:
    if not model_available():
        st.info("Train the model first to see performance metrics.")
    else:
        metrics_data = get_metrics()
        if not metrics_data:
            st.info("No metrics file found. Run `python src/train.py`.")
        else:
            bm = metrics_data.get("best_model", "--")
            st.markdown(f"**Best model:** {bm}")

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
                    }).highlight_min(subset=["RMSE (Rs)"], color="#e8e8e8"),
                    use_container_width=True,
                )

            c_l, c_r = st.columns(2)
            with c_l:
                avp = FIGURES_DIR / "actual_vs_predicted.png"
                if avp.exists():
                    st.image(str(avp), caption="Actual vs Predicted",
                             use_container_width=True)
            with c_r:
                fi = FIGURES_DIR / "feature_importances.png"
                if fi.exists():
                    st.image(str(fi), caption="Feature Importances (RF)",
                             use_container_width=True)

            mc = FIGURES_DIR / "model_comparison.png"
            if mc.exists():
                st.image(str(mc), caption="Model RMSE Comparison",
                         use_container_width=True)

with tab_about:
    st.markdown("""
### Data

All data used in this project is **fully synthetic** -- generated by
`src/data_generator.py`. No real transaction records were used.

**What the synthetic data encodes:**

| Assumption | Detail |
|---|---|
| Depreciation | Exponential decay; rate is brand-specific (Apple ~82%, Realme ~55% per year) |
| Battery penalty | Nonlinear; sharp drop below 80% health |
| Condition discounts | Screen cracked -> x0.70; body poor -> x0.68 |
| Box + bill premium | +3% |
| Warranty | +0.5% per remaining month |
| Demand index | Derived from brand tier + phone age + Gaussian noise |
| Price noise | Log-normal (sigma ~0.06) -- same phone, different sale price |
| Deliberate mess | 3% missing values, 1.5% duplicates, 1% absurd outliers, inconsistent casing |

**Why synthetic?**
Real used-phone transaction data for the Indian market is proprietary.
Synthetic data lets us design a controlled preprocessing exercise where we
know exactly what mess was injected and why.

**Scope:** India market only. Prices reflect ~2023-25 resale norms.

**Limitations:**
- Model learns the *synthetic* pricing function, not real market behaviour.
- Only 30 phone models from 8 brands are catalogued.
- Demand index is a proxy; real demand depends on platform (OLX, Cashify, etc.).
- Model trained on static snapshot; market prices drift over time.
""")
