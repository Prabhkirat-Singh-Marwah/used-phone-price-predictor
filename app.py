"""
app.py
------
Streamlit web application entrypoint for the Used Phone Price Predictor.
All UI presentation logic, CSS styling, components, and layouts are encapsulated
in src/ui.py.
"""

import json
import sys
from pathlib import Path

import streamlit as st

# -- Path setup -------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import METRICS_PATH, MODEL_PATH
from predict import load_model
import ui

# -- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Used Phone Price Estimator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -- Model & metrics caching ------------------------------------------------
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
    """Check if trained model binary exists on disk."""
    return MODEL_PATH.exists()


# -- Main Application Layout ------------------------------------------------
def main() -> None:
    # 1. Custom CSS styling & header banner
    ui.inject_custom_css()
    ui.render_header()

    # 2. Main valuation workspace (Left: Input Form, Right: Valuation Card)
    left_col, right_col = st.columns([5, 5], gap="large")

    with left_col:
        spec, release_year, original_price = ui.render_input_form()

    with right_col:
        if not model_available():
            st.warning(
                "**Model not found.** Run `python src/train.py` from the project "
                "root, then refresh this page.",
                icon=None,
            )
        else:
            model = get_model()
            metrics_data = get_metrics()
            ui.render_valuation_card(
                model=model,
                spec=spec,
                release_year=release_year,
                original_price=original_price,
                metrics_data=metrics_data,
            )

    # 3. Bottom Tabs Section
    st.markdown('<hr class="tab-sep">', unsafe_allow_html=True)

    tab_batch, tab_retrain, tab_perf, tab_about = st.tabs([
        "Batch valuation (CSV upload)",
        "Model retraining",
        "Model performance",
        "Dataset documentation",
    ])

    with tab_batch:
        model_obj = get_model() if model_available() else None
        ui.render_batch_tab(model=model_obj, metrics_data=get_metrics())

    with tab_retrain:
        ui.render_retrain_tab()

    with tab_perf:
        ui.render_performance_tab(
            model_available=model_available(), metrics_data=get_metrics()
        )

    with tab_about:
        ui.render_about_tab()


if __name__ == "__main__":
    main()
