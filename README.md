# Used Phone Price Predictor

A complete machine-learning project that estimates the resale value (INR) of a
second-hand smartphone from its specs and physical condition.

> **Note — Synthetic Data:** All data is generated programmatically by
> `src/data_generator.py`. No real transaction records were used. See
> *Data assumptions* below.

---

## Quick start

```bash
# 1. Clone / unzip and enter the project directory
cd phone-price-predictor

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate synthetic data
python src/data_generator.py

# 5. Train models (takes 3–8 min on a laptop)
python src/train.py

# 6. Launch the web app (supports single device + batch CSV upload)
streamlit run app.py

# 7. (Optional) Run tests
pytest tests/ -v
```

---

## Results

Final test-set metrics (Random Forest — best model):

| Metric | Value |
|--------|-------|
| MAE    | ~₹2,400 |
| RMSE   | ~₹4,200 |
| R²     | ~0.970  |
| MAPE   | ~9–12%  |

*(Exact values printed at end of `python src/train.py` and stored in `models/metrics.json`)*

---

## Project structure

```
phone-price-predictor/
├── data/
│   ├── raw/used_phones_raw.csv          # synthetic, intentionally messy
│   └── processed/                       # train.csv / test.csv (80:20 split)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb        # full EDA + modelling walkthrough
├── src/
│   ├── config.py          → all paths, constants, feature lists, random seed
│   ├── data_generator.py  → synthetic data generation
│   ├── preprocessing.py   → cleaning + sklearn ColumnTransformer pipeline
│   ├── train.py           → trains/compares 5 models, saves best
│   ├── evaluate.py        → metrics + plot generation
│   └── predict.py         → load model, predict from dict
├── models/
│   ├── best_model.joblib  → serialised sklearn Pipeline
│   └── metrics.json       → all model metrics
├── reports/figures/       → PNG plots saved during training
├── tests/                 → pytest test suite
├── app.py                 → Streamlit web application
└── requirements.txt
```

---

## Practical mapping

| Practical | Covered where |
|-----------|--------------|
| **Prac 1** — Data collection & EDA | `data_generator.py`, notebook §1–2 |
| **Prac 2** — Data cleaning & preprocessing | `preprocessing.py`, notebook §3 |
| **Prac 4** — Linear Regression | `train.py` → `train_linear()`, notebook §4 |
| **Prac 5** — Feature selection | `train.py` → `SelectKBest` in linear step |
| **Prac 8** — KNN Regressor | `train.py` → `train_knn()`, K vs RMSE plot |
| **Prac 10** — Decision Tree | `train.py` → `train_decision_tree()`, tree visualisation |
| **Prac 11** — Ensemble (Random Forest) | `train.py` → `train_random_forest()`, feature importances |

---

## Data assumptions

The synthetic pricing formula encodes the following:

1. **Depreciation** — Exponential decay per year; rate is brand-specific:
   Apple ≈ 82 %, Samsung ≈ 72 %, Xiaomi / Realme ≈ 55–58 %.
2. **Battery health** — Nonlinear multiplier; sharp drop below 80 %.
3. **Screen condition** — flawless ×1.0, minor scratches ×0.94, cracked ×0.70.
4. **Body condition** — like\_new ×1.0, good ×0.92, fair ×0.82, poor ×0.68.
5. **Box + bill** — +3 % premium.
6. **Warranty** — +0.5 % per remaining month.
7. **Demand index** — Shifts price by up to ±8 %; derived from brand popularity + age + noise.
8. **Log-normal noise** (σ ≈ 0.06) — Same phone, different sale price.

Deliberate mess injected for the preprocessing exercise:
- 3 % missing values in `battery_health_pct`, `storage_gb`, `body_condition`.
- 1.5 % duplicate rows.
- 1 % outlier rows (absurd prices, battery > 100).
- Inconsistent casing / whitespace in string columns.

---

## Limitations

- Model learns a **synthetic** pricing function, not real market data.
- Only **30 models** from **8 brands** catalogued — unseen phones extrapolate poorly.
- **India market only** — pricing norms differ by region/platform.
- **Static snapshot** — market prices drift; model needs periodic retraining.
- SVR and KNN scale poorly to very large datasets (not an issue here at 6 K rows).

---

## Tech stack

Python 3.11 · pandas · numpy · scikit-learn · matplotlib · seaborn · joblib · Streamlit
