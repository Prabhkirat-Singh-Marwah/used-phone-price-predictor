"""
evaluate.py
-----------
Metric computation and all plot generation.
Called from train.py; also used by app.py for the "Model performance" tab.
"""

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FIGURES_DIR, METRICS_PATH


# -- Metric helpers ---------------------------------------------------------

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error (%)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, MSE, RMSE, R2, MAPE."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    return {
        "MAE":  round(mae,  2),
        "MSE":  round(mse,  2),
        "RMSE": round(rmse, 2),
        "R2":   round(r2,   4),
        "MAPE": round(mape_val, 2),
    }


def save_metrics(metrics_dict: dict[str, Any], path: Path = METRICS_PATH) -> None:
    """Serialise metrics dictionary to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"[evaluate] Metrics saved -> {path}")


def load_metrics(path: Path = METRICS_PATH) -> dict[str, Any]:
    """Load metrics from JSON."""
    with open(path) as f:
        return json.load(f)


# -- Plot helpers -----------------------------------------------------------

def _savefig(name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[evaluate] Saved figure -> {path}")


def plot_model_comparison(
    results: dict[str, dict[str, float]],
    metric: str = "RMSE",
) -> None:
    """Bar chart comparing models on a chosen metric."""
    names  = list(results.keys())
    values = [results[n][metric] for n in names]
    colors = ["#e0e0e0"] * len(names)
    best_idx = int(np.argmin(values))
    colors[best_idx] = "#2d6a6a"

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(names, values, color=colors, edgecolor="#aaaaaa", linewidth=0.8)
    ax.bar_label(bars, fmt="{:.0f}", padding=4, fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_title(f"Model comparison -- {metric} (lower is better)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", rotation=15)
    _savefig("model_comparison.png")


def plot_actual_vs_predicted(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Actual vs Predicted Price",
) -> None:
    """Scatter of true price vs predicted price with perfect-fit line."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.4, s=12, color="#2d6a6a", edgecolors="none")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="Perfect fit")
    ax.set_xlabel("Actual price (Rs)", fontsize=11)
    ax.set_ylabel("Predicted price (Rs)", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("actual_vs_predicted.png")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Residual vs fitted plot for the best model."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=12,
               color="#2d6a6a", edgecolors="none")
    ax.axhline(0, color="#cc4444", linewidth=1, linestyle="--")
    ax.set_xlabel("Fitted value (Rs)", fontsize=11)
    ax.set_ylabel("Residual (Rs)", fontsize=11)
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("residuals.png")


def plot_feature_importances(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """Horizontal bar chart for Random Forest feature importances."""
    idx = np.argsort(importances)[::-1][:top_n]
    imp_sorted  = importances[idx]
    names_sorted = [feature_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    ax.barh(range(top_n), imp_sorted[::-1], color="#2d6a6a",
            edgecolor="#aaaaaa", linewidth=0.6)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_sorted[::-1], fontsize=8)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Top {top_n} feature importances (Random Forest)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("feature_importances.png")


def plot_knn_k_vs_error(
    k_values: list[int],
    cv_rmse: list[float],
) -> None:
    """Line plot of CV RMSE vs K for KNN."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_values, cv_rmse, marker="o", color="#2d6a6a",
            linewidth=1.5, markersize=5)
    best_k = k_values[int(np.argmin(cv_rmse))]
    ax.axvline(best_k, color="#cc4444", linewidth=1,
               linestyle="--", label=f"Best K={best_k}")
    ax.set_xlabel("K (neighbours)", fontsize=11)
    ax.set_ylabel("CV RMSE (Rs)", fontsize=11)
    ax.set_title("KNN: K vs CV RMSE", fontsize=12)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    _savefig("knn_k_vs_rmse.png")
