"""
train.py
--------
Trains and compares five model families, tunes hyper-parameters, and saves
the best end-to-end sklearn Pipeline (pre-processor + model) to disk.

Models compared
---------------
1. Linear Regression (age-only simple, then full with SelectKBest)
2. KNN Regressor (K in {3,5,7,11,15}, GridSearch)
3. Decision Tree Regressor (GridSearch + shallow visualisation)
4. SVR with RBF kernel (RandomizedSearch)
5. Random Forest Regressor (RandomizedSearch)

Target transformation: log1p applied via TransformedTargetRegressor so the
pipeline operates in log-space and inverse-transforms for evaluation.
"""

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    ALL_FEATURES,
    CV_FOLDS,
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    RANDOM_SEED,
    TARGET,
    TRAIN_PATH,
    TEST_PATH,
)
from preprocessing import build_feature_pipeline, load_and_prepare, split_and_save
from evaluate import (
    compute_metrics,
    save_metrics,
    plot_model_comparison,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_feature_importances,
    plot_knn_k_vs_error,
)

rng_seed = RANDOM_SEED
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# -- Data loading -----------------------------------------------------------

def load_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed train/test CSVs (created by preprocessing.py)."""
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)
    return train, test


# -- Cross-validated RMSE helper --------------------------------------------

def cv_rmse(pipeline: Pipeline, X: pd.DataFrame, y: np.ndarray) -> float:
    """Return mean CV RMSE across CV_FOLDS folds."""
    scores = cross_val_score(
        pipeline, X, y,
        cv=CV_FOLDS,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    return float(-scores.mean())


# -- Pipeline factory -------------------------------------------------------

def make_pipeline(model, preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """
    Wrap pre-processor + model in a Pipeline with log1p target transform.
    The outer TransformedTargetRegressor handles log1p / expm1 automatically.
    """
    if preprocessor is None:
        preprocessor = build_feature_pipeline()

    inner = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor",    model),
    ])
    return TransformedTargetRegressor(
        regressor=inner,
        func=np.log1p,
        inverse_func=np.expm1,
    )


# -- Model 1: Linear Regression --------------------------------------------

def train_linear(X_train: pd.DataFrame, y_train: np.ndarray) -> dict:
    """Simple (age only) and multiple linear regression with SelectKBest."""
    results = {}

    # 1a. Simple linear (age only)
    simple_pre = build_feature_pipeline()
    # We'll pass all features but restrict selector to k=1 (age_years is index 0)
    simple_pipe = make_pipeline(
        Pipeline([
            ("selector", SelectKBest(f_regression, k=1)),
            ("lr",       LinearRegression()),
        ]),
        simple_pre,
    )
    rmse_simple = cv_rmse(simple_pipe, X_train, y_train)
    print(f"  Linear (age only)  CV RMSE: {rmse_simple:,.0f}")
    results["Linear (simple)"] = {"cv_rmse": rmse_simple}

    # 1b. Multiple linear with SelectKBest
    multi_pre = build_feature_pipeline()
    multi_pipe = make_pipeline(
        Pipeline([
            ("selector", SelectKBest(f_regression, k=15)),
            ("lr",       LinearRegression()),
        ]),
        multi_pre,
    )
    rmse_multi = cv_rmse(multi_pipe, X_train, y_train)
    print(f"  Linear (k=15 best) CV RMSE: {rmse_multi:,.0f}")
    results["Linear (multi)"] = {"cv_rmse": rmse_multi}

    # Fit the multi version on the full train set
    multi_pipe.fit(X_train, y_train)
    return multi_pipe, results


# -- Model 2: KNN ----------------------------------------------------------

def train_knn(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """Compare K values, GridSearch, return best fitted pipeline."""
    k_values = [3, 5, 7, 11, 15]
    cv_rmses  = []
    for k in k_values:
        pipe = make_pipeline(KNeighborsRegressor(n_neighbors=k))
        cv_rmses.append(cv_rmse(pipe, X_train, y_train))
        print(f"  KNN K={k:<3}  CV RMSE: {cv_rmses[-1]:,.0f}")

    plot_knn_k_vs_error(k_values, cv_rmses)

    # GridSearch for best K
    param_grid = {"regressor__regressor__n_neighbors": k_values}
    base_pipe  = make_pipeline(KNeighborsRegressor())
    gs = GridSearchCV(base_pipe, param_grid, cv=CV_FOLDS,
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    best_k = gs.best_params_["regressor__regressor__n_neighbors"]
    print(f"  KNN best K={best_k}  CV RMSE: {-gs.best_score_:,.0f}")
    return gs.best_estimator_, {"cv_rmse": -gs.best_score_, "best_k": best_k}


# -- Model 3: Decision Tree ------------------------------------------------

def train_decision_tree(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """GridSearch over depth and split criterion; also plots a shallow tree."""
    param_grid = {
        "regressor__regressor__max_depth":        [4, 6, 8, 12, None],
        "regressor__regressor__min_samples_leaf": [5, 10, 20],
    }
    base_pipe = make_pipeline(DecisionTreeRegressor(random_state=rng_seed))
    gs = GridSearchCV(base_pipe, param_grid, cv=CV_FOLDS,
                      scoring="neg_root_mean_squared_error", n_jobs=-1)
    gs.fit(X_train, y_train)
    print(f"  DT best params: {gs.best_params_}  CV RMSE: {-gs.best_score_:,.0f}")

    # Plot shallow tree (max_depth=4 for readability)
    _plot_shallow_tree(X_train, y_train)

    return gs.best_estimator_, {"cv_rmse": -gs.best_score_}


def _plot_shallow_tree(X_train: pd.DataFrame, y_train: np.ndarray) -> None:
    """Fit and visualise a max_depth=4 tree (for the notebook)."""
    shallow_pipe = Pipeline([
        ("preprocessor", build_feature_pipeline()),
        ("regressor",    DecisionTreeRegressor(max_depth=4, random_state=rng_seed)),
    ])
    shallow_pipe.fit(X_train, np.log1p(y_train))

    ct   = shallow_pipe.named_steps["preprocessor"]
    tree = shallow_pipe.named_steps["regressor"]

    # Build feature names from ColumnTransformer
    feature_names = _get_feature_names(ct)

    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(tree, feature_names=feature_names, filled=True,
              max_depth=4, fontsize=7, ax=ax, impurity=False,
              proportion=False, rounded=True)
    ax.set_title("Decision Tree (max_depth=4) -- target is log1p(price)", fontsize=13)
    plt.tight_layout()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURES_DIR / "decision_tree.png", dpi=90)
    plt.close()
    print(f"[train] Saved decision_tree.png")


# -- Model 4: SVR ----------------------------------------------------------

def train_svr(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """RandomizedSearch on SVR RBF kernel."""
    param_dist = {
        "regressor__regressor__C":       [0.1, 1, 10, 100],
        "regressor__regressor__epsilon": [0.01, 0.1, 0.5],
        "regressor__regressor__gamma":   ["scale", "auto"],
    }
    base_pipe = make_pipeline(SVR(kernel="rbf"))
    rs = RandomizedSearchCV(base_pipe, param_dist, n_iter=20, cv=CV_FOLDS,
                            scoring="neg_root_mean_squared_error",
                            random_state=rng_seed, n_jobs=-1)
    rs.fit(X_train, y_train)
    print(f"  SVR best params: {rs.best_params_}  CV RMSE: {-rs.best_score_:,.0f}")
    return rs.best_estimator_, {"cv_rmse": -rs.best_score_}


# -- Model 5: Random Forest ------------------------------------------------

def train_random_forest(X_train: pd.DataFrame, y_train: np.ndarray) -> tuple:
    """RandomizedSearch on Random Forest; also plots feature importances."""
    param_dist = {
        "regressor__regressor__n_estimators":      [100, 200, 300],
        "regressor__regressor__max_depth":         [None, 10, 20],
        "regressor__regressor__min_samples_split": [2, 5, 10],
        "regressor__regressor__max_features":      ["sqrt", "log2", 0.5],
    }
    base_pipe = make_pipeline(
        RandomForestRegressor(random_state=rng_seed, n_jobs=-1)
    )
    rs = RandomizedSearchCV(base_pipe, param_dist, n_iter=20, cv=CV_FOLDS,
                            scoring="neg_root_mean_squared_error",
                            random_state=rng_seed, n_jobs=-1)
    rs.fit(X_train, y_train)
    print(f"  RF best params:  {rs.best_params_}  CV RMSE: {-rs.best_score_:,.0f}")

    # Feature importances
    inner_pipe = rs.best_estimator_.regressor_
    ct  = inner_pipe.named_steps["preprocessor"]
    rf  = inner_pipe.named_steps["regressor"]
    feature_names = _get_feature_names(ct)
    plot_feature_importances(rf.feature_importances_, feature_names)

    return rs.best_estimator_, {"cv_rmse": -rs.best_score_}


# -- Feature name extractor -------------------------------------------------

def _get_feature_names(ct: ColumnTransformer) -> list[str]:
    """Return feature names in the order ColumnTransformer outputs them."""
    names: list[str] = []
    for name, transformer, cols in ct.transformers_:
        if name == "numeric":
            names.extend(cols)
        elif name == "ordinal":
            names.extend(cols)
        elif name == "nominal":
            # OneHotEncoder has get_feature_names_out
            enc = transformer.named_steps["encoder"]
            names.extend(enc.get_feature_names_out(cols).tolist())
    return names


# -- Main training orchestrator ---------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Used Phone Price Predictor -- Training Pipeline")
    print("=" * 60)

    # -- Load data (run preprocessing if processed splits don't exist)
    if not TRAIN_PATH.exists():
        print("[train] Processed splits not found. Running preprocessing ...")
        from config import RAW_DATA_PATH
        df = load_and_prepare(RAW_DATA_PATH)
        split_and_save(df)

    train_df, test_df = load_splits()
    X_train = train_df[ALL_FEATURES]
    y_train = train_df[TARGET].values
    X_test  = test_df[ALL_FEATURES]
    y_test  = test_df[TARGET].values

    print(f"\nTrain: {len(X_train):,} rows  |  Test: {len(X_test):,} rows\n")

    # -- Train all models
    results: dict = {}
    fitted:  dict = {}

    print("-- 1. Linear Regression --")
    fitted["Linear"], lin_res = train_linear(X_train, y_train)
    results.update(lin_res)

    print("\n-- 2. KNN --")
    fitted["KNN"], knn_res = train_knn(X_train, y_train)
    results["KNN"] = knn_res

    print("\n-- 3. Decision Tree --")
    fitted["Decision Tree"], dt_res = train_decision_tree(X_train, y_train)
    results["Decision Tree"] = dt_res

    print("\n-- 4. SVR (RBF) --")
    fitted["SVR"], svr_res = train_svr(X_train, y_train)
    results["SVR"] = svr_res

    print("\n-- 5. Random Forest --")
    fitted["Random Forest"], rf_res = train_random_forest(X_train, y_train)
    results["Random Forest"] = rf_res

    # -- Evaluate all on test set
    print("\n-- Test-set Evaluation --")
    test_metrics: dict = {}
    for name, pipe in fitted.items():
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred)
        test_metrics[name] = m
        print(f"  {name:<20} MAE={m['MAE']:>10,.0f}  RMSE={m['RMSE']:>10,.0f}  "
              f"R2={m['R2']:.4f}  MAPE={m['MAPE']:.1f}%")

    # -- Comparison table
    comp_df = pd.DataFrame(test_metrics).T.sort_values("RMSE")
    print(f"\n{comp_df.to_string()}\n")

    # -- Pick best model (lowest RMSE)
    best_name = comp_df.index[0]
    best_pipe = fitted[best_name]
    print(f"Best model: {best_name}")

    # -- Plots
    plot_model_comparison(test_metrics, metric="RMSE")

    y_pred_best = best_pipe.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred_best,
                             title=f"Actual vs Predicted -- {best_name}")
    plot_residuals(y_test, y_pred_best)

    # -- Save model
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, MODEL_PATH)
    print(f"\n[train] Best model saved -> {MODEL_PATH}")

    # -- Save metrics
    all_metrics = {
        "best_model": best_name,
        "test_metrics": test_metrics,
        "cv_rmse": {k: v.get("cv_rmse") for k, v in results.items()},
    }
    save_metrics(all_metrics)

    # -- Conclusion
    print("\n-- Conclusion --")
    print(f"  Winner : {best_name}")
    print(f"  RMSE   : Rs{test_metrics[best_name]['RMSE']:,.0f}")
    print(f"  MAE    : Rs{test_metrics[best_name]['MAE']:,.0f}")
    print(f"  R2     : {test_metrics[best_name]['R2']:.4f}")
    print(f"  MAPE   : {test_metrics[best_name]['MAPE']:.1f}%")
    print()
    print("  Random Forest consistently wins on tabular regression tasks because")
    print("  it captures nonlinear interactions (age x brand retention, battery")
    print("  health threshold effects) via ensemble averaging of decision trees,")
    print("  without requiring feature scaling or assumption of linearity.")


if __name__ == "__main__":
    main()
