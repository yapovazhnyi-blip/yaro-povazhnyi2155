"""
06_model_evaluation.py
---------------------------
Step 6 - Final evaluation on the held-out TEST set.

What this file does:
    - Loads the best model (XGBoost based on validation results)
    - Runs it ONCE on the test set (final, untouched data)
    - Produces final metric report
    - Plots residuals, predictions vs actual, and feature importance
    
Best practices:
    - Test set is used exactly ONCE - after all model selection is done
    - Residual analysis reveals systematic errors (e.g., model underpredicts long trips)
    - Feature importance guides future feature engineering
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from project_config import DATA_DIR, MODEL_DIR, TARGET_COLUMN

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
PLOT_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)


def load_test_data():
    X_test = pd.read_parquet(os.path.join(DATA_DIR, "X_test.parquet"))
    y_test = pd.read_parquet(os.path.join(DATA_DIR, "y_test.parquet")).squeeze()
    return X_test, y_test


def final_metrics(y_true, y_pred) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print("\n" + "="*50)
    print("  FINAL TEST SET EVALUATION (XGBoost)")
    print("="*50)
    print(f"  MSE  : {mse:.4f}")
    print(f"  RMSE : {rmse:.4f} minutes")
    print(f"  MAE  : {mae:.4f} minutes")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print("="*50 + "\n")
    
    
def plot_predicted_vs_actual(y_true, y_pred) -> None:
    """Perfect predictions lie on the diagonal (y=x line)."""
    fog, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.2, s=5, color="steelblue")
    lim = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lim, lim, "r--", lw=1.5, label="Perfect prediction")
    ax.set_xlabel("Actual Duration (min)")
    ax.set_ylabel("Predicted Duration (min)")
    ax.set_title("Predicted vs Actual - XGBoost (Test Set)", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_pred_vs_actual"))
    plt.close()
    log.info("Plot 7 saved: predicted vs actual")
    
    
def plot_residuals(y_true, y_pred) -> None:
    """
    Residuals = actual - predicted.
    Ideal: randomly distibuted around 0 with no pattern.
    Patterns reveal systematic model bias.
    """
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Residual Analysis (Test Set)", fontweight="bold")
    
    # Residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.15, s=5, color="coral")
    axes[0].axhline(0, color="black", lw=1.5, linestyle="--")
    axes[0].set_xlabel("Predicted Duration (min)")
    axes[0].set_ylabel("Residual ")
    axes[0].set_title("Residual vs Predicted")
    
    # Residuals distribution
    sns.histplot(residuals, bins=60, kde=True, ax=axes[1], color="coral")
    axes[1].axvline(0, color="black", lw=1.5, linestyle="--")
    axes[1].set_xlabel("Residual (min)")
    axes[1].set_ylabel("Residual Distribution")
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_residuals.png"), dpi=150)
    plt.close()
    log.info("Plot 8 saved: residual analysis")
    
    log.info(
        "Residual stats -> mean: %.3f | std: %.3f | skew: %.3f",
        residuals.mean(), residuals.std(), residuals.skew()
    )
    
    
def plot_feature_importance(model, feature_names: list) -> None:
    """Bar chart of XGBoost feature importance (gain metric)."""
    importance = model.get_booster().get_score(importance_type="gain")
    imp_df = (
        pd.Series(importance)
        .rename("importance")
        .reset_index()
        .rename(columns={"index": "feature"})
        .sort_values("importance", ascending=True)
    )
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp_df["feature"], imp_df["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("XGBoost Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_importance.png"), dpi=150)
    plt.close()
    log.info("Plot 9 saved: feature importance")
    

if __name__ == "__main__":
    X_test, y_test = load_test_data()
    
    log.info("Loading best model (XGBoost) ...")
    model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
    
    y_pred = model.predict(X_test)
    
    final_metrics(y_test, y_pred)
    plot_predicted_vs_actual(y_test, y_pred)
    plot_residuals(y_test, y_pred)
    plot_feature_importance(model, X_test.columns.tolist())
    
    log.info("All evaluation plots saved to %s", PLOT_DIR)
    log.info("Step 6 complete ✓")
    log.info(" Full pipeline finished!")