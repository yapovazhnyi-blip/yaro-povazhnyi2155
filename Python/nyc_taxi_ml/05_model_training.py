"""
05_model_training.py
-----------------------
Step 5 - Train and compare multiple models

Models trained:
    - 5-fold cross-valodation on training data for unbiased model selection
    - Hyperparameter search with RandomizedSearchCV
    - All models saved with joblib for deployment
    - Validation set used for final model comparison BEFORE touching test set
"""

import os
import logging
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

from project_config import DATA_DIR, MODEL_DIR, RANDOM_STATE, XGB_PARAMS, RF_PARAMS

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def load_splits():
    """Load all train/vak arrays from disk.""" 
    X_train_sc = np.load(os.path.join(DATA_DIR, "X_train_sc.npy"))
    X_val_sc   = np.load(os.path.join(DATA_DIR, "X_val_sc.npy"))
    y_train    = pd.read_parquet(os.path.join(DATA_DIR, "y_train.parquet")).squeeze()
    y_val      = pd.read_parquet(os.path.join(DATA_DIR, "y_val.parquet")).squeeze()
    
    # XGBoost / RF also work on unscaled features
    X_train = pd.read_parquet(os.path.join(DATA_DIR, "X_train.parquet"))
    X_val   = pd.read_parquet(os.path.join(DATA_DIR, "X_val.parquet"))
    
    return X_train_sc, X_val_sc, X_train, X_val, y_train, y_val


def regression_metrics(y_true, y_pred, label: str) -> dict:
    """
    Compute a fill suite pf refression metrics.
    
    MSE  - penalises large error heavily (squared)
    RMSE - in the same unit as target (minutes), interpretable
    MAE  - average absolute error, robust to outliers
    R^2  - proportion of variance explained (1.0 = perfect)
    MAPE - percentage error, useful for stakeholderes
    """
    mse  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    log.info(
        "\n--- %s -----------------\n"
        " MSE   : %.4f\n"
        " RMSE  : %.4f min\n"
        " MAE   : %.4f min\n"
        " R2    : %.4f\n"
        " MAPE  : %.2f%%\n"
        "________________________________________",
        label, mse, rmse, mae, r2, mape
    )
    return {"model": label, "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "MAPE": mape}


def cross_validate_model(model, X, y, name: str, cv: int = 5) -> float:
    """Run k-fold CV and return mean RMSE"""
    scores = cross_val_score(
        model, X, y,
        scoring="neg_root_mean_squared_error",
        cv=cv, n_jobs=1
    )
    mean_rmse = -scores.mean()
    std_rmse  = scores.std()
    log.info("CV [%s]: RMSE = %.4f ± %.4f (k=%d)", name, mean_rmse, std_rmse, cv)
    return mean_rmse


def train_linear_regression(X_train_sc, y_train, X_val_sc, y_val) -> dict:
    log.info("Training Linear Regression ...")
    t0 = time.time()
    model = LinearRegression()
    
    cross_validate_model(model, X_train_sc, y_train, "LinearRegrssion")
    
    model.fit(X_train_sc, y_train)
    preds = model.predict(X_val_sc)
    
    joblib.dump(model, (os.path.join(MODEL_DIR, "linear_regression.joblib")))
    log.info("Trained in %.1fs", time.time() - t0)
    return regression_metrics(y_val, preds, "Linear Regression")


def train_random_forest(X_train, y_train, X_val, y_val) -> dict:
    log.info("Training Random Forest ...")
    t0 = time.time()
    
    # Randomised hyperparameter search
    param_dist = {
        "n_estimators": [100, 200],
        "max_depth":    [8, 12],
        "min_samples_split": [2, 5],
        "min_samples_leaf":   [1, 2],
    }
    base_rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=1)
    search  = RandomizedSearchCV(
        base_rf, param_dist,
        n_iter = 5, cv=3,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE, n_jobs=1, verbose=0
    )
    search.fit(X_train, y_train)
    model = search.best_estimator_
    log.info("Best RF params: %s", search.best_params_)
    
    preds = model.predict(X_val)
    joblib.dump(model, (os.path.join(MODEL_DIR, "random_forest.joblib")))
    log.info("Trained in %.1fs", time.time() - t0)
    return regression_metrics(y_val, preds, "Random Forest")


def train_xgboost(X_train, y_train, X_val, y_val) -> dict:
    log.info("Training XGBoost ...")
    t0 = time.time()
    
    model = xgb.XGBRegressor(**XGB_PARAMS, verbosity=0)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    preds = model.predict(X_val)
    joblib.dump(model, (os.path.join(MODEL_DIR, "xgboost.joblib")))
    log.info("Trained in %.1fs", time.time() - t0)
    return regression_metrics(y_val, preds, "XGBoost")


if __name__ == "__main__":
    X_train_sc, X_val_sc, X_train, X_val, y_train, y_val = load_splits()
    
    results = []
    results.append(train_linear_regression(X_train_sc, y_train, X_val_sc, y_val))
    results.append(train_random_forest(X_train, y_train, X_val, y_val))
    results.append(train_xgboost(X_train, y_train, X_val, y_val))
    
    summary = pd.DataFrame(results).set_index("model").sort_values("RMSE")
    log.info("\n\n--- Model Comparison (Validation Set) ----\n%s\n", summary.to_string())
    
    summary.to_csv(os.path.join(DATA_DIR, "model_comaprison.csv"))
    log.info("Step 5 complete ✓")