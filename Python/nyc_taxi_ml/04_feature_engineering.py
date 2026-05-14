"""
04_feature_engineering.py
--------------------------
Step 4 - Feature engineering and train/test split.

What this file does:
    - Extracts datetime features from pickup_datetime
    - Encodes categorical variables
    - Split data into train / validation / test sets (stratified by hour)
    - Applies StandardScaler to numeric features
    
Best practies:
    - Scaler is FIT only on train set, then TRANSFORM val and test
      (prevents data leakage - a very common mistake)
    - Label encoding is used for tree models; OHE noted as alternative
    - All processed sets are saved so each step reproducivle    
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

from project_config import DATA_DIR, MODEL_DIR, TARGET_COLUMN, FEATURE_COLUMNS, TEST_SIZE, VALIDATION_SIZE, RANDOM_STATE

def extract_datetime_features(df: DataFrame) -> DataFrame:
    """
    Extract rich temporal feature the strongest predictors in taxi datasets.
    These are typically the strogest predictors in taxi datasets.
    """
    df = df.copy()
    dt = df["pickup_datetime"]
    
    df["pickup_hour"]        = dt.dt.hour
    df["pickup_day_of_week"] = dt.dt.dayofweek           # 0=Mon, 6=Sun
    df["pickup_month"]       = dt.dt.month
    df["pickup_week"]        = dt.dt.isocalendar().week.astype(int)
    df["is_weekend"]         = (df["pickup_day_of_week"] >= 5).astype(int)
    df["is_rush_hour"]       = df["pickup_hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
    
    log.info("DateTime features extracted")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Label-encode low-cardinality categoricals.
    For tree-based models (RF, XGBoost) this is sufficient.
    Note: for liner models, consider OneHotEncoding instead.
    """
    df = df.copy()
    cat_cols = ["vendor_id"]
    
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            log.info(" Encoded '%s' : %d unique values", col, len(le.classes_))
            
    # Save encoders for inference
    joblib.dump(encoders, os.path.join(MODEL_DIR, "label_encoders.joblib"))
    return df


def split_data(df: pd.DataFrame):
    """
    Three-way split: train / validation / test.
    
    Why three splits?
        - train  -model learns from this
        - val    -tune hyperparameters (cross-validation also used)
        = test   -final, untouched evaluation (only used ONCE at the end)
    
    Startify by pickup_hour sp all time-of-day patterns are represented
    proportionally in every split.
    """
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    # First: carve off test set
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["pickup_hour"],     # stratify on time-of-day
    )
    
    # Then: split remaining into train and validation
    val_fraction = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_fraction,
        random_state=RANDOM_STATE,
        stratify=X_trainval["pickup_hour"],
    )
    
    log.info(
        "Split sizes -> train: %d | val: %d | test: %d",
        len(X_train), len(X_val), len(X_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_features(X_train, X_val, X_test):
    """
    StandardScaler: zero mean, unit variance.
    IMPORTANT: fit ONLY on X_train, then transform all three sets.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)
    
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))
    log.info("Scaler fitter and saved")
    return X_train_sc, X_val_sc, X_test_sc


if __name__ == "__main__":
    clean_path = os.path.join(DATA_DIR, "clean_taxi_data.parquet")
    df = pd.read_parquet(clean_path)
    
    df = extract_datetime_features(df)
    df = encode_categoricals(df)
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_sc, X_val_sc, X_test_sc = scale_features(X_train, X_val, X_test)
    
    # Persist all splits
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_sc": X_train_sc, "X_val_sc": X_val_sc, "X_test_sc": X_test_sc,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
    }
    for name, obj in splits.items():
        if isinstance(obj, np.ndarray):
            np.save(os.path.join(DATA_DIR, f"{name}.npy"), obj)
        elif isinstance(obj, pd.DataFrame):
            obj.to_parquet(os.path.join(DATA_DIR, f"{name}.parquet"))
        elif isinstance(obj, pd.Series):
            obj.to_frame().to_parquet(os.path.join(DATA_DIR, f"{name}.parquet"))
            
    log.info("All splits saved %s", DATA_DIR)
    log.info("Step 4 complete ✓")