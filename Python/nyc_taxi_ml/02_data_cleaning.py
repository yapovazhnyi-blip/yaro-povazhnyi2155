"""
 02_data_cleaning.py
 -------------------
 Step 2 - Data cleaning and validation.
    
 What this file does:
    - Loads raw Parquet file from Step 1
    - Computes the target variable: trip_duration_minutes
    - Removes outliers using IQR and domain-knowledge rules
    - Handles missing values
    - Reports a before/after cleaning summary
    - Saves cleaned data
      
 Best practices:
    - Every filter decision is documented with a reason
    - We log rows removed at each stage (full audit trail)
    - Outlier detection uses statistical methods (IQR), nit magic numbers alone    
"""
    
import os
import logging
import numpy as np
import pandas as pd
    
from project_config import DATA_DIR, TARGET_COLUMN
    
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)
    
def load_raw(path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    
    # BigQuery returns Decimal types for numeric columns - convert to float64
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass
            
    # Force all numberic-looking columns to float64
    numeric_cols = ["trip_distance", "fare_amount", "tip_amoune",
                    "total_amount", "passenger_count"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(float)
            
    log.info("Loaded raw data: %d rows", len(df))
    return df
    
    
def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive trip duration in minutes fro, pickup/dropoff timestamps.
    This is our regression target.
    """
        
    df = df.copy()
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"])
    df["dropoff_datetime"] = pd.to_datetime(df["dropoff_datetime"])
    df[TARGET_COLUMN] = (
        df["dropoff_datetime"] - df["pickup_datetime"]
    ).dt.total_seconds() / 60
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(float)
    return df
    
    
def remove_invalid_durations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove trips with physically impossible durations.
    NYC taxi trips: reasonable range is 1-180 minutes.
    """
    before = len(df)
    df = df[(df[TARGET_COLUMN] >= 1) & (df[TARGET_COLUMN] <= 180)]
    log.info("Duration filter: removed %d rows", before - len(df))
    return df


def remove_outliers_iqr(df: pd.DataFrame, columns: list, multiplier: float = 1.5) -> pd.DataFrame:
    """
    Remove rows where any of the given columns fall outside
    [Q1 - multiplier*IQR, Q3 + miltipler*IQR].
    Standard statistical outlier removal.
    """
    before = len(df)
    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - multiplier * iqr, q3 + multiplier* iqr
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        log.info(" IQR filter [%s]: kept rows in [%.2f, %.2f]", col, lower, upper)
    log.info("Outlier removal: removed %d rows total", before - len(df))
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy:
        - Numeric columns -> fill with median (robust to skew)
        - Categorical columns -> fill with mode
    """
    before = df.isnull().sum().sum()
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
            
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode)
    
    after = df.isnull().sum().sum()
    log.info("Missing values: %d -> %d", before, after)
    return df


def cleaning_report(raw: pd.DataFrame, clean: pd.DataFrame) -> None:
    rows_removed = len(raw) - len(clean)
    pct = rows_removed / len(raw) * 100
    log.info(
        "\n--- Cleaning Report -----------------------------------\n",
        "   Raw rows    : %d\n",
        "   Clean rows  : %d\n",
        "   Removed     : %d (%.1f%%)\n",
        "   Target stats:\n%s\n",
        len(raw), len(clean), rows_removed, pct,
        clean[TARGET_COLUMN].describe().to_string()
    )
    

if __name__ == "__main__":
    raw_path   = os.path.join(DATA_DIR, "raw_taxi_data.parquet")
    clean_path = os.path.join(DATA_DIR, "clean_taxi_data.parquet")
    
    raw = load_raw(raw_path)
    df  = compute_target(raw)
    
    df  = remove_invalid_durations(df)
    df  = remove_outliers_iqr(df, columns=[TARGET_COLUMN, "trip_distance", "fare_amount"])
    df  = handle_missing(df)
    
    cleaning_report(raw, df)
    df.to_parquet(clean_path, index=False)
    log.info("Saved clean data -> %s", clean_path)
    log.info("Step 2 complete ✓")