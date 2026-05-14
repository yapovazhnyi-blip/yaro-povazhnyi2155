"""
project_config.py
---------
Central configation for the project.
All BigQuery settings, model hyperparameters, and path live here.
"""

import os

#  Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Cloud / BigQuery settings
# Option 1: set env var GOOGLE_APPLICATION_CREDENTIALS externally
# Option 2: point directly to your service-account JSON here

GCP_CREDENTIALS_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")

# Your GCP project ID
GCP_PROJECT_ID = "your_project_name"

# BigQuery dataset
BQ_DATASET = "bigquery-public-data.new_york_taxi_trips"
BQ_TABLE = "tlc_yellow_trips_2022"

# How many rows to pull from BigQuery for training
SAMPLE_ROWS = 500_000

# Target & features columns
TARGET_COLUMN = "trip_duration_minutes"

FEATURE_COLUMNS = [
    "pickup_hour",
    "pickup_day_of_week",
    "pickup_month",
    "is_weekend",
    "passenger_count",
    "trip_distance",
    "rate_code",
    "pu_location_id",
    "do_location_id",
    "vendor_id",
]

# Model hyperparameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 12,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}