"""
01_data_connection.py
----------------------
Step 1 - Connect to BigQuery and pull raw data.
    
What this file does:
    - Authenticates with Google Cloud using a service account.
    - Runs a prameterized SQL query against a public BigQuery dataset
    - Saves the results to a local Parquet file for repoducibility
        
Best practices applied:
    - Credentials are never hard-coded - read from env or file
    - Row limit is configurable so you don't accidentlly bill yourself for TBs
    - Arrow/Parquet used instead of CSV for speed and type fidelity
 """
    
import os
import logging

import pandas as pd
from google.cloud import bigquery
from google.oauth2 import service_account

from project_config import (
    GCP_PROJECT_ID,
    GCP_CREDENTIALS_PATH,
    BQ_DATASET,
    BQ_TABLE,
    SAMPLE_ROWS,
    DATA_DIR,
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger(__name__)

def get_bq_client() -> bigquery.Client:
    """
    Build and return an authenticated BigQuery client.
    
    Authentication priority:
      1. Service-account JSON path from env/config
      2. Application Default Credentials (gcloud auth login)
    """
    if GCP_CREDENTIALS_PATH and os.path.exists(GCP_CREDENTIALS_PATH):
        log.info("Autheticating with service-account key: %s", GCP_CREDENTIALS_PATH)
        creds = service_account.Credentials.from_service_account_file(
            GCP_CREDENTIALS_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        return bigquery.Client(project=GCP_PROJECT_ID, credentials=creds)
    
    log.info("Using Application Default Credentials (ADC)")
    return bigquery.Client(project=GCP_PROJECT_ID)


def build_query(dataset: str, table: str, limit: int) -> str:
    """
    Return the SQL query that extracts raw trip data.
    
    We pull only the columns we need to minimise bites billed.
    Filtering happens in SQL - cheaper then pulling everything.
    """
    return f"""
    SELECT
        vendor_id,
        pickup_datetime,
        dropoff_datetime,
        passenger_count,
        trip_distance,
        rate_code,
        pickup_location_id AS pu_location_id,
        dropoff_location_id AS do_location_id,
        fare_amount,
        tip_amount,
        total_amount
    FROM
        `{dataset}.{table}`
    WHERE
        -- Basic sanity filters applied at query time
        pickup_datetime IS NOT NULL
        AND dropoff_datetime IS NOT NULL
        AND trip_distance > 0
        AND fare_amount > 0
        AND passenger_count BETWEEN 1 and 6
    ORDER BY
        RAND()   -- random sample so we don't get one time-of-day bias
    LIMIT {limit}
    """
    
    
def pull_data(client: biguery.Client, query: str) -> pd.DataFrame:
    """Execute query and return a pandas DataFrame"""
    log.info("Executing BigQuery query ...")
    job = client.query(query)
    df = job.result().to_dataframe()
    log.info("Download %d rows, %d columns", *df.shape)
    return df

def save_data(df: pd.DataFram, path: Path) -> None:
    """Persist DataFrame as Parquet (preserves dtypes, faster then CSV)."""
    df.to_parquet(path, index=False)
    log.info("Saved raw data -> %s", path)
    
    
# Main
if __name__== "__main__":
    output_path = os.path.join(DATA_DIR, "raw_taxi_data.parquet")
    
    client = get_bq_client()
    query  = build_query(BQ_DATASET, BQ_TABLE, SAMPLE_ROWS)
    df = pull_data(client, query)
    
    log.info("\nSample rows:\n%s", df.head(3).to_string())
    log.info("\nColumn dtypes:\n%s", df.dtypes)
    
    save_data(df, output_path)
    log.info("Step 1 complete ✓")