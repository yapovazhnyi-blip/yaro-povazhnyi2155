"""
run_pipline.py
----------------
Run the entire ML pipeline end-to-end with one command:
    python run_pipeline.py
"""

import subprocess, sys, os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set credantials in the enviroment that all subporcesses will inherit
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"path\to\your\key.json"
os.environ["GCP_PROJECT_ID"] = "iconic-parces-269314"

STEPS = [
    ("Step 1 — Data Connection",       "01_data_connection.py"),
    ("Step 2 — Data Cleaning",         "02_data_cleaning.py"),
    ("Step 3 — EDA & Visualisation",   "03_eda_visualization.py"),
    ("Step 4 — Feature Engineering",   "04_feature_engineering.py"),
    ("Step 5 — Model Training",        "05_model_training.py"),
    ("Step 6 — Model Evaluation",      "06_model_evaluation.py"),
]

for label, script in STEPS:
    print(f"\n{'='*55}\n  {label}\n{'='*55}")
    script_path = os.path.join(SCRIPT_DIR, script)
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n Pipeline failed at: {script}")
        sys.exit(1)
        
print("\n Full pipeline completed successfully!")