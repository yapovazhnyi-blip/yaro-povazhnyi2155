# 🚕 NYC Taxi Trip Duration Prediction

A complete end-to-end Machine Learning pipeline using **Google BigQuery public datasets**, Python, and Scikit-learn.

## 📌 Project Overview
Predicts taxi trip duration in New York City using historical trip data from the TLC Yellow Taxi dataset.

## 🔧 Tech Stack
- Python 3.10+
- Google BigQuery (via `google-cloud-bigquery`)
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- XGBoost

## 📊 ML Metrics Used
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R² Score

## 🚀 Setup
```bash
pip install -r requirements.txt
```
Set your GCP credentials:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/key.json"
```

## 📂 Pipeline Steps
1. `01_data_connection.py` — Connect to BigQuery & pull data
2. `02_data_cleaning.py` — Clean & validate data
3. `03_eda_visualization.py` — Exploratory Data Analysis
4. `04_feature_engineering.py` — Feature creation
5. `05_model_training.py` — Train & compare multiple models
6. `06_model_evaluation.py` — Evaluate with full metrics

## 📈 Results
| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression | ~2.1 min | ~3.8 min | 0.61 |
| Random Forest | ~2.5 min | ~2.7 min | 0.78 |
| XGBoost | ~1.6 min | ~2.3 min | 0.83 |
