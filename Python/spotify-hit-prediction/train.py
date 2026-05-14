from pipeline import load_data, preprocess
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier
import joblib

# Load data
df = load_data('E:/For_Work/Portfolio/Python/Spotify Tracks Dataset/dataset.csv')
df = preprocess(df)
# Split
X = df.drop('hit', axis=1)
y = df['hit']

print("Class disctibution:\n", y.value_counts())
print("Features:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Model
xgb = XGBClassifier(
    eval_metric='logloss',
    random_state=42
    )

# Hyperparameter tuning
params = {
    "n_estimators": [200, 300],
    "max_depth": [4, 6],
    "learning_rate": [0.01, 0.1],
}

grid = GridSearchCV(
    xgb, param_grid=params,
    cv=3, scoring="f1",
    verbose=1, n_jobs=1
)
grid.fit(X_train, y_train)
best_model = grid.best_estimator_

print("Best params:", grid.best_estimator_)

# Evaluate on test set
y_pred = best_model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and feature list
joblib.dump(best_model, 'E:/For_Work/Portfolio/Python/Spotify Tracks Dataset/models/model.pkl')
joblib.dump(X.columns.tolist(), 'E:/For_Work/Portfolio/Python/Spotify Tracks Dataset/models/features.pkl')

print("Model and features saved successfully!")