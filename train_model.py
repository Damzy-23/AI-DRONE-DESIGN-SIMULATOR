import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# =========================
# 1. LOAD DATA
# =========================
# Replace this with your actual file name
df = pd.read_csv("hvn_drone_dataset.csv")

# Quick check
print(df.head())
print(df.info())

# =========================
# 2. DEFINE FEATURES + TARGET
# =========================
# Example columns:
# battery_mAh
# frame_weight_kg
# payload_weight_kg
# motor_power_W
# propeller_size_in
# receiver_weight_g
# receiver_power_W
# receiver_type   <-- optional categorical column
# flight_time_min <-- target

target = "flight_time_min"

features = [
    "battery_capacity_mah",
    "battery_weight_g",
    "num_motors",
    "motor_thrust_per_g",
    "propeller_size",
    "frame_size",
    "radio_receiver",
    "payload_capacity_g"
]

# Include this only if you have a receiver type column
if "receiver_type" in df.columns:
    features.append("receiver_type")

X = df[features]
y = df[target]

# =========================
# 3. HANDLE NUMERIC / CATEGORICAL COLUMNS
# =========================
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# =========================
# 4. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 5. BASELINE MODEL
# =========================
linear_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)
linear_preds = linear_model.predict(X_test)

print("\n--- Linear Regression Results ---")
print("MAE:", mean_absolute_error(y_test, linear_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, linear_preds)))
print("R2:", r2_score(y_test, linear_preds))

# =========================
# 6. RANDOM FOREST MODEL
# =========================
rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

print("\n--- Random Forest Results ---")
print("MAE:", mean_absolute_error(y_test, rf_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_preds)))
print("R2:", r2_score(y_test, rf_preds))

# =========================
# 7. HYPERPARAMETER TUNING
# =========================
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
best_preds = best_model.predict(X_test)

print("\n--- Tuned Random Forest Results ---")
print("Best Parameters:", grid_search.best_params_)
print("MAE:", mean_absolute_error(y_test, best_preds))
print("RMSE:", np.sqrt(mean_squared_error(y_test, best_preds)))
print("R2:", r2_score(y_test, best_preds))

# =========================
# 8. EXAMPLE PREDICTION
# =========================
example_input = pd.DataFrame([{
    "battery_capacity_mah": 5000,
    "battery_weight_g": 520,
    "num_motors": 4,
    "motor_thrust_per_g": 900,
    "propeller_size": '9"',
    "frame_size": "330mm Medium",
    "radio_receiver": "ExpressLRS 900",
    "payload_capacity_g": 500
}])

# Remove receiver_type if it does not exist in dataset
if "radio_receiver" not in df.columns:
    example_input = example_input.drop(columns=["radio_receiver"])

predicted_flight_time = best_model.predict(example_input)[0]
print("\nPredicted Flight Time (minutes):", round(predicted_flight_time, 2))