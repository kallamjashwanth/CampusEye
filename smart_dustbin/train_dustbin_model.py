import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = "smart_dustbin"
DATA_PATH = os.path.join(BASE_DIR, "dustbin_30day_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "dustbin_stack_model.pkl")

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

df = df.loc[:, ~df.columns.str.contains("Unnamed")]
if "timestamp" in df.columns:
    df = df.drop(columns=["timestamp"])

slot_order = {
    "Morning": 0,
    "Afternoon": 1,
    "Evening": 2,
    "Night": 3
}

df["slot_order"] = df["time_slot"].map(slot_order)

df = df.sort_values(["bin_name", "day", "slot_order"]).reset_index(drop=True)

df["prev_fill"] = df.groupby("bin_name")["fill_percentage"].shift(1)
df["fill_rate"] = df["fill_percentage"] - df["prev_fill"]
df["next_fill_percentage"] = df.groupby("bin_name")["fill_percentage"].shift(-1)

df = df.dropna()

X = df[
    [
        "day",
        "slot_order",
        "bin_name",
        "location",
        "fill_percentage",
        "prev_fill",
        "fill_rate"
    ]
]

y = df["next_fill_percentage"]

categorical_cols = ["bin_name", "location"]
numeric_cols = [
    "day",
    "slot_order",
    "fill_percentage",
    "prev_fill",
    "fill_rate"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols)
    ]
)

svm_model = SVR(kernel="rbf", C=100, gamma="scale")

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=42
)

cat_model = CatBoostRegressor(
    iterations=200,
    learning_rate=0.05,
    depth=5,
    verbose=0,
    random_state=42
)

meta_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)

stack_model = StackingRegressor(
    estimators=[
        ("svm", svm_model),
        ("xgb", xgb_model),
        ("cat", cat_model)
    ],
    final_estimator=meta_model,
    cv=5
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", stack_model)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

joblib.dump(pipeline, MODEL_PATH)

print("Model saved:", MODEL_PATH)