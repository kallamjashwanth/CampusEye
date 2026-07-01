import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("smart_classroom/dataset/classroom_energy_dataset.csv")

# Convert time to hour
df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour

X = df[
    [
        "hour",
        "person_count",
        "temperature",
        "humidity",
        "ldr"
    ]
]

y = df["energy_saved"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, pred))
print("R2 Score:", r2_score(y_test, pred))

joblib.dump(model, "smart_classroom/energy_prediction_model.pkl")

print("Model Saved")