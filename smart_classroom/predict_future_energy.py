# Future energy saving is predicted using the learned daily classroom usage pattern from our collected dataset.
import pandas as pd
import joblib

df = pd.read_csv("classroom_energy_dataset.csv")
model = joblib.load("energy_prediction_model.pkl")

df.columns = df.columns.str.strip()

df["time"] = pd.to_datetime(df["time"])
df["hour"] = df["time"].dt.hour

# Use today's learned pattern as tomorrow's input
future = df[["hour", "person_count", "temperature", "humidity", "ldr"]].copy()

# Optional: make tomorrow slightly different, but still based on data
# future["temperature"] = future["temperature"] + 1

future["predicted_energy_saved"] = model.predict(future)

total_saving = future["predicted_energy_saved"].sum()

print(future.head())
print("Predicted Tomorrow Energy Saving:", round(total_saving, 2), "Wh")
print("Predicted Tomorrow Energy Saving:", round(total_saving / 1000, 3), "kWh")

# Weekly prediction
weekly_saving = total_saving * 7
print("Predicted Weekly Energy Saving:", round(weekly_saving / 1000, 3), "kWh")

future.to_csv("future_energy_prediction.csv", index=False)
print("Saved: future_energy_prediction.csv")