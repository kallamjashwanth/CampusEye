import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

df = pd.read_csv("timeseries_dataset.csv")
df = df.sort_values(["day", "minutes"])

data = df["count"].values.reshape(-1, 1)

scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

model = load_model("crowd_lstm_model.h5", compile=False)

window_size = 5

# Take last 5 crowd counts
last_5 = data_scaled[-window_size:]
last_5 = last_5.reshape(1, window_size, 1)

# Predict next count
pred_scaled = model.predict(last_5)
pred_count = scaler.inverse_transform(pred_scaled)[0][0]

pred_count = round(pred_count)

def crowd_level(count):
    if count == 0:
        return "Empty"
    elif count <= 20:
        return "Less Crowded"
    elif count <= 40:
        return "Moderate"
    elif count <= 60:
        return "Crowded"
    else:
        return "Very Crowded"

def recommendation(level):
    if level == "Empty":
        return "Mess is almost empty. Best time to visit."
    elif level == "Less Crowded":
        return "Good time to visit mess."
    elif level == "Moderate":
        return "Crowd is manageable."
    elif level == "Crowded":
        return "Mess is crowded. Wait for some time if possible."
    else:
        return "Mess is very crowded. Prefer going later."

level = crowd_level(pred_count)
rec = recommendation(level)

print("Predicted next crowd count:", pred_count)
print("Predicted crowd level:", level)
print("Recommendation:", rec)