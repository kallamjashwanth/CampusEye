import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load dataset
df = pd.read_csv("mess_project/timeseries_dataset.csv")

# Sort properly
df = df.sort_values(["day", "minutes"])

# Use count column
data = df["count"].values.reshape(-1,1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
X = []
y = []

window_size = 5

for i in range(len(data_scaled)-window_size):
    X.append(data_scaled[i:i+window_size])
    y.append(data_scaled[i+window_size])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Build LSTM
model = Sequential()

model.add(
    LSTM(
        50,
        activation="relu",
        input_shape=(window_size,1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)

# Train
history = model.fit(
    X,
    y,
    epochs=20,
    batch_size=32,
    verbose=1
)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

y_pred = model.predict(X)

y_true_original = scaler.inverse_transform(y)
y_pred_original = scaler.inverse_transform(y_pred)

mae = mean_absolute_error(y_true_original, y_pred_original)
rmse = np.sqrt(mean_squared_error(y_true_original, y_pred_original))
r2 = r2_score(y_true_original, y_pred_original)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)

# Save model
model.save("mess_project/crowd_lstm_model.h5")

print("Model Saved")

