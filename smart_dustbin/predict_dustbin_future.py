import pandas as pd
import os

BASE_DIR = "smart_dustbin"

DATA_PATH = os.path.join(BASE_DIR, "dustbin_30day_dataset.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "future_dustbin_prediction.csv")

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

future_rows = []

for bin_name in df["bin_name"].unique():

    bin_df = df[df["bin_name"] == bin_name].copy()
    location = bin_df["location"].iloc[-1]

    # Difference between consecutive readings
    bin_df["diff"] = bin_df["fill_percentage"].diff()

    # Use only positive increases as actual filling
    positive_rates = bin_df[bin_df["diff"] > 0]["diff"]

    if len(positive_rates) == 0:
        avg_slot_rate = 5
    else:
        avg_slot_rate = positive_rates.mean()

    latest_fill = bin_df["fill_percentage"].iloc[-1]

    current_fill = latest_fill

    full_reached = False
    collection_done = False

    for future_day in range(1, 31):

        for slot_name, order in slot_order.items():

            if current_fill >= 100:
                current_fill = 5
                collection_done = True

            else:
                current_fill = current_fill + avg_slot_rate
                current_fill = min(current_fill, 100)

            if current_fill >= 90:
                priority = "Urgent"
            elif current_fill >= 75:
                priority = "High"
            elif current_fill >= 50:
                priority = "Medium"
            else:
                priority = "Low"

            if current_fill >= 100:
                days_remaining = 0
            else:
                remaining_slots = (100 - current_fill) / avg_slot_rate
                days_remaining = round(remaining_slots / 4, 1)

            future_rows.append({
                "future_day": future_day,
                "time_slot": slot_name,
                "bin_name": bin_name,
                "location": location,
                "predicted_fill": round(current_fill, 2),
                "priority": priority,
                "days_remaining": days_remaining,
                "collection_status": "Collected / Reset" if collection_done else "Not Collected"
            })

            collection_done = False

future_df = pd.DataFrame(future_rows)
future_df.to_csv(OUTPUT_PATH, index=False)

print("Saved:", OUTPUT_PATH)
print(future_df.head(20))
print("Rows:", len(future_df))