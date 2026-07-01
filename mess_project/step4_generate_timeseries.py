import pandas as pd
import numpy as np

df = pd.read_csv("prepared_crowd_dataset.csv")

all_days = []

for day in range(1, 31):  # 30 days
    temp = df.copy()

    # Add random variation
    variation = np.random.randint(-3, 4, size=len(temp))

    temp["count"] = temp["final_count"] + variation

    temp["count"] = temp["count"].clip(lower=0)

    temp["day"] = day

    all_days.append(temp)

final_df = pd.concat(all_days, ignore_index=True)

final_df.to_csv("timeseries_dataset.csv", index=False)

print(final_df.head())
print("Rows:", len(final_df))