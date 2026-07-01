import pandas as pd
import numpy as np

df = pd.read_csv("smart_dustbin/dustbin_daywise_dataset.csv")

all_data = []

for cycle in range(3):   # 10 days × 3 = 30 days

    temp = df.copy()

    temp["day"] = temp["day"] + cycle * 10

    # small variation
    temp["fill_percentage"] = (
        temp["fill_percentage"]
        + np.random.randint(-3, 4, len(temp))
    )

    temp["fill_percentage"] = temp["fill_percentage"].clip(0, 100)

    all_data.append(temp)

final_df = pd.concat(all_data)

final_df.to_csv("smart_dustbin/dustbin_30day_dataset.csv", index=False)

print(final_df.shape)
print("Saved dustbin_30day_dataset.csv")