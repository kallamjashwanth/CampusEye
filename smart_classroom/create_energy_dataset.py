import pandas as pd

df = pd.read_csv("classroom_sensor_dataset.csv")

# Appliance ratings
LIGHT_POWER = 20  # Watts
FAN_POWER = 30    # Watts

light_status = []
fan_status = []
energy_used = []
energy_saved = []

for _, row in df.iterrows():

    persons = row["person_count"]
    temp = row["temperature"]
    ldr = row["ldr"]

    # Light Logic
    if persons > 0 and ldr == 0:
        light = 1
    else:
        light = 0

    # Fan Logic
    if persons > 0 and temp > 30:
        fan = 1
    else:
        fan = 0

    light_status.append(light)
    fan_status.append(fan)

    # Energy consumed in this slot
    used = light * LIGHT_POWER + fan * FAN_POWER

    # Without smart system assume both ON
    without_system = LIGHT_POWER + FAN_POWER

    saved = without_system - used

    energy_used.append(used)
    energy_saved.append(saved)

df["light_status"] = light_status
df["fan_status"] = fan_status
df["energy_used"] = energy_used
df["energy_saved"] = energy_saved

df.to_csv("classroom_energy_dataset.csv", index=False)

print("Dataset Created Successfully")
print("Total Energy Used :", df["energy_used"].sum(), "Wh")
print("Total Energy Saved:", df["energy_saved"].sum(), "Wh")