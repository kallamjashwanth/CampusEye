import serial
import time
import pandas as pd
from datetime import datetime

COM_PORT = "COM3"   # change this
READINGS_PER_SLOT = 5

esp = serial.Serial(COM_PORT, 115200, timeout=1)
time.sleep(2)

def read_ultrasonic_average():
    distances = []
    fills = []

    while len(fills) < READINGS_PER_SLOT:
        if esp.in_waiting:
            line = esp.readline().decode(errors="ignore").strip()

            if "DISTANCE:" in line and "FILL:" in line:
                try:
                    parts = line.split(",")
                    distance = float(parts[0].split(":")[1])
                    fill = float(parts[1].split(":")[1])

                    distances.append(distance)
                    fills.append(fill)

                    print("Reading:", distance, fill)

                except:
                    pass

    avg_distance = sum(distances) / len(distances)
    avg_fill = sum(fills) / len(fills)

    return round(avg_distance, 2), round(avg_fill, 2)


data = []

bins = {
    "Mess Bin": "Mess Area",
    "Classroom Bin": "Classroom Area",
    "Hostel Bin": "Hostel Area"
}

days = int(input("Enter number of days to simulate/collect: "))

time_slots = ["Morning", "Afternoon", "Evening", "Night"]

for day in range(1, days + 1):
    print(f"\n===== DAY {day} =====")

    for slot in time_slots:
        print(f"\n--- {slot} ---")

        for bin_name, location in bins.items():
            input(f"\nPlace sensor on {bin_name} ({location}) for Day {day} - {slot}, then press ENTER...")

            avg_distance, avg_fill = read_ultrasonic_average()

            data.append({
                "day": day,
                "time_slot": slot,
                "bin_name": bin_name,
                "location": location,
                "distance_cm": avg_distance,
                "fill_percentage": avg_fill,
                "timestamp": datetime.now()
            })

            print(f"Saved: Day {day}, {slot}, {bin_name}, Fill={avg_fill}%")

df = pd.DataFrame(data)
df.to_csv("dustbin_daywise_dataset.csv", index=False)

esp.close()

print("\nDataset saved as dustbin_daywise_dataset.csv")