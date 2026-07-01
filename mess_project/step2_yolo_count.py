from ultralytics import YOLO
import pandas as pd
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

# Read image-time CSV from Step 1
df = pd.read_csv("image_times.csv")

image_folder = "images"

counts = []

for index, row in df.iterrows():
    image_name = row["image"]
    time_value = row["time"]

    image_path = os.path.join(image_folder, image_name)

    # Detect only person class
    results = model(image_path, classes=[0])

    # Number of people detected
    person_count = len(results[0].boxes)

    counts.append(person_count)

    print(image_name, "| Time:", time_value, "| People:", person_count)

# Add people count to CSV
df["count"] = counts

# Save final counted data
df.to_csv("crowd_counts.csv", index=False)

print("\nDone. File created: crowd_counts.csv")
print(df.head())