from ultralytics import YOLO
import pandas as pd
import os

model = YOLO("yolov8n.pt")

df = pd.read_csv("image_times.csv")
image_folder = "images"

yolo_counts = []
methods = []

for index, row in df.iterrows():
    image_path = os.path.join(image_folder, row["image"])

    results = model(image_path, classes=[0])
    count = len(results[0].boxes)

    yolo_counts.append(count)

    # If YOLO count is 15 or more, we assume image may be dense
    if count >= 15:
        methods.append("CSRNet")
    else:
        methods.append("YOLO")

    print(row["image"], "| YOLO count:", count, "| Method:", methods[-1])

df["yolo_count"] = yolo_counts
df["method"] = methods

df.to_csv("hybrid_step1.csv", index=False)

print("\nFile created: hybrid_step1.csv")