from ultralytics import YOLO
import cv2
import os
import pandas as pd

model = YOLO("yolov8n.pt")

photo_folder = "images"
data = []

for img_name in os.listdir(photo_folder):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(photo_folder, img_name)

        results = model(img_path, conf=0.45)

        person_count = 0

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            if class_name == "person":
                person_count += 1

        data.append({
            "image_name": img_name,
            "person_count": person_count
        })

df = pd.DataFrame(data)
df.to_csv("classroom_occupancy_raw.csv", index=False)

print("CSV created: classroom_occupancy_raw.csv")
print(df.head())