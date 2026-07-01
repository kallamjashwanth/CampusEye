from ultralytics import YOLO
import os
import pandas as pd
import serial
import time

COM_PORT = "COM3"  
PHOTO_FOLDER = "images"

model = YOLO("yolov8n.pt")
esp = serial.Serial(COM_PORT, 115200, timeout=1)
time.sleep(2)

def read_sensor():
    temp, hum, ldr = None, None, None

    start = time.time()
    while time.time() - start < 3:
        if esp.in_waiting:
            line = esp.readline().decode(errors="ignore").strip()
            if "TEMP:" in line:
                try:
                    parts = line.split(",")
                    temp = float(parts[0].split(":")[1])
                    hum = float(parts[1].split(":")[1])
                    ldr = int(parts[2].split(":")[1])
                    return temp, hum, ldr
                except:
                    pass

    return 0, 0, 0

data = []

for img_name in sorted(os.listdir(PHOTO_FOLDER)):
    if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
        img_path = os.path.join(PHOTO_FOLDER, img_name)

        results = model(img_path, conf=0.45)

        person_count = 0
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if model.names[cls_id] == "person":
                person_count += 1

        temp, hum, ldr = read_sensor()

        current_time = pd.Timestamp.now()
        hour = current_time.hour
        minute = current_time.minute

        data.append({
            "image_name": img_name,
            "time": current_time,
            "hour": hour,
            "minute": minute,
            "person_count": person_count,
            "temperature": temp,
            "humidity": hum,
            "ldr": ldr
        })

        print(img_name, person_count, temp, hum, ldr)

df = pd.DataFrame(data)
df.to_csv("classroom_sensor_dataset.csv", index=False)

esp.close()

print("Created classroom_sensor_dataset.csv")