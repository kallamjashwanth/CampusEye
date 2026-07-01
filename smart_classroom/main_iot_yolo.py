from ultralytics import YOLO
import cv2
import serial
import time

esp = serial.Serial("COM3", 115200, timeout=1)
time.sleep(2)

model = YOLO("yolov8n.pt")

video_path = "smart_classroom/videos/students_entering.mp4"
output_path = "smart_classroom/videos/iot_output_students_entering.mp4"

TEMP_THRESHOLD = 24

last_temp = 0
last_hum = 0
last_ldr = 0

last_light_status = None
last_fan_status = None


def read_sensors():
    global last_temp, last_hum, last_ldr

    while esp.in_waiting:
        line = esp.readline().decode(errors="ignore").strip()

        if "TEMP:" in line:
            try:
                parts = line.split(",")

                temp_value = parts[0].split(":")[1]
                hum_value = parts[1].split(":")[1]
                ldr_value = parts[2].split(":")[1]

                if temp_value != "ERROR":
                    last_temp = float(temp_value)

                if hum_value != "ERROR":
                    last_hum = float(hum_value)

                last_ldr = int(ldr_value)

            except:
                pass

    return last_temp, last_hum, last_ldr


def send_command(cmd):
    esp.write((cmd + "\n").encode())


cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.45)

    person_count = 0

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]

        if class_name == "person":
            person_count += 1

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.putText(
                frame,
                "person",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    temp, hum, ldr = read_sensors()

    if person_count == 0:
        light_status = "OFF"
        fan_status = "OFF"
    else:
        # For your LDR: 0 means dark, 1 means bright
        if ldr == 0:
            light_status = "ON"
        else:
            light_status = "OFF"

        if temp > TEMP_THRESHOLD:
            fan_status = "ON"
        else:
            fan_status = "OFF"

    if light_status != last_light_status:
        send_command("LIGHT_ON" if light_status == "ON" else "LIGHT_OFF")
        last_light_status = light_status

    if fan_status != last_fan_status:
        send_command("FAN_ON" if fan_status == "ON" else "FAN_OFF")
        last_fan_status = fan_status

    cv2.putText(frame, f"Persons: {person_count}", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Temp: {temp} C | Hum: {hum}% | LDR: {ldr}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.putText(frame, f"Light: {light_status} | Fan: {fan_status}", (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    print("Persons:", person_count, "Temp:", temp, "LDR:", ldr,
          "Light:", light_status, "Fan:", fan_status)

    out.write(frame)

cap.release()
out.release()

send_command("ALL_OFF")
esp.close()

print("Done. Output saved:", output_path)