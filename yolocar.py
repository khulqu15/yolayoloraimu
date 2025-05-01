from ultralytics import YOLO
import cv2
import json
import os
import uuid
import math
import numpy as np

model = YOLO("yolov8n.pt")

vehicle_classes = ['car', 'truck', 'bus']
class_colors = {
    'car': (0, 255, 0),
    'truck': (255, 0, 0),
    'bus': (0, 0, 255),
}

cap = cv2.VideoCapture("mobil.mp4")
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output_mobil.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_index = 0
detected_data = []
class_counts = {cls: 0 for cls in vehicle_classes}
tracked_objects = {cls: [] for cls in vehicle_classes}

DISTANCE_THRESHOLD = 50

polygon_pts = np.array([
    [1500, 1060],
    [1540, 560],
    [880, 530],
    [0, 1060],
], np.int32)

def is_duplicate(label, x, y):
    for (px, py) in tracked_objects[label]:
        distance = math.sqrt((x - px) ** 2 + (y - py) ** 2)
        if distance < DISTANCE_THRESHOLD:
            return True
    return False

def is_inside_roi(x, y, polygon):
    return cv2.pointPolygonTest(polygon, (x, y), False) >= 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    total_seconds = int(frame_index / fps)
    frame_minute = total_seconds // 60
    frame_second = total_seconds % 60
    cv2.polylines(frame, [polygon_pts], isClosed=True, color=(0, 0, 255), thickness=5)

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])
        center_x = (x1 + x2) // 2
        bottom_y = y2

        if label in vehicle_classes and is_inside_roi(center_x, bottom_y, polygon_pts):

            if not is_duplicate(label, center_x, bottom_y):
                obj_uuid = str(uuid.uuid4())

                detected_data.append({
                    "uuid": obj_uuid,
                    "minute": frame_minute,
                    "second": frame_second,
                    "location": [center_x, bottom_y],
                    "class": label
                })

                class_counts[label] += 1
                tracked_objects[label].append((center_x, bottom_y))

                img_folder = f"images/{label}"
                os.makedirs(img_folder, exist_ok=True)
                cropped = frame[y1:y2, x1:x2]
                filename = f"{img_folder}/{obj_uuid}.jpg"
                cv2.imwrite(filename, cropped)

            color = class_colors.get(label, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


    # roi_mask = frame.copy()
    # cv2.imshow("ROI Area", roi_mask)
    
    out.write(frame)
    frame_index += 1

cap.release()
out.release()

with open("vehicle_detection.json", "w") as f:
    json.dump({
        "detections": detected_data,
        "count_per_class": class_counts
    }, f, indent=2)

print("Video disimpan ke 'output_mobil.mp4'")
print("Data kendaraan (tanpa duplikasi) ke 'vehicle_detection.json'")
print("Gambar per object disimpan di folder /images/{class}/")
