from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture("class.mp4")

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter("output_video.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    chairs = []
    persons = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]
        conf = float(box.conf[0])

        if label == 'chair':
            chairs.append((x1, y1, x2, y2))
        elif label == 'person':
            persons.append((x1, y1, x2, y2))

    # Tampilkan kotak person
    for (x1, y1, x2, y2) in persons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Person", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan kotak chair dan cek keberadaan orang di atasnya
    for (cx1, cy1, cx2, cy2) in chairs:
        has_person = False
        for (px1, py1, px2, py2) in persons:
            # Cek apakah pusat bawah bounding box person berada dalam area chair
            person_center_x = (px1 + px2) // 2
            person_bottom_y = py2
            if cx1 <= person_center_x <= cx2 and cy1 <= person_bottom_y <= cy2:
                has_person = True
                break

        # Kotak dan label kursi
        color = (0, 255, 0) if has_person else (0, 0, 255)
        label = "Siswa hadir" if has_person else "Siswa tidak hadir"
        cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), color, 2)
        cv2.putText(frame, label, (cx1, cy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out.write(frame)

cap.release()
out.release()
print("✅ Output video saved to output_video.mp4")
