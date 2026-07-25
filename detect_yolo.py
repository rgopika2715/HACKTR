"""
BALL DETECTION - METHOD B: YOLOv8 (deep learning object detector)
-------------------------------------------------------------------
More robust than color thresholding: works regardless of ball color,
lighting, or background, because it's a neural network trained on
real images (COCO dataset includes a "sports ball" class, id 32).

HOW IT WORKS (in plain English):
1. Grab a frame from the webcam.
2. Feed it to a pretrained YOLOv8 model (downloaded automatically
   the first time you run this - needs internet once).
3. YOLO returns a list of detected objects: class, confidence,
   bounding box (x1, y1, x2, y2).
4. We filter to keep only the "sports ball" class.
5. Draw the box, compute the center (x, y), overlay FPS.

WHY TWO METHODS?
Present both in your PPT: color-thresholding is fast & simple but
fails if lighting changes or ball color isn't unique. YOLO is
heavier but far more robust - a good "compare & justify" story for
judges, and it directly demonstrates the F1-score idea because YOLO
gives you a confidence score per detection.

RUN:
    python detect_yolo.py

First run downloads yolov8n.pt (~6MB, the smallest/fastest variant).
"""

import cv2
import time
from ultralytics import YOLO

SPORTS_BALL_CLASS_ID = 32   # COCO class index for "sports ball"
CONF_THRESHOLD = 0.35


def main():
    print("Loading YOLOv8 model (first run downloads weights)...")
    model = YOLO("yolov8n.pt")  # nano version = fastest, good for real-time

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam.")
        return

    prev_time = time.time()
    fps = 0.0

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        ball_found = False
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if cls_id != SPORTS_BALL_CLASS_ID:
                continue
            ball_found = True
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, f"Ball {conf:.2f} ({cx},{cy})",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else fps
        prev_time = curr_time

        status = "Ball Detected" if ball_found else "No Ball"
        color = (0, 255, 0) if ball_found else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Ball Detection - YOLOv8 Method", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
