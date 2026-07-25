"""
F1 SCORE EVALUATION
--------------------
This script measures how ACCURATE your detector is on a folder of
still test images (not live video) - exactly what judges mean by
"F1 Score" for the ball detector.

CONCEPTS (plain English):
- Ground truth: the REAL location of the ball in each test image,
  which YOU label once ahead of time (see labels/ format below).
- Prediction: where YOUR detector THINKS the ball is.
- IoU (Intersection over Union): how much the predicted box overlaps
  the true box. IoU=1 means perfect overlap, IoU=0 means no overlap.
- A prediction counts as a TRUE POSITIVE (TP) if IoU >= 0.5 with a
  ground-truth box.
- FALSE POSITIVE (FP): detector found a "ball" where there isn't one
  (or box doesn't overlap enough).
- FALSE NEGATIVE (FN): there IS a ball in the image but detector
  missed it.
- Precision = TP / (TP + FP)   -> "of what I found, how much was right?"
- Recall    = TP / (TP + FN)   -> "of all real balls, how many did I find?"
- F1 Score  = 2 * (Precision * Recall) / (Precision + Recall)
              -> balances both into a single number.

DATASET FORMAT EXPECTED:
    test_images/
        img001.jpg
        img002.jpg
        ...
    labels/
        img001.txt   <- one line: x1 y1 x2 y2   (ground truth box, pixels)
        img002.txt   <- empty file if no ball present in that image
        ...

RUN:
    python evaluate_f1.py --images test_images --labels labels --method color
    python evaluate_f1.py --images test_images --labels labels --method yolo
"""

import os
import argparse
import cv2
import numpy as np
import time


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    if inter_area == 0:
        return 0.0

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter_area / float(boxA_area + boxB_area - inter_area)


def detect_color(frame, hsv_range):
    lower, upper = hsv_range
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 300:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    return [x, y, x + w, y + h]


def load_ground_truth(label_path):
    if not os.path.exists(label_path) or os.path.getsize(label_path) == 0:
        return None
    with open(label_path, "r") as f:
        line = f.readline().strip()
        if not line:
            return None
        x1, y1, x2, y2 = map(int, line.split())
        return [x1, y1, x2, y2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True, help="folder of test images")
    parser.add_argument("--labels", required=True, help="folder of ground-truth txt files")
    parser.add_argument("--method", choices=["color", "yolo"], default="color")
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    args = parser.parse_args()

    if args.method == "yolo":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        SPORTS_BALL_ID = 32

    hsv_range = (np.array([5, 120, 100]), np.array([20, 255, 255]))  # tune as needed

    tp, fp, fn = 0, 0, 0
    frame_times = []

    image_files = sorted([f for f in os.listdir(args.images)
                           if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    for fname in image_files:
        img_path = os.path.join(args.images, fname)
        label_path = os.path.join(args.labels, os.path.splitext(fname)[0] + ".txt")

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        gt_box = load_ground_truth(label_path)

        start = time.time()
        if args.method == "color":
            pred_box = detect_color(frame, hsv_range)
        else:
            results = model.predict(frame, conf=0.35, verbose=False)[0]
            pred_box = None
            best_conf = 0
            for box in results.boxes:
                if int(box.cls[0]) == SPORTS_BALL_ID and float(box.conf[0]) > best_conf:
                    best_conf = float(box.conf[0])
                    pred_box = list(map(int, box.xyxy[0]))
        frame_times.append(time.time() - start)

        if gt_box is None and pred_box is None:
            continue  # true negative, not counted in F1
        elif gt_box is None and pred_box is not None:
            fp += 1
        elif gt_box is not None and pred_box is None:
            fn += 1
        else:
            iou = compute_iou(gt_box, pred_box)
            if iou >= args.iou_thresh:
                tp += 1
            else:
                fp += 1
                fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    avg_fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0

    print("\n===== EVALUATION RESULTS =====")
    print(f"Images evaluated : {len(image_files)}")
    print(f"True Positives   : {tp}")
    print(f"False Positives  : {fp}")
    print(f"False Negatives  : {fn}")
    print(f"Precision        : {precision:.3f}")
    print(f"Recall           : {recall:.3f}")
    print(f"F1 Score         : {f1:.3f}")
    print(f"Avg FPS (approx) : {avg_fps:.1f}")


if __name__ == "__main__":
    main()
