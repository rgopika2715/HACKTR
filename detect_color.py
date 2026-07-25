"""
BALL DETECTION - METHOD A: Color / HSV Thresholding
----------------------------------------------------
Fast, lightweight, no training data needed. Best for a ball of a
known, fairly uniform color (e.g. an orange/red/yellow/green ball).

HOW IT WORKS (in plain English):
1. Grab a frame from the webcam.
2. Convert it from BGR (blue-green-red) color space to HSV
   (Hue-Saturation-Value). HSV is much easier to threshold by
   color because "hue" isolates the color itself from lighting.
3. Create a mask: pixels inside our chosen HSV range become white,
   everything else becomes black.
4. Clean up the mask (remove noise) with morphological operations.
5. Find contours (blobs) in the mask.
6. Pick the largest contour above a minimum size -> that's the ball.
7. Fit a circle / bounding box around it, compute its center (x, y).
8. Overlay bounding box, center point, coordinates, and FPS on screen.

RUN:
    python detect_color.py

CONTROLS:
    - A "Trackbars" window lets you tune the HSV range live until
      only the ball is detected in the mask window.
    - Press 'q' to quit.
    - Press 's' to print+save the current HSV values to hsv_config.txt
"""

import cv2
import numpy as np
import time
import json
import os

CONFIG_FILE = "hsv_config.txt"
MIN_RADIUS = 10          # ignore tiny blobs (noise)
MIN_CONTOUR_AREA = 300    # ignore small contours


def nothing(x):
    pass


def load_default_hsv():
    """Default range tuned for an ORANGE ball. Change via trackbars."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {"h_min": 5, "h_max": 20, "s_min": 120, "s_max": 255,
            "v_min": 100, "v_max": 255}


def create_trackbars(hsv_defaults):
    cv2.namedWindow("Trackbars")
    cv2.resizeWindow("Trackbars", 400, 300)
    cv2.createTrackbar("H Min", "Trackbars", hsv_defaults["h_min"], 179, nothing)
    cv2.createTrackbar("H Max", "Trackbars", hsv_defaults["h_max"], 179, nothing)
    cv2.createTrackbar("S Min", "Trackbars", hsv_defaults["s_min"], 255, nothing)
    cv2.createTrackbar("S Max", "Trackbars", hsv_defaults["s_max"], 255, nothing)
    cv2.createTrackbar("V Min", "Trackbars", hsv_defaults["v_min"], 255, nothing)
    cv2.createTrackbar("V Max", "Trackbars", hsv_defaults["v_max"], 255, nothing)


def get_trackbar_values():
    h_min = cv2.getTrackbarPos("H Min", "Trackbars")
    h_max = cv2.getTrackbarPos("H Max", "Trackbars")
    s_min = cv2.getTrackbarPos("S Min", "Trackbars")
    s_max = cv2.getTrackbarPos("S Max", "Trackbars")
    v_min = cv2.getTrackbarPos("V Min", "Trackbars")
    v_max = cv2.getTrackbarPos("V Max", "Trackbars")
    return h_min, h_max, s_min, s_max, v_min, v_max


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: could not open webcam. Check camera index/permissions.")
        return

    hsv_defaults = load_default_hsv()
    create_trackbars(hsv_defaults)

    prev_time = time.time()
    fps = 0.0

    print("Press 'q' to quit, 's' to save current HSV range.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)  # mirror, feels natural
        blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        h_min, h_max, s_min, s_max, v_min, v_max = get_trackbar_values()
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Clean the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        ball_found = False
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area > MIN_CONTOUR_AREA:
                (x, y), radius = cv2.minEnclosingCircle(largest)
                if radius > MIN_RADIUS:
                    ball_found = True
                    center = (int(x), int(y))
                    cv2.circle(frame, center, int(radius), (0, 255, 0), 2)
                    cv2.circle(frame, center, 4, (0, 0, 255), -1)
                    cv2.putText(frame, f"Ball ({center[0]},{center[1]})",
                                (center[0] - 60, center[1] - int(radius) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # FPS calculation
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if curr_time != prev_time else fps
        prev_time = curr_time

        status = "Ball Detected" if ball_found else "No Ball"
        color = (0, 255, 0) if ball_found else (0, 0, 255)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, color, 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("Ball Detection - Color Method", frame)
        cv2.imshow("Mask", mask)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            values = {"h_min": h_min, "h_max": h_max, "s_min": s_min,
                      "s_max": s_max, "v_min": v_min, "v_max": v_max}
            with open(CONFIG_FILE, "w") as f:
                json.dump(values, f, indent=2)
            print(f"Saved HSV config: {values}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
