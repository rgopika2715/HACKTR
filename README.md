# Ball Detection — Task 1

A real-time ball detector with two interchangeable methods, plus an
F1-score evaluator. Built to run on your own laptop with a webcam.

## Files
- `detect_color.py` — fast HSV color-thresholding detector (no training, tunable via live trackbars)
- `detect_yolo.py` — YOLOv8-based detector (robust, works on any ball color)
- `evaluate_f1.py` — measures Precision / Recall / F1 on a labeled test set
- `requirements.txt` — Python dependencies

---

## STEP 1 — Set up your environment (10 min)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

You need a working webcam and Python 3.9+.

## STEP 2 — Run the simple version first (5 min)

```bash
python detect_color.py
```

- Two windows open: the camera feed and the "Mask" (black/white).
- Hold up your ball. Drag the trackbar sliders (H Min/Max, S Min/Max,
  V Min/Max) until **only the ball is white** in the Mask window —
  everything else black.
- Once the ball gets a green circle + red center dot on the main
  window, press **`s`** to save that HSV range to `hsv_config.txt`
  (auto-loads next run). Press **`q`** to quit.

This is your **baseline working demo** — get this solid before touching YOLO.

## STEP 3 — Run the robust version (10–15 min, needs internet once)

```bash
python detect_yolo.py
```

First run downloads a small pretrained model (~6MB). It detects a
"sports ball" class directly — no color tuning needed, works even if
the ball is multi-colored or lighting changes.

## STEP 4 — Build a small labeled test set for F1 scoring (30–45 min)

1. Capture ~30–50 still photos of your ball in different positions,
   distances, and lighting (some frames with NO ball too — this
   tests false positives). Save into `test_images/`.
2. For each image, create a matching `.txt` file in `labels/` with
   the same base filename containing the ball's true bounding box:
   ```
   labels/img001.txt  →  120 80 210 170
   ```
   (format: `x1 y1 x2 y2` in pixels — top-left and bottom-right corners.
   Leave the file empty if that image has no ball.)
   You can get these coordinates quickly with any free annotation
   tool (e.g. `labelImg`, or even eyeballing pixel coords via `cv2.imshow` + mouse callback).
3. Run the evaluator:
   ```bash
   python evaluate_f1.py --images test_images --labels labels --method color
   python evaluate_f1.py --images test_images --labels labels --method yolo
   ```
   This prints Precision, Recall, **F1 Score**, and approximate FPS
   for each method — this is your head-to-head comparison slide.

## STEP 5 — Package results for your presentation

Your PPT/demo should show:
1. **Problem restated simply**: camera → detect ball → (x,y) position, in real time.
2. **Pipeline diagram**: Camera → Preprocess → Detect → Locate center → Overlay/Output.
3. **Two methods compared**: color-threshold (fast, simple, lighting-sensitive) vs YOLOv8 (robust, slightly heavier).
4. **Live demo** (or a screen recording as backup in case the venue lighting ruins your HSV tuning).
5. **Metrics table**: Precision / Recall / F1 / FPS for both methods on your test set.
6. **Trade-off conclusion**: which one you'd deploy and why (e.g. "color method for controlled lighting + max speed; YOLO for general-purpose robustness").

---

## How to explain each concept if judges ask

- **(x, y) position** — pixel coordinates of the ball's center in the image frame; x = across, y = down.
- **FPS** — how many frames your pipeline processes per second; measured by timing consecutive frames (`1 / (time_now - time_prev)`).
- **F1 Score** — harmonic mean of Precision and Recall; punishes methods that are only good at one of "finding balls" (recall) or "not crying wolf" (precision).
- **IoU (Intersection over Union)** — overlap ratio between your predicted box and the true box; used to decide if a detection "counts" as correct (we use IoU ≥ 0.5).
- **Why two methods** — shows you understand the classic trade-off in CV: simple/fast/fragile vs. learned/robust/heavier — a strong point to raise in Q&A even if you ship the color version as your primary demo.

## Common issues

- **No detection at all**: your HSV range trackbars are wrong for your ball's color/lighting — retune with the Mask window open.
- **Flickering detection**: increase `MIN_CONTOUR_AREA` in `detect_color.py`, or improve lighting.
- **YOLO slow on your laptop**: it's still running on CPU by default, which is fine for ~10-20 FPS on `yolov8n.pt`; that's expected and still "real-time" for this use case.
- **Camera won't open**: change `cv2.VideoCapture(0)` to `1` or `2` if you have multiple cameras/virtual cameras.
