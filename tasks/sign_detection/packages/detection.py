import numpy as np
import cv2
from typing import List, Tuple

# =========================================================
# GLOBAL CONFIG
# =========================================================
IMG_WIDTH  = 640
IMG_HEIGHT = 480

CLASS_NAMES  = {0: 'duckie', 1: 'truck', 2: 'sign'}
CLASS_COLORS = {0: (0, 215, 255), 1: (180, 100, 220), 2: (50, 205, 50)}

# =========================================================
# THRESHOLDS — sign check / red-line stop / turning
# =========================================================
EMERGENCY_STOP_AREA_THRESHOLD = 30000
MIN_OBJECT_HEIGHT              = 100
MIN_HEIGHT_WIDTH_RATIO         = 0.65
EMERGENCY_CX_MIN               = 0.2  * IMG_WIDTH
EMERGENCY_CX_MAX               = 0.8  * IMG_WIDTH
EMERGENCY_Y2_MIN               = 0.50 * IMG_HEIGHT


# =========================================================
# HELPERS
# =========================================================
def _mask_to_bboxes(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h, area))
    return boxes


# =========================================================
# DETECTION
# =========================================================
def detect_obstacles(frame_rgb: np.ndarray) -> list:
    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    yellow_lower = np.array([18, 120, 120])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask  = cv2.inRange(hsv, yellow_lower, yellow_upper)

    blue_lower   = np.array([90,  80,  80])
    blue_upper   = np.array([130, 255, 255])
    blue_mask    = cv2.inRange(hsv, blue_lower, blue_upper)

    detections = []
    for x1, y1, x2, y2, area in _mask_to_bboxes(yellow_mask):
        detections.append(((x1, y1, x2, y2), min(1.0, area / 5000.0), 0))
    for x1, y1, x2, y2, area in _mask_to_bboxes(blue_mask):
        detections.append(((x1, y1, x2, y2), min(1.0, area / 5000.0), 1))

    return detections


# =========================================================
# VEHICLE DETECTION (intersection crossing check)
# =========================================================
vehicle_min_bbox_area = 800

def vehicle_detected(detections):
    for (x1, y1, x2, y2), score, cls_id in detections:
        if cls_id != 1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < vehicle_min_bbox_area:
            continue
        centre_x = (x1 + x2) / 2.0
        offset   = abs(centre_x - IMG_WIDTH / 2) / IMG_WIDTH
        print(f"[SignBehavior] vehicle detected (area={area:.0f}, offset={offset:.3f})")
        return True, offset
    return False, 0.0


# =========================================================
# EMERGENCY STOP — strict thresholds for intersection context
# (sign check / red-line stop / left+right turns)
# =========================================================
def should_stop(
    detections: List[Tuple[Tuple[int, int, int, int], float, int]]
) -> Tuple[bool, str]:

    for (x1, y1, x2, y2), score, cls_id in detections:

        if cls_id not in (0, 1):
            continue

        area   = (x2 - x1) * (y2 - y1)
        height = y2 - y1
        width  = x2 - x1
        cx     = (x1 + x2) / 2.0
        cy     = (y1 + y2) / 2.0

        # ── DUCKIE ──────────────────────────────────────────
        if cls_id == 0:
            if area   < 40000:                                        continue
            if height < MIN_OBJECT_HEIGHT:                            continue
            if height / (width + 1e-6) < MIN_HEIGHT_WIDTH_RATIO:     continue
            if cx < EMERGENCY_CX_MIN or cx > EMERGENCY_CX_MAX:       continue
            if y2 < EMERGENCY_Y2_MIN:                                 continue

            # print(f"[ObstacleStop] DUCKIE STOP (area={area:.0f})")
            # return True, "duckie too close"

        # ── TRUCK ────────────────────────────────────────────
        if cls_id == 1:
            if area   < 3000:                                         continue
            if height < 15:                                           continue
            if cy     < IMG_HEIGHT * 0.25:                            continue
            if cx < 0.08 * IMG_WIDTH or cx > 0.92 * IMG_WIDTH:       continue

            print(f"[ObstacleStop] TRUCK STOP (area={area:.0f}, h={height:.0f})")
            return True, "truck too close"

    return False, ""