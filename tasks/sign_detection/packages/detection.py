import numpy as np
import cv2
from typing import List, Tuple

# =========================================================
# GLOBAL CONFIG
# =========================================================

IMG_WIDTH = 640
IMG_HEIGHT = 480

CLASS_NAMES = {
    0: "duckie",
    1: "truck",
    2: "sign",
}

CLASS_COLORS = {
    0: (0, 215, 255),
    1: (180, 100, 220),
    2: (50, 205, 50),
}

Detection = Tuple[Tuple[int, int, int, int], float, int]

# Stop tuning.
# Bigger value = stop later / closer.
# Smaller value = stop earlier / farther.
DUCK_STOP_Y2_RATIO = 0.70
TRUCK_STOP_Y2_RATIO = 0.60

DUCK_CONFIRM_FRAMES = 2
TRUCK_CONFIRM_FRAMES = 1
STOP_LATCH_FRAMES = 10

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0


# =========================================================
# MASK HELPERS
# =========================================================

def _clean_mask(mask: np.ndarray, open_size: int = 3, close_size: int = 5) -> np.ndarray:
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, close_kernel)

    return cleaned


def _mask_to_candidates(mask: np.ndarray, min_area: float = 120.0):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []

    for cnt in contours:
        contour_area = float(cv2.contourArea(cnt))

        if contour_area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w <= 2 or h <= 2:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        bbox_area = float(w * h)
        extent = contour_area / (bbox_area + 1e-6)

        rect = cv2.minAreaRect(cnt)
        rect_w, rect_h = rect[1]

        if rect_w <= 1 or rect_h <= 1:
            rotated_aspect = max(w, h) / (min(w, h) + 1e-6)
        else:
            rotated_aspect = max(rect_w, rect_h) / (min(rect_w, rect_h) + 1e-6)

        candidates.append({
            "bbox": (x1, y1, x2, y2),
            "area": contour_area,
            "bbox_area": bbox_area,
            "extent": extent,
            "rotated_aspect": rotated_aspect,
        })

    return candidates


# =========================================================
# DUCK FILTERING
# =========================================================

def _is_duck_candidate(candidate: dict, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = candidate["bbox"]

    box_w = x2 - x1
    box_h = y2 - y1
    area = candidate["area"]
    bbox_area = candidate["bbox_area"]
    rotated_aspect = candidate["rotated_aspect"]

    cx = (x1 + x2) / 2.0
    y2_ratio = y2 / frame_h
    aspect = box_w / (box_h + 1e-6)

    # Too high/far. We do not stop for tiny far yellow things.
    if y2_ratio < 0.30:
        return False

    # Duck must be somewhere in/near our driving path.
    if cx < frame_w * 0.12 or cx > frame_w * 0.88:
        return False

    # Too small.
    if box_h < frame_h * 0.035:
        return False

    if bbox_area < frame_w * frame_h * 0.0008:
        return False

    # Main yellow-line rejection:
    # yellow road dashes are long/thin; ducks are compact.
    if aspect > 2.20:
        return False

    if aspect < 0.35:
        return False

    if rotated_aspect > 2.60:
        return False

    # Very rectangular yellow regions are often road markings.
    # Duck silhouettes are usually less perfect rectangles.
    if candidate["extent"] > 0.92 and rotated_aspect > 1.80:
        return False

    return True


# =========================================================
# TRUCK FILTERING
# =========================================================

def _is_truck_candidate(candidate: dict, frame_w: int, frame_h: int) -> bool:
    x1, y1, x2, y2 = candidate["bbox"]

    box_w = x2 - x1
    box_h = y2 - y1
    bbox_area = candidate["bbox_area"]

    cx = (x1 + x2) / 2.0
    y2_ratio = y2 / frame_h

    # Ignore sky/top blue noise.
    if y2_ratio < 0.22:
        return False

    # Truck can be a little left or right, but not completely outside path.
    if cx < frame_w * 0.03 or cx > frame_w * 0.97:
        return False

    if box_h < frame_h * 0.045:
        return False

    if bbox_area < frame_w * frame_h * 0.0012:
        return False

    return True


# =========================================================
# DETECTION
# =========================================================

def detect_obstacles(frame_rgb: np.ndarray) -> List[Detection]:
    global IMG_WIDTH, IMG_HEIGHT

    frame_h, frame_w = frame_rgb.shape[:2]
    IMG_WIDTH = frame_w
    IMG_HEIGHT = frame_h

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # Yellow = duck + road line.
    yellow_lower = np.array([18, 80, 80], dtype=np.uint8)
    yellow_upper = np.array([38, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Blue = truck body.
    # Use wider blue range because truck can be dark/shadowed.
    blue_lower = np.array([85, 40, 20], dtype=np.uint8)
    blue_upper = np.array([140, 255, 255], dtype=np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Ignore very top area to avoid sky/sign noise.
    top_cut = int(frame_h * 0.10)
    yellow_mask[:top_cut, :] = 0
    blue_mask[:top_cut, :] = 0

    yellow_mask = _clean_mask(yellow_mask, open_size=3, close_size=5)
    blue_mask = _clean_mask(blue_mask, open_size=5, close_size=9)

    detections: List[Detection] = []

    # Duck candidates from yellow mask.
    for candidate in _mask_to_candidates(yellow_mask, min_area=100.0):
        if not _is_duck_candidate(candidate, frame_w, frame_h):
            continue

        bbox = candidate["bbox"]
        score = min(1.0, candidate["bbox_area"] / 7000.0)
        detections.append((bbox, score, 0))

    # Truck candidates from blue mask.
    for candidate in _mask_to_candidates(blue_mask, min_area=180.0):
        if not _is_truck_candidate(candidate, frame_w, frame_h):
            continue

        bbox = candidate["bbox"]
        score = min(1.0, candidate["bbox_area"] / 15000.0)
        detections.append((bbox, score, 1))

    return detections


# =========================================================
# VEHICLE DETECTION FOR INTERSECTION CHECK
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
        offset = abs(centre_x - IMG_WIDTH / 2) / IMG_WIDTH

        print(f"[SignBehavior] vehicle detected (area={area:.0f}, offset={offset:.3f})")
        return True, offset

    return False, 0.0


# =========================================================
# STOP LOGIC
# =========================================================

def _duck_close_enough(
    bbox: Tuple[int, int, int, int],
    score: float,
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    area = box_w * box_h
    cx = (x1 + x2) / 2.0
    aspect = box_w / (box_h + 1e-6)

    if score < 0.25:
        return False

    # Only stop for duck in our lane area.
    if cx < IMG_WIDTH * 0.20 or cx > IMG_WIDTH * 0.80:
        return False

    # Reject yellow road dashes again.
    if aspect > 2.20:
        return False

    if aspect < 0.35:
        return False

    if box_h < IMG_HEIGHT * 0.04:
        return False

    if area < IMG_WIDTH * IMG_HEIGHT * 0.001:
        return False

    # Main distance rule.
    return y2 >= IMG_HEIGHT * DUCK_STOP_Y2_RATIO


def _truck_close_enough(
    bbox: Tuple[int, int, int, int],
    score: float,
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1
    cx = (x1 + x2) / 2.0

    if score < 0.20:
        return False

    if cx < IMG_WIDTH * 0.06 or cx > IMG_WIDTH * 0.94:
        return False

    # Main distance rule.
    if y2 >= IMG_HEIGHT * TRUCK_STOP_Y2_RATIO:
        return True

    # Backup: if the truck suddenly becomes huge, stop even if y2 is unstable.
    if y2 >= IMG_HEIGHT * 0.48 and box_h >= IMG_HEIGHT * 0.34:
        return True

    return False


def should_stop(
    detections: List[Detection],
) -> Tuple[bool, str]:
    global _duck_counter, _truck_counter, _stop_latch
    #
    # if detections is None:
    #     detections = []
    #
    # duck_threat = False
    # truck_threat = False
    # reason = ""
    #
    # for bbox, score, cls_id in detections:
    #     if cls_id == 0:
    #         if _duck_close_enough(bbox, score):
    #             duck_threat = True
    #             reason = f"duckie close enough score={score:.2f} bbox={bbox}"
    #
    #     elif cls_id == 1:
    #         if _truck_close_enough(bbox, score):
    #             truck_threat = True
    #             reason = f"truck close enough score={score:.2f} bbox={bbox}"
    #
    # if duck_threat:
    #     _duck_counter += 1
    # else:
    #     _duck_counter = max(0, _duck_counter - 1)
    #
    # if truck_threat:
    #     _truck_counter += 1
    # else:
    #     _truck_counter = max(0, _truck_counter - 1)
    #
    # confirmed_duck = _duck_counter >= DUCK_CONFIRM_FRAMES
    # confirmed_truck = _truck_counter >= TRUCK_CONFIRM_FRAMES
    #
    # if confirmed_duck or confirmed_truck:
    #     _stop_latch = STOP_LATCH_FRAMES
    # else:
    #     _stop_latch = max(0, _stop_latch - 1)
    #
    # if _stop_latch > 0:
    #     return True, reason or "stop latch active"

    return False, ""

def reset_detection_state():
    global _duck_counter, _truck_counter, _stop_latch

    try:
        _duck_counter = 0
    except NameError:
        pass

    try:
        _truck_counter = 0
    except NameError:
        pass

    try:
        _stop_latch = 0
    except NameError:
        pass

    print("[Detection] state reset")