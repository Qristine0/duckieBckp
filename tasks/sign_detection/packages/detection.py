import numpy as np

CLASS_NAMES  = {0: 'duckie', 1: 'truck', 2: 'sign'}
CLASS_COLORS = {0: (0, 215, 255), 1: (180, 100, 220), 2: (50, 205, 50)}



import cv2
import numpy as np

CLASS_NAMES = {
    0: "duckie",
    1: "truck",
    2: "sign"
}


def _mask_to_bboxes(mask):
    """Convert binary mask to bounding boxes."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200:  # filter noise
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append((x, y, x + w, y + h, area))

    return boxes


def detect_obstacles(frame_rgb: np.ndarray) -> list:
    """
    Returns:
        [((x1,y1,x2,y2), score, class_id), ...]
    """

    hsv = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2HSV)

    # --- Yellow (duckie) ---
    yellow_lower = np.array([18, 120, 120])
    yellow_upper = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # --- Blue (truck) ---
    blue_lower = np.array([90, 80, 80])
    blue_upper = np.array([130, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    detections = []

    # Duckies
    for x1, y1, x2, y2, area in _mask_to_bboxes(yellow_mask):
        score = min(1.0, area / 5000.0)
        detections.append(((x1, y1, x2, y2), score, 0))

    # Trucks
    for x1, y1, x2, y2, area in _mask_to_bboxes(blue_mask):
        score = min(1.0, area / 5000.0)
        detections.append(((x1, y1, x2, y2), score, 1))

    return detections



from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

_stop_counter = 0
_stop_latch = 0

# FIXED PARAMETERS FOR ANTI-FLICKER LOGIC
_STOP_FRAMES_REQUIRED = 1  # Stop immediately on the very first detection frame
_STOP_LATCH_FRAMES = 12  # Crucial: Hold the brakes for 12 frames even if the truck flickers out



# while moving - stop when duckie or bot right in front
def should_stop(detections: List[Detection]) -> Tuple[bool, str]:
    # print(detections)
    return False, ""


    # global _stop_counter, _stop_latch

    # stop = False
    # reason = ""

    # img_size = 416
    
    # for (x1, y1, x2, y2), score, cls_id in detections:
    #     box_h = y2 - y1
    #     box_w = x2 - x1
    #     box_cx = (x1 + x2) / 2

    #     if cls_id == 0:  # duckie
    #         if box_cx < img_size * 0.25 or box_cx > img_size * 0.70:
    #             continue
    #         if y2 > 0.45 * img_size and box_h > 18:
    #             stop = True
    #             reason = f"duckie y2={y2} h={box_h}"
    #             break

    #     elif cls_id == 1:  # truck
    #         # Relaxed width constraint to catch the truck if it wanders sideways in the frame
    #         if box_cx < img_size * 0.15 or box_cx > img_size * 0.85:
    #             continue

    #         distance_ratio = y2 / img_size

    #         # Safe distance threshold before the bumper goes underneath
    #         STOP_THRESHOLD = 0.45

    #         if distance_ratio >= STOP_THRESHOLD or box_w > (0.19 * img_size):
    #             stop = True
    #             reason = f"truck SAFE STOP y2={y2} ratio={distance_ratio:.2f} w={box_w}"
    #             break

    # if stop:
    #     _stop_counter += 1
    #     _stop_latch = _STOP_LATCH_FRAMES  # Lock the brakes down
    # else:
    #     # CRITICAL FIX: Do NOT clear the stop counter back to 0 instantly if it's flickering!
    #     # Slow down the counter clear rate so a single missed frame won't make the robot accelerate.
    #     _stop_counter = max(0, _stop_counter - 1)
    #     _stop_latch = max(0, _stop_latch - 1)

    # # If we detected a stop threat recently, or the latch timer is still active, KEEP BRAKING!
    # if _stop_counter >= _STOP_FRAMES_REQUIRED or _stop_latch > 0:
    #     return True, reason or "anti-flicker latch active"

    # return False, ""


# check for vehicles in frame; needed for intersection
def vehicle_detected(signBehaviourFSM, detections):
    for det in detections:
        bbox, score, cls_id = det
        if cls_id != signBehaviourFSM.cfg.vehicle_class_id:
            continue
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        if area >= signBehaviourFSM.cfg.vehicle_min_bbox_area:
            print(f"[SignBehavior] vehicle detected (area={area:.0f})")
            return True
    return False