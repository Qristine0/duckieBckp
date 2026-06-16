from typing import List, Tuple
from tasks.sign_detection.packages.sign_behavior_config import State


# pull working simulation visual lane servoing on another branch
# test obj detection
# checkpath will need some different handling (color may be enough)


Detection = Tuple[Tuple[int, int, int, int], float, int]

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0

_DUCK_FRAMES_REQUIRED = 2
_TRUCK_FRAMES_REQUIRED = 1

_STOP_LATCH_FRAMES = 20

# Distance tuning.
# Bigger value = stops later / closer.
# Smaller value = stops earlier / farther.
_DUCK_STOP_Y2_RATIO = 0.70
_TRUCK_STOP_Y2_RATIO = 0.45


def _valid_duck_threat(
    bbox: Tuple[int, int, int, int],
    score: float,
    img_size: int,
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    box_cx = (x1 + x2) / 2
    aspect = box_w / box_h
    area = box_w * box_h

    # Only stop for ducks in our driving path.
    if box_cx < img_size * 0.22 or box_cx > img_size * 0.78:
        return False

    if score < 0.35:
        return False

    # Reject yellow road-line false positives.
    if aspect > 2.40:
        return False

    if aspect < 0.30:
        return False

    if box_h < img_size * 0.04:
        return False

    # if area < img_size * img_size * 0.0012:
    if area < 1000:
        return False

    # Main distance rule.
    return y2 >= img_size * _DUCK_STOP_Y2_RATIO


def _valid_truck_threat(
    bbox: Tuple[int, int, int, int],
    score: float,
    img_size: int,
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    box_cx = (x1 + x2) / 2

    if score < 0.30:
        return False

    # Truck can be slightly sideways, but must still be near our path.
    if box_cx < img_size * 0.10 or box_cx > img_size * 0.90:
        return False

    # Main rule: stop when truck bottom is close enough.
    if y2 >= img_size * _TRUCK_STOP_Y2_RATIO:
        return True

    # Emergency backup:
    # Sometimes when truck gets very close, bbox bottom is unstable,
    # but the box becomes very tall/large. Stop before losing detection.
    if y2 >= img_size * 0.52 and box_h >= img_size * 0.38:
        return True

    return False


def should_stop(detections: List[Detection], img_size: int, state = None) -> Tuple[bool, str]:
    global _duck_counter, _truck_counter, _stop_latch
    if detections is None:
        detections = []

    if state == State.CHECKPATH:
        return False, ""
    
    duck_threat = False
    truck_threat = False
    reason = ""

    for bbox, score, cls_id in detections:
        if cls_id == 0:
            if _valid_duck_threat(bbox, score, img_size):
                duck_threat = True
                reason = f"duckie close enough score={score:.2f} bbox={bbox}"

        elif cls_id == 1:
            if _valid_truck_threat(bbox, score, img_size):
                truck_threat = True
                reason = f"truck close enough score={score:.2f} bbox={bbox}"

    if duck_threat:
        _duck_counter += 1
    else:
        _duck_counter = max(0, _duck_counter - 1)

    if truck_threat:
        _truck_counter += 1
    else:
        _truck_counter = max(0, _truck_counter - 1)

    confirmed_duck = _duck_counter >= _DUCK_FRAMES_REQUIRED
    confirmed_truck = _truck_counter >= _TRUCK_FRAMES_REQUIRED

    if confirmed_duck or confirmed_truck:
        _stop_latch = _STOP_LATCH_FRAMES
    else:
        _stop_latch = max(0, _stop_latch - 1)

    if _stop_latch > 0:
        return True, reason or "stop latch active"

    return False, ""



# def should_stop(detections: List[Detection], img_size: int, state = None) -> Tuple[bool, str]:
#     global _stop_counter, _stop_latch

#     if state == State.CHECKPATH:
#         return False, ""
    
#     stop = False
#     reason = ""

#     for (x1, y1, x2, y2), score, cls_id in detections:
#         box_h = y2 - y1
#         box_w = x2 - x1
#         box_cx = (x1 + x2) / 2

#         if cls_id == 0:  # duckie
#             if box_cx < img_size * 0.25 or box_cx > img_size * 0.70:
#                 continue
#             if y2 > 0.45 * img_size and box_h > 18:
#                 stop = True
#                 reason = f"duckie y2={y2} h={box_h}"
#                 break

#         elif cls_id == 1:  # truck
#             # Relaxed width constraint to catch the truck if it wanders sideways in the frame
#             if box_cx < img_size * 0.15 or box_cx > img_size * 0.85:
#                 continue

#             distance_ratio = y2 / img_size

#             # Safe distance threshold before the bumper goes underneath
#             STOP_THRESHOLD = 0.45

#             if distance_ratio >= STOP_THRESHOLD or box_w > (0.19 * img_size):
#                 stop = True
#                 reason = f"truck SAFE STOP y2={y2} ratio={distance_ratio:.2f} w={box_w}"
#                 break

#     if stop:
#         _stop_counter += 1
#         _stop_latch = _STOP_LATCH_FRAMES  # Lock the brakes down
#     else:
#         # CRITICAL FIX: Do NOT clear the stop counter back to 0 instantly if it's flickering!
#         # Slow down the counter clear rate so a single missed frame won't make the robot accelerate.
#         _stop_counter = max(0, _stop_counter - 1)
#         _stop_latch = max(0, _stop_latch - 1)

#     # If we detected a stop threat recently, or the latch timer is still active, KEEP BRAKING!
#     if _stop_counter >= _STOP_FRAMES_REQUIRED or _stop_latch > 0:
#         return True, reason or "anti-flicker latch active"

#     return False, ""
