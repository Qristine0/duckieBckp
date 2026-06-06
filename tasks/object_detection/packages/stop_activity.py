from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0

_DUCK_FRAMES_REQUIRED = 2
_TRUCK_FRAMES_REQUIRED = 1

_STOP_LATCH_FRAMES = 10

# Distance tuning.
# Bigger value = stops later / closer.
# Smaller value = stops earlier / farther.
_DUCK_STOP_Y2_RATIO = 0.70
_TRUCK_STOP_Y2_RATIO = 0.66


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

    if area < img_size * img_size * 0.0012:
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


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    global _duck_counter, _truck_counter, _stop_latch

    if detections is None:
        detections = []

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