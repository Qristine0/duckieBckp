from typing import List, Tuple
from tasks.sign_detection.packages.sign_behavior_config import State



Detection = Tuple[Tuple[int, int, int, int], float, int]

_duck_counter = 0
_truck_counter = 0
_stop_latch = 0

_DUCK_FRAMES_REQUIRED = 2
_TRUCK_FRAMES_REQUIRED = 2

_STOP_LATCH_FRAMES = 40

# Distance tuning.
# Bigger value = stops later / closer.
# Smaller value = stops earlier / farther.
_DUCK_STOP_Y2_RATIO = 0.70
_TRUCK_STOP_Y2_RATIO = 0.75

# self.img_size       = cfg.get('img_size',       416)


def _valid_duck_threat(
    bbox: Tuple[int, int, int, int],
    score: float,
    img_size: int,
    state = None
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
    if state != State.TURNING and (box_cx < img_size * 0.25 or box_cx > img_size * 0.7):
        return False
    # if state == State.TURNING and (box_cx < img_size * 0.1 or box_cx > img_size * 0.9):
    #     return False

    if score < 0.5:
        return False

    # Reject yellow road-line false positives.
    if aspect > 2.40:
        return False

    if aspect < 0.30:
        return False

    # if box_h < img_size * 0.04:   # approx 17
    if box_h < 18:
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
    state = None
) -> bool:
    x1, y1, x2, y2 = bbox

    box_w = x2 - x1
    box_h = y2 - y1

    if box_w <= 0 or box_h <= 0:
        return False

    box_cx = (x1 + x2) / 2

    if score < 0.60:
        return False

    # Truck can be slightly sideways, but must still be near our path.
    if state == State.TURNING and (box_cx < img_size * 0.45 or box_cx > img_size * 0.65):
        return False
    
    if state != State.TURNING and (box_cx < img_size * 0.1 or box_cx > img_size * 0.9):
        return False
    
    area = box_w * box_h
    
    distance_ratio = y2 / img_size
    width_ratio = box_w / img_size

    close_by_y = distance_ratio >= 0.65
    close_by_width = width_ratio >= 0.2
    if close_by_y and (area > 4000 or close_by_width):
        return True


    return False


def should_stop(detections: List[Detection], img_size: int, state = None) -> Tuple[bool, str]:
    global _duck_counter, _truck_counter, _stop_latch
    if detections is None:
        detections = []

    if state == State.CHECKPATH or state == State.STOPPED or state == State.SLOWING:
        return False, ""
    
    duck_threat = False
    truck_threat = False
    reason = ""

    for bbox, score, cls_id in detections:
        if cls_id == 0:
            if _valid_duck_threat(bbox, score, img_size, state):
                duck_threat = True
                reason = f"duckie close enough score={score:.2f} bbox={bbox}"

        elif cls_id == 1:
            if _valid_truck_threat(bbox, score, img_size, state):
                truck_threat = True
                reason = f"truck close enough score={score:.2f} bbox={bbox}"

    if duck_threat:
        _duck_counter = min(_duck_counter + 1, _DUCK_FRAMES_REQUIRED)
    else:
        _duck_counter = max(0, _duck_counter - 1)

    if truck_threat:
        _truck_counter = min(_truck_counter + 1, _TRUCK_FRAMES_REQUIRED)
    else:
        _truck_counter = max(0, _truck_counter - 1)
        

    confirmed_duck = ((state == State.TURNING or state == State.EXITING or state == State.POST_STOP) and _duck_counter >= 1) or _duck_counter >= _DUCK_FRAMES_REQUIRED
    confirmed_truck = ((state == State.TURNING or state == State.EXITING or state == State.POST_STOP) and  _truck_counter >= 1) or _truck_counter >= _TRUCK_FRAMES_REQUIRED

    if confirmed_duck or confirmed_truck:
        _stop_latch = _STOP_LATCH_FRAMES
    else:
        _stop_latch = max(0, _stop_latch - 1)

    if _stop_latch > 0:
        return True, reason or f"stop latch active {_stop_latch}"

    return False, ""
