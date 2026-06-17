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

_STOP_LATCH_FRAMES = 3

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
    if box_cx < img_size * 0.25 or box_cx > img_size * 0.7:
        return False

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
    if state == State.MOVING and (box_cx < img_size * 0.45 or box_cx > img_size * 0.65):
        return False
    
    if state != State.MOVING and (box_cx < img_size * 0.1 or box_cx > img_size * 0.9):
        return False
    
    print("---")
    print(img_size)
    print(box_cx)
    area = box_w * box_h
    
    distance_ratio = y2 / img_size
    width_ratio = box_w / img_size

    close_by_y = distance_ratio >= 0.45
    close_by_width = width_ratio >= 0.19

    if close_by_y and (area > 4000 or close_by_width):
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
            if _valid_truck_threat(bbox, score, img_size, state):
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





# Horizontal: accept ducks/trucks in central 80% of frame width (cx 0.10–0.90).
# Widen if robot veers and ducks appear near edges.
CENTERED_MIN = 0.10
CENTERED_MAX = 0.90

# Vertical: bottom edge of bbox must be at or below this fraction of frame height.
# From logs, duck on road peaks at cy_bottom ~0.46 before passing under camera.
# 0.35 catches it with reaction time. Lower toward 0.25 for earlier trigger.
# Raise toward 0.50 only if getting false positives from distant objects.
LOWER_ZONE_THRESHOLD = 0.35

# Minimum bbox area relative to frame area.
# Small Duckietown duckies at trigger distance are ~0.001–0.004.
# Raise if stopping for far-away/tiny false detections.
MIN_AREA_FRACTION = 0.0005


class_names = {0: 'duckie', 1: 'truck', 2: 'sign'}

# def should_stop(
#     detections: List[Detection],
#     img_size: int, 
#     state = None
# ) -> Tuple[bool, str]:
#     if not detections:
#         return False, ''

#     frame_w = 640
#     frame_h = 480
#     frame_area = max(1, frame_w * frame_h)

#     for bbox, score, cls_id in detections:
#         if cls_id not in (0, 1):
#             continue

#         x1, y1, x2, y2 = bbox
#         bw = max(0, x2 - x1)
#         bh = max(0, y2 - y1)

#         cx_norm   = ((x1 + x2) / 2.0) / frame_w
#         cy_bottom = y2 / frame_h
#         area_frac = (bw * bh) / frame_area

#         centered     = CENTERED_MIN <= cx_norm <= CENTERED_MAX
#         in_lower     = cy_bottom >= LOWER_ZONE_THRESHOLD
#         close_enough = area_frac >= MIN_AREA_FRACTION

#         if centered and in_lower and close_enough:
#             name = class_names.get(cls_id, str(cls_id))
#             return True, (
#                 f'{name} in lower zone: score={score:.2f}, '
#                 f'area={area_frac:.4f}, cy_bottom={cy_bottom:.2f}'
#             )

#     return False, ''
