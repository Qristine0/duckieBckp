from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

_stop_counter = 0
_stop_latch = 0
_STOP_FRAMES_REQUIRED = 2
_STOP_LATCH_FRAMES = 3


def should_stop(detections: List[Detection], img_size: int) -> Tuple[bool, str]:
    global _stop_counter, _stop_latch

    stop = False
    reason = ""

    for (x1, y1, x2, y2), score, cls_id in detections:
        box_h = y2 - y1
        box_w = x2 - x1
        box_cx = (x1 + x2) / 2

        if cls_id == 0:  # duckie
            if box_cx < img_size * 0.25 or box_cx > img_size * 0.70:
                continue
            if y2 > 0.45 * img_size and box_h > 18:
                stop = True
                reason = f"duckie y2={y2} h={box_h} cx={box_cx:.0f}"
                break

        # elif cls_id == 1:  # truck
        #     # trucks are large so require stricter conditions:
        #     # must be centered, very close (low y2), and a large box
        #     if box_cx < img_size * 0.20 or box_cx > img_size * 0.80:
        #         continue
        #     if y2 > 0.60 * img_size and box_w > 0.15 * img_size:
        #         stop = True
        #         reason = f"truck y2={y2} w={box_w} cx={box_cx:.0f}"
        #         break

    if stop:
        _stop_counter += 1
        _stop_latch = _STOP_LATCH_FRAMES
    else:
        _stop_counter = 0
        _stop_latch = max(0, _stop_latch - 1)

    if _stop_counter >= _STOP_FRAMES_REQUIRED or _stop_latch > 0:
        return True, reason or "latch"

    return False, ""