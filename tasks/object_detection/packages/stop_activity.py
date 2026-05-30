from typing import List, Tuple

Detection = Tuple[Tuple[int, int, int, int], float, int]

_stop_counter = 0
_stop_latch = 0

# FIXED PARAMETERS FOR ANTI-FLICKER LOGIC
_STOP_FRAMES_REQUIRED = 1  # Stop immediately on the very first detection frame
_STOP_LATCH_FRAMES = 12  # Crucial: Hold the brakes for 12 frames even if the truck flickers out


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
                reason = f"duckie y2={y2} h={box_h}"
                break

        elif cls_id == 1:  # truck
            # Relaxed width constraint to catch the truck if it wanders sideways in the frame
            if box_cx < img_size * 0.15 or box_cx > img_size * 0.85:
                continue

            distance_ratio = y2 / img_size

            # Safe distance threshold before the bumper goes underneath
            STOP_THRESHOLD = 0.45

            if distance_ratio >= STOP_THRESHOLD or box_w > (0.19 * img_size):
                stop = True
                reason = f"truck SAFE STOP y2={y2} ratio={distance_ratio:.2f} w={box_w}"
                break

    if stop:
        _stop_counter += 1
        _stop_latch = _STOP_LATCH_FRAMES  # Lock the brakes down
    else:
        # CRITICAL FIX: Do NOT clear the stop counter back to 0 instantly if it's flickering!
        # Slow down the counter clear rate so a single missed frame won't make the robot accelerate.
        _stop_counter = max(0, _stop_counter - 1)
        _stop_latch = max(0, _stop_latch - 1)

    # If we detected a stop threat recently, or the latch timer is still active, KEEP BRAKING!
    if _stop_counter >= _STOP_FRAMES_REQUIRED or _stop_latch > 0:
        return True, reason or "anti-flicker latch active"

    return False, ""
