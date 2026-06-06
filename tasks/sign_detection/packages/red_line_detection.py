import numpy as np
import cv2


def detect_red_line(signBehavior, frame_rgb: np.ndarray) -> bool:
    h, w = frame_rgb.shape[:2]

    # Smaller bottom ROI means the red line must be closer before detection.
    strip_h = max(2, int(h * signBehavior.cfg.red_strip_frac))

    # Look only around the center of the driving lane.
    l = int(w * 0.30)
    r = int(w * 0.70)

    strip = frame_rgb[h - strip_h:, l:r]
    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)

    lo1 = np.array(signBehavior.cfg.red_hsv_low1, dtype=np.uint8)
    hi1 = np.array(signBehavior.cfg.red_hsv_high1, dtype=np.uint8)
    lo2 = np.array(signBehavior.cfg.red_hsv_low2, dtype=np.uint8)
    hi2 = np.array(signBehavior.cfg.red_hsv_high2, dtype=np.uint8)

    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    # Remove small red noise.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Important:
    # Only check the lower part of the ROI.
    # This prevents stopping when the red line is still far away.
    near_start = int(strip_h_actual * 0.55)
    near_mask = mask[near_start:, :]

    near_pixels = float(np.count_nonzero(near_mask))
    near_area = float(near_mask.shape[0] * near_mask.shape[1])

    if near_area <= 0:
        return False

    near_fraction = near_pixels / near_area
    threshold = signBehavior.cfg.red_pixel_frac

    detected = near_fraction >= threshold

    if detected:
        print(
            f"[SignBehavior] red line close — "
            f"near={near_fraction:.3f}(thr={threshold:.3f}) "
            f"roi_h={strip_h_actual}"
        )

    return detected