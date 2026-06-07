import numpy as np
import cv2


# Print debug only every N frames, so terminal is not spammed too much.
_DEBUG_EVERY_FRAMES = 8


def _get_frame_counter(signBehavior) -> int:
    if not hasattr(signBehavior, "_red_debug_frame"):
        signBehavior._red_debug_frame = 0

    signBehavior._red_debug_frame += 1
    return signBehavior._red_debug_frame


def detect_red_line(signBehavior, frame_rgb: np.ndarray) -> bool:
    """
    Detect the red stop/intersection line.

    Logic:
    - Look only at the lower/front part of the image.
    - Look mostly in the center driving lane area.
    - Detect true red + red/orange under camera lighting.
    - Accept only horizontal, wide components.
    - Return True only when the red line is close enough.
    - Print debug when red is seen and when it is confirmed.
    """

    frame_no = _get_frame_counter(signBehavior)

    h, w = frame_rgb.shape[:2]

    # Bigger strip = sees the line earlier.
    # Smaller strip = sees the line later.
    strip_frac = getattr(signBehavior.cfg, "red_strip_frac", 0.55)
    strip_h = max(2, int(h * strip_frac))

    # Look in front/middle road area, not whole image.
    l = int(w * 0.20)
    r = int(w * 0.80)

    strip = frame_rgb[h - strip_h:, l:r]
    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)

    # Normal red ranges.
    lo1 = np.array(getattr(signBehavior.cfg, "red_hsv_low1", (0, 100, 70)), dtype=np.uint8)
    hi1 = np.array(getattr(signBehavior.cfg, "red_hsv_high1", (12, 255, 255)), dtype=np.uint8)

    lo2 = np.array(getattr(signBehavior.cfg, "red_hsv_low2", (165, 100, 70)), dtype=np.uint8)
    hi2 = np.array(getattr(signBehavior.cfg, "red_hsv_high2", (179, 255, 255)), dtype=np.uint8)

    red_mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    # Some red lines look orange on the real camera.
    # But we require R channel dominance, so yellow lane dashes are not accepted.
    orange_mask = cv2.inRange(
        hsv,
        np.array([8, 90, 70], dtype=np.uint8),
        np.array([22, 255, 255], dtype=np.uint8),
    )

    r_ch = strip[:, :, 0].astype(np.int16)
    g_ch = strip[:, :, 1].astype(np.int16)
    b_ch = strip[:, :, 2].astype(np.int16)

    red_dominant = ((r_ch > g_ch + 18) & (r_ch > b_ch + 30)).astype(np.uint8) * 255
    orange_red_mask = cv2.bitwise_and(orange_mask, red_dominant)

    mask = red_mask | orange_red_mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Bigger = stop later / closer.
    # Smaller = stop earlier / farther.
    close_ratio = getattr(signBehavior.cfg, "red_line_close_y2_ratio", 0.58)

    best_any = None
    best_close = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))

        if area < 60:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 0 or bh <= 0:
            continue

        y2 = y + bh
        aspect = bw / float(bh + 1e-6)
        y2_ratio = y2 / float(strip_h_actual)

        # Red stop line should look mostly horizontal.
        if aspect < 1.5:
            continue

        # Ignore tiny red fragments.
        if bw < strip_w_actual * 0.10:
            continue

        score = area * aspect

        candidate = {
            "area": area,
            "x": x,
            "y": y,
            "w": bw,
            "h": bh,
            "y2": y2,
            "aspect": aspect,
            "y2_ratio": y2_ratio,
            "score": score,
        }

        if best_any is None or score > best_any["score"]:
            best_any = candidate

        # Main closeness rule.
        if y2_ratio >= close_ratio:
            if best_close is None or score > best_close["score"]:
                best_close = candidate

    if best_close is not None:
        print(
            f"[SignBehavior] RED LINE DETECTED CLOSE — "
            f"area={best_close['area']:.0f}, "
            f"bbox=({best_close['x']},{best_close['y']},{best_close['w']},{best_close['h']}), "
            f"y2_ratio={best_close['y2_ratio']:.2f}, "
            f"required={close_ratio:.2f}, "
            f"aspect={best_close['aspect']:.2f}"
        )
        return True

    # Debug: red exists, but not close enough yet.
    if best_any is not None and frame_no % _DEBUG_EVERY_FRAMES == 0:
        print(
            f"[SignBehavior] red seen but NOT close — "
            f"area={best_any['area']:.0f}, "
            f"bbox=({best_any['x']},{best_any['y']},{best_any['w']},{best_any['h']}), "
            f"y2_ratio={best_any['y2_ratio']:.2f}, "
            f"required={close_ratio:.2f}, "
            f"aspect={best_any['aspect']:.2f}"
        )

    return False