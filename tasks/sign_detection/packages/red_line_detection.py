import numpy as np
import cv2


def detect_red_line(signBehavior, frame_rgb: np.ndarray) -> bool:
    h, w = frame_rgb.shape[:2]

    strip_h = max(2, int(h * signBehavior.cfg.red_strip_frac))

    # Look in front/middle road area, not whole image.
    l = int(w * 0.24)
    r = int(w * 0.76)

    strip = frame_rgb[h - strip_h:, l:r]
    strip_h_actual, strip_w_actual = strip.shape[:2]

    hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)

    lo1 = np.array(signBehavior.cfg.red_hsv_low1, dtype=np.uint8)
    hi1 = np.array(signBehavior.cfg.red_hsv_high1, dtype=np.uint8)
    lo2 = np.array(signBehavior.cfg.red_hsv_low2, dtype=np.uint8)
    hi2 = np.array(signBehavior.cfg.red_hsv_high2, dtype=np.uint8)

    mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    close_ratio = getattr(signBehavior.cfg, "red_line_close_y2_ratio", 0.82)

    best = None

    for cnt in contours:
        area = float(cv2.contourArea(cnt))

        if area < 80:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bw <= 0 or bh <= 0:
            continue

        y2 = y + bh
        aspect = bw / float(bh)

        # Red stop line must look mostly horizontal.
        if aspect < 1.8:
            continue

        # Ignore tiny red fragments.
        if bw < strip_w_actual * 0.12:
            continue

        # Main closeness rule.
        # Bigger close_ratio = stop later / closer.
        if y2 < strip_h_actual * close_ratio:
            continue

        score = area * aspect

        if best is None or score > best["score"]:
            best = {
                "area": area,
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "y2": y2,
                "aspect": aspect,
                "score": score,
            }

    if best is not None:
        print(
            f"[SignBehavior] red line close — "
            f"area={best['area']:.0f}, "
            f"bbox=({best['x']},{best['y']},{best['w']},{best['h']}), "
            f"y2_ratio={best['y2'] / strip_h_actual:.2f}, "
            f"required={close_ratio:.2f}, "
            f"aspect={best['aspect']:.2f}"
        )
        return True

    return False