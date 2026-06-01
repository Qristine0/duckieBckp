import numpy as np
import cv2


def detect_red_line(signBehavior, frame_rgb: np.ndarray) -> bool:
        h, w    = frame_rgb.shape[:2]
        strip_h = max(2, int(h * signBehavior.cfg.red_strip_frac))
        
        # todo check rame ar aurios
        l = int(w / 3)
        r = int(2*w/3)
        strip   = frame_rgb[h - strip_h:, l:r]
        # strip   = frame_rgb[h - strip_h:, ]

        hsv = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
        lo1 = np.array(signBehavior.cfg.red_hsv_low1,  dtype=np.uint8)
        hi1 = np.array(signBehavior.cfg.red_hsv_high1, dtype=np.uint8)
        lo2 = np.array(signBehavior.cfg.red_hsv_low2,  dtype=np.uint8)
        hi2 = np.array(signBehavior.cfg.red_hsv_high2, dtype=np.uint8)
        mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

        mid            = strip_h // 2
        far_mask       = mask[:mid, :]
        far_fraction   = float(far_mask.sum()) / (255.0 * mid * w)
        far_threshold  = signBehavior.cfg.red_pixel_frac * 0.5
        near_mask      = mask[mid:, :]
        near_fraction  = float(near_mask.sum()) / (255.0 * (strip_h - mid) * w)
        near_threshold = signBehavior.cfg.red_pixel_frac

        detected = bool(far_fraction >= far_threshold or near_fraction >= near_threshold)
        if detected:
            print(f"[SignBehavior] red line — "
                  f"far={far_fraction:.3f}(thr={far_threshold:.3f}) "
                  f"near={near_fraction:.3f}(thr={near_threshold:.3f})")
        return detected