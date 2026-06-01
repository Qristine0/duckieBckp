from typing import Tuple
import os
import numpy as np
import cv2
import yaml

_HSV_FILE = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config', 'lane_servoing_hsv_config.yaml')
try:
    with open(_HSV_FILE) as _f:
        _h = yaml.safe_load(_f) or {}
except FileNotFoundError:
    _h = {}

_yellow_lower = np.array([_h.get('yellow_lower_h', 22),  _h.get('yellow_lower_s', 100), _h.get('yellow_lower_v', 100)])
_yellow_upper = np.array([_h.get('yellow_upper_h', 35),  _h.get('yellow_upper_s', 255), _h.get('yellow_upper_v', 255)])
_white_lower  = np.array([_h.get('white_lower_h',   0),  _h.get('white_lower_s',    0), _h.get('white_lower_v',  180)])
_white_upper  = np.array([_h.get('white_upper_h', 179),  _h.get('white_upper_s',   60), _h.get('white_upper_v',  255)])


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]

    imghsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    roi_mask = np.zeros((h, w), dtype=np.uint8)
    roi_mask[int(h * 0.45):, :] = 255

    raw_yellow = cv2.bitwise_and(cv2.inRange(imghsv, _yellow_lower, _yellow_upper), roi_mask)
    raw_white  = cv2.bitwise_and(cv2.inRange(imghsv, _white_lower,  _white_upper),  roi_mask)

    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    clean_yellow = cv2.morphologyEx(raw_yellow, cv2.MORPH_OPEN,  k_open)
    clean_yellow = cv2.morphologyEx(clean_yellow, cv2.MORPH_CLOSE, k_close)

    clean_white  = cv2.morphologyEx(raw_white,  cv2.MORPH_OPEN,  k_open)
    clean_white  = cv2.morphologyEx(clean_white,  cv2.MORPH_CLOSE, k_close)

    # return (clean_yellow > 0).astype(np.float32), (clean_white > 0).astype(np.float32)

    left_mask = np.zeros((h, w), dtype=np.uint8)
    left_mask[:, :w//2] = 255
    
    right_mask = np.zeros((h, w), dtype=np.uint8)
    right_mask[:, w//2:] = 255

    # Apply split
    yellow_left = cv2.bitwise_and(clean_yellow, left_mask)
    white_right = cv2.bitwise_and(clean_white, right_mask)

    return (
        (yellow_left > 0).astype(np.float32),
        (white_right > 0).astype(np.float32)
    )


# def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     h, w, _ = image.shape
#     roi_start = int(h * 0.40)
#     roi = image[roi_start:, :]
#     roi_h = roi.shape[0]
#     small   = cv2.resize(roi, (w // 2, roi_h // 2))
#     blurred = cv2.GaussianBlur(small, (5, 5), 0)
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     yellow_mask = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
#     hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)
#     white_mask = cv2.inRange(hls, _white_lower, _white_upper)
#     # White only on right 70% — clip scaled to half resolution
#     white_mask[:, :int((w // 2) * 0.30)] = 0
#     kernel = np.ones((3, 3), np.uint8)
#     yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
#     white_mask  = cv2.dilate(white_mask,  kernel, iterations=1)
#     yellow_mask = cv2.resize(yellow_mask, (w, roi_h), interpolation=cv2.INTER_NEAREST)
#     white_mask  = cv2.resize(white_mask,  (w, roi_h), interpolation=cv2.INTER_NEAREST)
#     mask_left_edge  = np.zeros((h, w), dtype=float)
#     mask_right_edge = np.zeros((h, w), dtype=float)
#     mask_left_edge[roi_start:, :]  = (yellow_mask > 0).astype(float)
#     mask_right_edge[roi_start:, :] = (white_mask  > 0).astype(float)
#     return mask_left_edge, mask_right_edge

# def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     h, w, _ = image.shape

#     # ─────────────────────────────────────────────
#     # 1. ROI (road only)
#     # ─────────────────────────────────────────────
#     roi_start = int(h * 0.40)
#     roi = image[roi_start:, :]
#     roi_h, roi_w = roi.shape[:2]

#     small = cv2.resize(roi, (w // 2, roi_h // 2))
#     blurred = cv2.GaussianBlur(small, (5, 5), 0)

#     # ─────────────────────────────────────────────
#     # 2. Color spaces
#     # ─────────────────────────────────────────────
#     hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
#     hls = cv2.cvtColor(blurred, cv2.COLOR_BGR2HLS)

#     yellow_mask = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
#     white_mask  = cv2.inRange(hls, _white_lower, _white_upper)

#     kernel = np.ones((3, 3), np.uint8)
#     yellow_mask = cv2.dilate(yellow_mask, kernel, iterations=1)
#     white_mask  = cv2.dilate(white_mask, kernel, iterations=1)

#     # ─────────────────────────────────────────────
#     # 3. Restore to ROI resolution
#     # ─────────────────────────────────────────────
#     yellow_mask = cv2.resize(yellow_mask, (w, roi_h), interpolation=cv2.INTER_NEAREST)
#     white_mask  = cv2.resize(white_mask,  (w, roi_h), interpolation=cv2.INTER_NEAREST)

#     # ─────────────────────────────────────────────
#     # 4. SAFE SEPARATION (FIX FOR CROSSING YELLOW)
#     # ─────────────────────────────────────────────

#     # Find where yellow is strongest (lane boundary estimate)
#     yellow_x = np.where(yellow_mask.sum(axis=0) > 0)[0]

#     if len(yellow_x) > 0:
#         yellow_right_edge = int(np.percentile(yellow_x, 90))
#     else:
#         yellow_right_edge = int(w * 0.45)  # fallback

#     # Safety margin so white lane never crosses into yellow lane
#     safety_margin = int(w * 0.05)
#     white_cutoff = min(w, yellow_right_edge + safety_margin)

#     white_mask[:, :white_cutoff] = 0

#     # ─────────────────────────────────────────────
#     # 5. Output masks
#     # ─────────────────────────────────────────────
#     mask_left_edge  = np.zeros((h, w), dtype=np.uint8)
#     mask_right_edge = np.zeros((h, w), dtype=np.uint8)

#     mask_left_edge[roi_start:, :]  = (yellow_mask > 0).astype(np.uint8)
#     mask_right_edge[roi_start:, :] = (white_mask  > 0).astype(np.uint8)

#     return mask_left_edge, mask_right_edge


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper
    _yellow_lower = np.array(yellow_lower)
    _yellow_upper = np.array(yellow_upper)
    _white_lower  = np.array(white_lower)
    _white_upper  = np.array(white_upper)

def get_hsv_bounds():
    return {
        'yellow_lower_h': int(_yellow_lower[0]), 'yellow_upper_h': int(_yellow_upper[0]),
        'yellow_lower_s': int(_yellow_lower[1]), 'yellow_upper_s': int(_yellow_upper[1]),
        'yellow_lower_v': int(_yellow_lower[2]), 'yellow_upper_v': int(_yellow_upper[2]),
        'white_lower_h':  int(_white_lower[0]),  'white_upper_h':  int(_white_upper[0]),
        'white_lower_s':  int(_white_lower[1]),  'white_upper_s':  int(_white_upper[1]),
        'white_lower_v':  int(_white_lower[2]),  'white_upper_v':  int(_white_upper[2]),
    }