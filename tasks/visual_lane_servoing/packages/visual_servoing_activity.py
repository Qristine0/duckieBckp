from typing import Tuple
import os

import cv2
import numpy as np
import yaml


def _find_config_file() -> str:
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
    )

    yaml_path = os.path.join(config_dir, "lane_servoing_hsv_config.yaml")
    yml_path = os.path.join(config_dir, "lane_servoing_hsv_config.yml")

    if os.path.exists(yaml_path):
        return yaml_path

    return yml_path


HSV_FILE = _find_config_file()


def _load_hsv_config() -> dict:
    try:
        with open(HSV_FILE, "r") as f:
            return yaml.safe_load(f) or {}
    except FileNotFoundError:
        print(f"[LaneDetection] HSV config not found: {HSV_FILE}")
        return {}
    except Exception as exc:
        print(f"[LaneDetection] Could not load HSV config: {exc}")
        return {}


_h = _load_hsv_config()

_yellow_lower = np.array([
    _h.get("yellow_lower_h", 18),
    _h.get("yellow_lower_s", 70),
    _h.get("yellow_lower_v", 70),
], dtype=np.uint8)

_yellow_upper = np.array([
    _h.get("yellow_upper_h", 40),
    _h.get("yellow_upper_s", 255),
    _h.get("yellow_upper_v", 255),
], dtype=np.uint8)

_white_lower = np.array([
    _h.get("white_lower_h", 0),
    _h.get("white_lower_s", 0),
    _h.get("white_lower_v", 140),
], dtype=np.uint8)

_white_upper = np.array([
    _h.get("white_upper_h", 179),
    _h.get("white_upper_s", 90),
    _h.get("white_upper_v", 255),
], dtype=np.uint8)


def _roi_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)

    # We only care about the lower part of the image where the road is.
    roi_start = int(height * 0.45)
    mask[roi_start:, : ] = 255

    return mask


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_close)

    return cleaned


def _detect_from_hsv(hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    h, w = hsv.shape[:2]
    roi = _roi_mask(h, w)

    raw_yellow = cv2.inRange(hsv, _yellow_lower, _yellow_upper)
    raw_white = cv2.inRange(hsv, _white_lower, _white_upper)

    raw_yellow = cv2.bitwise_and(raw_yellow, roi)
    raw_white = cv2.bitwise_and(raw_white, roi)

    # Hide far-left white line.
    # Do NOT use 0.52 here, because on curves the correct right white line
    # can move closer to the center. 0.40 is safer.
    ignore_left_until_x = int(w * 0.40)
    raw_white[:, :ignore_left_until_x] = 0

    clean_yellow = _clean_mask(raw_yellow)
    clean_white = _clean_mask(raw_white)

    return clean_yellow, clean_white


def _mask_score(yellow_mask: np.ndarray, white_mask: np.ndarray) -> int:
    yellow_pixels = int(np.count_nonzero(yellow_mask))
    white_pixels = int(np.count_nonzero(white_mask))

    # Yellow is more useful for detecting whether the channel order is correct.
    return yellow_pixels * 2 + white_pixels


def detect_lane_markings(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect left yellow lane marking and right white lane marking.

    The notebook says this function receives BGR images, but real/sim pipelines
    can sometimes differ. To make this safer, we try both BGR and RGB HSV
    conversions and keep the one with stronger lane evidence.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected color image with shape HxWx3, got {image.shape}")

    hsv_from_bgr = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    yellow_bgr, white_bgr = _detect_from_hsv(hsv_from_bgr)

    hsv_from_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    yellow_rgb, white_rgb = _detect_from_hsv(hsv_from_rgb)

    score_bgr = _mask_score(yellow_bgr, white_bgr)
    score_rgb = _mask_score(yellow_rgb, white_rgb)

    if score_rgb > score_bgr:
        yellow_mask = yellow_rgb
        white_mask = white_rgb
    else:
        yellow_mask = yellow_bgr
        white_mask = white_bgr

    return (
        (yellow_mask > 0).astype(np.float32),
        (white_mask > 0).astype(np.float32),
    )


def set_hsv_bounds(yellow_lower, yellow_upper, white_lower, white_upper):
    global _yellow_lower, _yellow_upper, _white_lower, _white_upper

    _yellow_lower = np.array(yellow_lower, dtype=np.uint8)
    _yellow_upper = np.array(yellow_upper, dtype=np.uint8)
    _white_lower = np.array(white_lower, dtype=np.uint8)
    _white_upper = np.array(white_upper, dtype=np.uint8)


def get_hsv_bounds():
    return {
        "yellow_lower_h": int(_yellow_lower[0]),
        "yellow_upper_h": int(_yellow_upper[0]),
        "yellow_lower_s": int(_yellow_lower[1]),
        "yellow_upper_s": int(_yellow_upper[1]),
        "yellow_lower_v": int(_yellow_lower[2]),
        "yellow_upper_v": int(_yellow_upper[2]),

        "white_lower_h": int(_white_lower[0]),
        "white_upper_h": int(_white_upper[0]),
        "white_lower_s": int(_white_lower[1]),
        "white_upper_s": int(_white_upper[1]),
        "white_lower_v": int(_white_lower[2]),
        "white_upper_v": int(_white_upper[2]),
    }