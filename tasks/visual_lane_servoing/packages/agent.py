import os
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

from . import visual_servoing_activity as student
from .curve_behavior import detect_curve


def _find_config_file() -> str:
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "config")
    )

    yaml_path = os.path.join(config_dir, "lane_servoing_config.yaml")
    yml_path = os.path.join(config_dir, "lane_servoing_config.yml")

    if os.path.exists(yaml_path):
        return yaml_path

    return yml_path


_CONFIG_FILE = _find_config_file()

# Use only the road area close enough to the bot.
# Far-away lines near intersections create false detections.
_ROI_START = 0.58
_ROI_END = 0.88

_NUM_SLICES = 5
_SLICE_TOL = 8

_INITIAL_LANE_HALF_WIDTH = 130.0
_MIN_COMPONENT_AREA = 10


def _component_centers(strip: np.ndarray) -> List[Tuple[float, int]]:
    binary = (strip > 0).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    components = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])

        if area < _MIN_COMPONENT_AREA:
            continue

        x_center = float(centroids[label][0])
        components.append((x_center, area))

    return components


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE

        try:
            with open(path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            print(f"[LaneServoingAgent] Config not found: {path}")
            cfg = {}
        except Exception as exc:
            print(f"[LaneServoingAgent] Could not load config: {exc}")
            cfg = {}

        self.p_gain = cfg.get("p_gain", 0.32)
        self.d_gain = cfg.get("d_gain", 0.18)

        # This is normalized steering, not direct PWM.
        self.max_steer = cfg.get("max_steer", 0.55)

        self.base_speed = cfg.get("base_speed", 0.11)
        self.curve_speed = cfg.get("curve_speed", 0.085)
        self.search_speed = cfg.get("search_speed", 0.04)

        # Wheel difference is limited relative to speed.
        # This prevents one wheel from stopping suddenly.
        self.turn_ratio = cfg.get("turn_ratio", 0.70)

        self.curve_threshold = cfg.get("curve_threshold", 35)
        self.curve_boost = cfg.get("curve_boost", 1.08)

        self.max_lost_frames = cfg.get("max_lost_frames", 8)

        self.error_smoothing = cfg.get("error_smoothing", 0.78)
        self.heading_smoothing = cfg.get("heading_smoothing", 0.72)
        self.steering_smoothing = cfg.get("steering_smoothing", 0.82)

        self.max_steering_delta = cfg.get("max_steering_delta", 0.06)
        self.debug_print_every = cfg.get("debug_print_every", 20)

        self.frame_count = 0

        self._lane_half_width = _INITIAL_LANE_HALF_WIDTH
        self._last_lane_center: Optional[float] = None

        self._filtered_lateral_error = 0.0
        self._filtered_heading_error = 0.0
        self._filtered_steering = 0.0

        self._lost_frames = 0

        self._left_history = deque(maxlen=4)
        self._right_history = deque(maxlen=4)

        self.last_debug_info = self._empty_debug_info(480, 640)

    def _select_lane_points(
        self,
        mask_yellow: np.ndarray,
        mask_white: np.ndarray,
    ) -> Tuple[List[int], List[int], List[float], List[int]]:
        h, w = mask_yellow.shape

        start_y = int(h * _ROI_START)
        end_y = int(h * _ROI_END)

        slice_ys = np.linspace(start_y, end_y, _NUM_SLICES).astype(int).tolist()

        yellow_xs: List[int] = []
        white_xs: List[int] = []
        lane_centers: List[float] = []

        expected_center = self._last_lane_center
        if expected_center is None:
            expected_center = w / 2.0

        expected_width = 2.0 * self._lane_half_width

        min_lane_width = w * 0.16
        max_lane_width = w * 0.70

        for y in slice_ys:
            y1 = max(0, y - _SLICE_TOL)
            y2 = min(h, y + _SLICE_TOL + 1)

            yellow_components = _component_centers(mask_yellow[y1:y2, :])
            white_components = _component_centers(mask_white[y1:y2, :])

            best_pair = None
            best_score = float("inf")

            for yx, y_area in yellow_components:
                for wx, w_area in white_components:
                    # In our lane: yellow must be left, white must be right.
                    if wx <= yx:
                        continue

                    lane_width = wx - yx

                    if lane_width < min_lane_width or lane_width > max_lane_width:
                        continue

                    center = (yx + wx) / 2.0

                    width_score = abs(lane_width - expected_width)
                    center_score = abs(center - expected_center)

                    # Prefer strong components, but not too aggressively.
                    area_bonus = min(y_area + w_area, 180)

                    score = width_score * 0.65 + center_score * 1.10 - area_bonus * 0.04

                    if score < best_score:
                        best_score = score
                        best_pair = (yx, wx, center, lane_width)

            if best_pair is not None:
                yx, wx, center, lane_width = best_pair

                yellow_xs.append(int(yx))
                white_xs.append(int(wx))
                lane_centers.append(float(center))

                half_width = lane_width / 2.0
                self._lane_half_width = 0.94 * self._lane_half_width + 0.06 * half_width

                continue

            # If both lines are not visible, use yellow only.
            if yellow_components:
                expected_yellow = expected_center - self._lane_half_width

                yx, _ = min(
                    yellow_components,
                    key=lambda component: abs(component[0] - expected_yellow),
                )

                center = yx + self._lane_half_width

                if 0.12 * w <= center <= 0.88 * w:
                    yellow_xs.append(int(yx))
                    lane_centers.append(float(center))
                    continue

            # If only white is visible, choose the expected RIGHT white line.
            if white_components:
                expected_white = expected_center + self._lane_half_width

                # Ignore very-left white components.
                right_white_candidates = [
                    component for component in white_components
                    if component[0] > w * 0.38
                ]

                if not right_white_candidates:
                    right_white_candidates = white_components

                wx, _ = min(
                    right_white_candidates,
                    key=lambda component: abs(component[0] - expected_white),
                )

                center = wx - self._lane_half_width

                if 0.12 * w <= center <= 0.88 * w:
                    white_xs.append(int(wx))
                    lane_centers.append(float(center))
                    continue

        if lane_centers:
            # Update expected center slowly to avoid jumping.
            measured_center = float(np.mean(lane_centers[-2:]))

            if self._last_lane_center is None:
                self._last_lane_center = measured_center
            else:
                self._last_lane_center = (
                    0.82 * self._last_lane_center
                    + 0.18 * measured_center
                )

        return yellow_xs, white_xs, lane_centers, slice_ys

    def _calculate_errors(
        self,
        lane_centers: List[float],
        image_width: int,
    ) -> Tuple[float, float]:
        if not lane_centers:
            return self._filtered_lateral_error, self._filtered_heading_error

        image_center = image_width / 2.0

        # Weighted center: lower slices are closer to the robot, so they matter more.
        weights = np.linspace(1.0, 2.2, len(lane_centers))
        target_center = float(np.average(lane_centers, weights=weights))

        lateral_error = (image_center - target_center) / image_center

        if len(lane_centers) >= 2:
            top_center = lane_centers[0]
            bottom_center = lane_centers[-1]
            heading_error = (bottom_center - top_center) / image_center
        else:
            heading_error = 0.0

        lateral_error = float(np.clip(lateral_error, -1.0, 1.0))
        heading_error = float(np.clip(heading_error, -1.0, 1.0))

        self._filtered_lateral_error = (
            self.error_smoothing * self._filtered_lateral_error
            + (1.0 - self.error_smoothing) * lateral_error
        )

        self._filtered_heading_error = (
            self.heading_smoothing * self._filtered_heading_error
            + (1.0 - self.heading_smoothing) * heading_error
        )

        return self._filtered_lateral_error, self._filtered_heading_error

    def _calculate_steering(
        self,
        lateral_error: float,
        heading_error: float,
        is_curve: bool,
    ) -> float:
        raw_steering = self.p_gain * lateral_error + self.d_gain * heading_error

        if is_curve:
            raw_steering *= self.curve_boost

        raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Rate limit: prevents sudden left-right changes.
        delta = raw_steering - self._filtered_steering
        delta = float(np.clip(delta, -self.max_steering_delta, self.max_steering_delta))

        limited_steering = self._filtered_steering + delta

        # Smooth steering like a good driver.
        self._filtered_steering = (
            self.steering_smoothing * self._filtered_steering
            + (1.0 - self.steering_smoothing) * limited_steering
        )

        return float(np.clip(self._filtered_steering, -self.max_steer, self.max_steer))

    def _motor_commands(
        self,
        steering: float,
        lane_detected: bool,
        is_curve: bool,
        both_visible: bool,
    ) -> Tuple[float, float]:
        if not lane_detected:
            self._lost_frames += 1

            if self._lost_frames > self.max_lost_frames:
                return 0.0, 0.0

            speed = self.search_speed
            steering = self._filtered_steering * 0.5
        else:
            self._lost_frames = 0
            speed = self.curve_speed if is_curve else self.base_speed

            if not both_visible:
                speed *= 0.85

        # Convert normalized steering to PWM turn amount.
        # This is the key smoothing fix.
        max_turn_pwm = speed * self.turn_ratio
        turn_pwm = float(np.clip(steering * speed, -max_turn_pwm, max_turn_pwm))

        left = speed - turn_pwm
        right = speed + turn_pwm

        return (
            float(np.clip(left, 0.0, 1.0)),
            float(np.clip(right, 0.0, 1.0)),
        )

    def _smooth_wheels(self, left: float, right: float) -> Tuple[float, float]:
        self._left_history.append(left)
        self._right_history.append(right)

        return (
            float(sum(self._left_history) / len(self._left_history)),
            float(sum(self._right_history) / len(self._right_history)),
        )

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1

        try:
            mask_yellow_float, mask_white_float = student.detect_lane_markings(image)
        except Exception as exc:
            print(f"[LaneServoingAgent] detect_lane_markings error: {exc}")
            return 0.0, 0.0

        mask_yellow = (mask_yellow_float * 255).astype(np.uint8)
        mask_white = (mask_white_float * 255).astype(np.uint8)

        h, w = mask_yellow.shape

        yellow_pixels = int(np.count_nonzero(mask_yellow))
        white_pixels = int(np.count_nonzero(mask_white))
        total_pixels = yellow_pixels + white_pixels

        yellow_xs, white_xs, lane_centers, slice_ys = self._select_lane_points(
            mask_yellow,
            mask_white,
        )

        lane_detected = len(lane_centers) >= 1
        both_visible = len(yellow_xs) >= 1 and len(white_xs) >= 1

        is_curve, curve_dir = detect_curve(
            yellow_xs=yellow_xs,
            white_xs=white_xs,
            curve_threshold=self.curve_threshold,
        )

        lateral_error, heading_error = self._calculate_errors(lane_centers, w)

        steering = self._calculate_steering(
            lateral_error=lateral_error,
            heading_error=heading_error,
            is_curve=is_curve,
        )

        left, right = self._motor_commands(
            steering=steering,
            lane_detected=lane_detected,
            is_curve=is_curve,
            both_visible=both_visible,
        )

        left, right = self._smooth_wheels(left, right)

        combined = np.clip(mask_yellow_float + mask_white_float, 0, 1)

        self.last_debug_info = {
            "roi": image,
            "lane_mask": (combined * 255).astype(np.uint8),
            "white_mask": mask_white,
            "yellow_mask": mask_yellow,

            "yellow_pixels": yellow_pixels,
            "white_pixels": white_pixels,
            "total_lane_pixels": total_pixels,

            "yellow_xs": yellow_xs,
            "white_xs": white_xs,
            "lane_centers": lane_centers,
            "slice_ys": slice_ys,

            "lane_detected": lane_detected,
            "both_visible": both_visible,

            "lateral_error": lateral_error,
            "heading_error": heading_error,
            "steering": steering,

            "is_curve": is_curve,
            "curve_dir": curve_dir,

            "left_command": left,
            "right_command": right,

            "lane_half_width": self._lane_half_width,
            "lost_frames": self._lost_frames,
            "frame_count": self.frame_count,
        }

        if self.debug_print_every and self.frame_count % self.debug_print_every == 0:
            print(
                "[LaneServoingAgent] "
                f"yellow_xs={yellow_xs}, white_xs={white_xs}, "
                f"centers={[round(c, 1) for c in lane_centers]}, "
                f"lat={lateral_error:.3f}, head={heading_error:.3f}, "
                f"steer={steering:.3f}, "
                f"left={left:.3f}, right={right:.3f}, "
                f"lane={lane_detected}, both={both_visible}"
            )

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h: int, w: int) -> dict:
        return {
            "roi": np.zeros((h, w, 3), dtype=np.uint8),
            "lane_mask": np.zeros((h, w), dtype=np.uint8),
            "white_mask": np.zeros((h, w), dtype=np.uint8),
            "yellow_mask": np.zeros((h, w), dtype=np.uint8),

            "yellow_pixels": 0,
            "white_pixels": 0,
            "total_lane_pixels": 0,

            "yellow_xs": [],
            "white_xs": [],
            "lane_centers": [],
            "slice_ys": [],

            "lane_detected": False,
            "both_visible": False,

            "lateral_error": 0.0,
            "heading_error": 0.0,
            "steering": 0.0,

            "is_curve": False,
            "curve_dir": 0,

            "left_command": 0.0,
            "right_command": 0.0,

            "lane_half_width": _INITIAL_LANE_HALF_WIDTH,
            "lost_frames": 0,
            "frame_count": 0,
        }