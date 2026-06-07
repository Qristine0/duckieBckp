# import os
# import yaml
# import numpy as np
# import cv2
# from collections import deque
# from typing import Tuple
#
# from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
# from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve
#
# _CONFIG_FILE = os.path.normpath(os.path.join(
#     os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_config.yaml"
# ))
#
# _LINE_OFFSET = 160
# _SLICE_TOL = 6
#
# # Keep close to your original. Do not overreact to far top pixels.
# _SLICE_Y_RATIOS = [0.56, 0.64, 0.72, 0.80, 0.88]
#
#
# def _strip_center_x(mask: np.ndarray, y: int, prefer_right: bool = False):
#     h, w = mask.shape[:2]
#
#     y1 = max(0, y - _SLICE_TOL)
#     y2 = min(h, y + _SLICE_TOL)
#
#     strip = mask[y1:y2, :]
#     idx = np.where(strip > 0)[1]
#
#     if len(idx) < 3:
#         return None
#
#     if prefer_right:
#         # Do not use all white pixels together.
#         # Split into clusters and use the strongest right-side cluster.
#         idx = idx[idx > int(w * 0.35)]
#
#         if len(idx) < 3:
#             return None
#
#         idx = np.sort(idx)
#         gaps = np.where(np.diff(idx) > 20)[0] + 1
#         clusters = np.split(idx, gaps)
#
#         clusters = [c for c in clusters if len(c) >= 3]
#
#         if not clusters:
#             return None
#
#         best = max(clusters, key=lambda c: len(c))
#
#         return int(np.median(best))
#
#     return int(np.median(idx))
#
#
# def detect_lines_in_slices(
#     mask_yellow: np.ndarray,
#     mask_white: np.ndarray,
#     h: int,
# ) -> Tuple[list, list]:
#     yellow_xs = []
#     white_xs = []
#
#     for ratio in _SLICE_Y_RATIOS:
#         y = int(h * ratio)
#
#         yellow_x = _strip_center_x(mask_yellow, y, prefer_right=False)
#         white_x = _strip_center_x(mask_white, y, prefer_right=True)
#
#         if yellow_x is not None:
#             yellow_xs.append(yellow_x)
#
#         if white_x is not None:
#             white_xs.append(white_x)
#
#     return yellow_xs, white_xs
#
#
# class LaneServoingAgent:
#
#     def __init__(self, config_path: str = None):
#         path = config_path or _CONFIG_FILE
#
#         try:
#             with open(path) as f:
#                 cfg = yaml.safe_load(f) or {}
#         except Exception:
#             cfg = {}
#
#         self.p_gain = cfg.get("p_gain", 0.12)
#         self.d_gain = cfg.get("d_gain", 0.04)
#         self.max_steer = cfg.get("max_steer", 0.20)
#         self.base_speed = cfg.get("base_speed", 0.12)
#         self.curve_speed = cfg.get("curve_speed", 0.10)
#
#         # Important: 350 was too late, but very low values overreact.
#         self.curve_threshold = cfg.get("curve_threshold", 160)
#
#         self.steering_threshold = cfg.get("steering_threshold", 0.2)
#         self.curve_boost = cfg.get("curve_boost", 1.0)
#         self.detection_threshold = cfg.get("detection_threshold", 80)
#
#         self.frame_count = 0
#
#         self._prev_error = 0.0
#         self._prev_diff = 0.0
#         self._filtered_error = 0.0
#         self._filtered_steering = 0.0
#
#         self._lane_half_width = float(_LINE_OFFSET)
#
#         self._left_history = deque(maxlen=4)
#         self._right_history = deque(maxlen=4)
#
#         self._lost_lane_frames = 0
#         self._max_lost_lane_frames = 5
#
#         # Short memory for line omissions.
#         self._last_yellow_xs = []
#         self._last_white_xs = []
#         self._yellow_memory = 0
#         self._white_memory = 0
#         self._max_memory_frames = 3
#
#         self.last_debug_info = self._empty_debug_info(480, 640)
#
#     def _apply_line_memory(self, yellow_xs, white_xs):
#         if yellow_xs:
#             self._last_yellow_xs = list(yellow_xs)
#             self._yellow_memory = self._max_memory_frames
#         elif self._yellow_memory > 0 and self._last_yellow_xs:
#             yellow_xs = list(self._last_yellow_xs)
#             self._yellow_memory -= 1
#
#         if white_xs:
#             self._last_white_xs = list(white_xs)
#             self._white_memory = self._max_memory_frames
#         elif self._white_memory > 0 and self._last_white_xs:
#             white_xs = list(self._last_white_xs)
#             self._white_memory -= 1
#
#         return yellow_xs, white_xs
#
#     def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
#         if left_det and right_det and yellow_xs and white_xs:
#             y_mean = float(np.median(yellow_xs))
#             w_mean = float(np.median(white_xs))
#
#             if w_mean <= y_mean + 50:
#                 error = self._prev_error * (w / 2.0)
#             else:
#                 measured = (w_mean - y_mean) / 2.0
#
#                 if 60 < measured < 360:
#                     self._lane_half_width = 0.92 * self._lane_half_width + 0.08 * measured
#
#                 lane_center = (y_mean + w_mean) / 2.0
#                 error = w / 2.0 - lane_center
#
#         elif left_det and yellow_xs:
#             y_mean = float(np.median(yellow_xs))
#             lane_center = y_mean + self._lane_half_width
#             error = w / 2.0 - lane_center
#
#         elif right_det and white_xs:
#             w_mean = float(np.median(white_xs))
#             lane_center = w_mean - self._lane_half_width
#             error = w / 2.0 - lane_center
#
#         else:
#             error = self._prev_error * (w / 2.0)
#
#         return float(np.clip(error / (w / 2.0), -1.0, 1.0))
#
#     def _calculate_steering(self, error: float, is_curve: bool) -> float:
#         raw_diff = error - self._prev_error
#
#         if is_curve:
#             error_diff = 0.72 * self._prev_diff + 0.28 * raw_diff
#             max_delta = 0.028
#         else:
#             error_diff = 0.80 * self._prev_diff + 0.20 * raw_diff
#             max_delta = 0.022
#
#         self._prev_diff = error_diff
#         self._prev_error = error
#
#         raw_steering = self.p_gain * error + self.d_gain * error_diff
#         raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))
#
#         delta = np.clip(raw_steering - self._filtered_steering, -max_delta, max_delta)
#
#         self._filtered_steering += delta
#         self._filtered_steering = float(
#             np.clip(self._filtered_steering, -self.max_steer, self.max_steer)
#         )
#
#         return self._filtered_steering
#
#     def _motor_commands(
#         self,
#         steering: float,
#         recovery: bool,
#         is_curve: bool,
#         both_visible: bool,
#         one_visible: bool,
#     ):
#         if recovery:
#             return 0.0, 0.0
#
#         speed = self.curve_speed if is_curve else self.base_speed
#
#         if not both_visible and one_visible:
#             speed *= 0.78
#
#         speed *= max(0.72, 1.0 - abs(steering) * 1.3)
#
#         left = speed - steering
#         right = speed + steering
#
#         return float(np.clip(left, 0.0, 0.30)), float(np.clip(right, 0.0, 0.30))
#
#     def _smooth(self, left, right, both_visible):
#         buf = 4 if both_visible else 3
#
#         if self._left_history.maxlen != buf:
#             self._left_history = deque(maxlen=buf)
#             self._right_history = deque(maxlen=buf)
#
#         self._left_history.append(left)
#         self._right_history.append(right)
#
#         return (
#             sum(self._left_history) / len(self._left_history),
#             sum(self._right_history) / len(self._right_history),
#         )
#
#     def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
#         self.frame_count += 1
#
#         bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         try:
#             mask_left, mask_right = student.detect_lane_markings(bgr)
#         except Exception as e:
#             print(f"[Agent] detect_lane_markings error: {e}")
#             return 0.0, 0.0
#
#         mask_y = (mask_left * 255).astype(np.uint8)
#         mask_w = (mask_right * 255).astype(np.uint8)
#
#         yellow_pixels = int(np.count_nonzero(mask_y))
#         white_pixels = int(np.count_nonzero(mask_w))
#         total_pixels = yellow_pixels + white_pixels
#
#         h, w = mask_y.shape
#
#         yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
#
#         yellow_xs, white_xs = self._apply_line_memory(yellow_xs, white_xs)
#
#         left_det = len(yellow_xs) > 0
#         right_det = len(white_xs) > 0
#
#         both_visible = left_det and right_det
#         one_visible = left_det or right_det
#
#         if one_visible:
#             self._lost_lane_frames = 0
#         else:
#             self._lost_lane_frames += 1
#
#         recovery = self._lost_lane_frames > self._max_lost_lane_frames
#
#         is_curve, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)
#
#         raw_error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)
#
#         if is_curve:
#             self._filtered_error = 0.82 * self._filtered_error + 0.18 * raw_error
#         else:
#             self._filtered_error = 0.88 * self._filtered_error + 0.12 * raw_error
#
#         steering = self._calculate_steering(self._filtered_error, is_curve)
#
#         left, right = self._motor_commands(
#             steering,
#             recovery,
#             is_curve,
#             both_visible,
#             one_visible,
#         )
#
#         left, right = self._smooth(left, right, both_visible)
#
#         combined = np.clip(mask_left + mask_right, 0, 1)
#
#         self.last_debug_info = {
#             "roi": image,
#             "lane_mask": (combined * 255).astype(np.uint8),
#             "white_mask": mask_w,
#             "yellow_mask": mask_y,
#             "total_lane_pixels": total_pixels,
#             "lateral_error": float(np.clip(self._prev_error, -1.0, 1.0)),
#             "lane_detected": not recovery,
#             "frame_count": self.frame_count,
#             "yellow_xs": yellow_xs,
#             "white_xs": white_xs,
#             "slice_ys": [int(h * r) for r in _SLICE_Y_RATIOS],
#             "is_curve": is_curve,
#             "curve_dir": curve_dir,
#             "lost_lane_frames": self._lost_lane_frames,
#         }
#
#         if self.frame_count % 20 == 0:
#             print(
#                 f"[LaneServoingAgent] yellow_xs={yellow_xs}, white_xs={white_xs}, "
#                 f"curve={is_curve}, lost={self._lost_lane_frames}, "
#                 f"err={self._filtered_error:.3f}, steer={steering:.3f}, "
#                 f"left={left:.3f}, right={right:.3f}, lane={not recovery}"
#             )
#
#         return left, right
#
#     def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
#         left, right = self.compute_commands(image)
#         wheels_driver.set_wheels_speed(left, right)
#         return left, right
#
#     def get_debug_info(self, image: np.ndarray) -> dict:
#         return self.last_debug_info
#
#     def _empty_debug_info(self, h, w):
#         return {
#             "roi": np.zeros((h, w, 3), dtype=np.uint8),
#             "lane_mask": np.zeros((h, w), dtype=np.uint8),
#             "white_mask": np.zeros((h, w), dtype=np.uint8),
#             "yellow_mask": np.zeros((h, w), dtype=np.uint8),
#             "total_lane_pixels": 0,
#             "lateral_error": 0.0,
#             "lane_detected": False,
#             "frame_count": 0,
#             "yellow_xs": [],
#             "white_xs": [],
#             "slice_ys": [],
#             "is_curve": False,
#             "curve_dir": 0,
#             "lost_lane_frames": 0,
#         }
import os
import yaml
import numpy as np
import cv2
from collections import deque
from typing import Tuple

from tasks.visual_lane_servoing.packages import visual_servoing_activity as student
from tasks.visual_lane_servoing.packages.cuvrve_behavior import detect_curve

_CONFIG_FILE = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "config", "lane_servoing_config.yaml"
))

_LINE_OFFSET = 160
_SLICE_TOL = 6

# Keep close to your working version.
_SLICE_Y_RATIOS = [0.56, 0.64, 0.72, 0.80, 0.88]

def _strip_center_x(mask: np.ndarray, y: int, prefer_right: bool = False):
    h, w = mask.shape[:2]

    y1 = max(0, y - _SLICE_TOL)
    y2 = min(h, y + _SLICE_TOL)

    strip = mask[y1:y2, :]
    idx = np.where(strip > 0)[1]

    if len(idx) < 3:
        return None

    if prefer_right:
        # Ignore left-side white line and only look on the right half/side.
        idx = idx[idx > int(w * 0.35)]

        if len(idx) < 3:
            return None

        idx = np.sort(idx)

        # Split separate white objects/lines into clusters.
        gaps = np.where(np.diff(idx) > 20)[0] + 1
        clusters = np.split(idx, gaps)

        valid_clusters = []

        for cluster in clusters:
            if len(cluster) < 3:
                continue

            cluster_left = int(cluster[0])
            cluster_right = int(cluster[-1])
            cluster_width = cluster_right - cluster_left + 1
            cluster_center = float(np.median(cluster))

            # Remove tiny noise.
            if cluster_width < 3:
                continue

            # Remove far-right other-road lines in upper/middle image.
            if cluster_center > w * 0.88 and y < h * 0.78:
                continue

            valid_clusters.append(cluster)

        if not valid_clusters:
            return None

        # Choose the closest/right-lane cluster, not the biggest far-road cluster.
        best = min(valid_clusters, key=lambda c: np.median(c))

        # IMPORTANT:
        # Use the LEFT EDGE of the white cluster.
        # If white ground/reflection is attached to the right side, median would be wrong.
        left_edge = int(best[0])

        # Small offset inside the white tape so it is not too unstable.
        return left_edge + 8

    return int(np.median(idx))

def detect_lines_in_slices(
    mask_yellow: np.ndarray,
    mask_white: np.ndarray,
    h: int,
) -> Tuple[list, list]:
    yellow_xs = []
    white_xs = []

    for ratio in _SLICE_Y_RATIOS:
        y = int(h * ratio)

        yellow_x = _strip_center_x(mask_yellow, y, prefer_right=False)
        white_x = _strip_center_x(mask_white, y, prefer_right=True)

        if yellow_x is not None:
            yellow_xs.append(yellow_x)

        if white_x is not None:
            white_xs.append(white_x)

    return yellow_xs, white_xs


class LaneServoingAgent:

    def __init__(self, config_path: str = None):
        path = config_path or _CONFIG_FILE

        try:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        self.p_gain = cfg.get("p_gain", 0.095)
        self.d_gain = cfg.get("d_gain", 0.025)
        self.max_steer = cfg.get("max_steer", 0.16)
        self.base_speed = cfg.get("base_speed", 0.115)
        self.curve_speed = cfg.get("curve_speed", 0.095)
        self.curve_threshold = cfg.get("curve_threshold", 230)
        self.steering_threshold = cfg.get("steering_threshold", 0.2)
        self.curve_boost = cfg.get("curve_boost", 1.0)
        self.detection_threshold = cfg.get("detection_threshold", 80)

        self.frame_count = 0

        self._prev_error = 0.0
        self._prev_diff = 0.0
        self._filtered_error = 0.0
        self._filtered_steering = 0.0

        self._lane_half_width = float(_LINE_OFFSET)

        self._left_history = deque(maxlen=5)
        self._right_history = deque(maxlen=5)

        self._lost_lane_frames = 0
        self._max_lost_lane_frames = 5

        self._last_yellow_xs = []
        self._last_white_xs = []
        self._yellow_memory = 0
        self._white_memory = 0
        self._max_memory_frames = 3

        # Prevent false curve mode on straight road.
        self._curve_counter = 0

        self.last_debug_info = self._empty_debug_info(480, 640)

    def _apply_line_memory(self, yellow_xs, white_xs):
        if yellow_xs:
            self._last_yellow_xs = list(yellow_xs)
            self._yellow_memory = self._max_memory_frames
        elif self._yellow_memory > 0 and self._last_yellow_xs:
            yellow_xs = list(self._last_yellow_xs)
            self._yellow_memory -= 1

        if white_xs:
            self._last_white_xs = list(white_xs)
            self._white_memory = self._max_memory_frames
        elif self._white_memory > 0 and self._last_white_xs:
            white_xs = list(self._last_white_xs)
            self._white_memory -= 1

        return yellow_xs, white_xs

    def _calculate_error(self, yellow_xs, white_xs, left_det, right_det, w):
        if left_det and right_det and yellow_xs and white_xs:
            y_mean = float(np.median(yellow_xs))
            w_mean = float(np.median(white_xs))

            if w_mean <= y_mean + 50:
                error = self._prev_error * (w / 2.0)
            else:
                measured = (w_mean - y_mean) / 2.0

                if 60 < measured < 360:
                    self._lane_half_width = 0.92 * self._lane_half_width + 0.08 * measured

                lane_center = (y_mean + w_mean) / 2.0
                error = w / 2.0 - lane_center

        elif left_det and yellow_xs:
            y_mean = float(np.median(yellow_xs))
            lane_center = y_mean + self._lane_half_width
            error = w / 2.0 - lane_center

        elif right_det and white_xs:
            w_mean = float(np.median(white_xs))
            lane_center = w_mean - self._lane_half_width
            error = w / 2.0 - lane_center

        else:
            error = self._prev_error * (w / 2.0)

        return float(np.clip(error / (w / 2.0), -1.0, 1.0))

    def _calculate_steering(self, error: float, is_curve: bool) -> float:
        # Deadband: ignore tiny errors on straight road.
        # This is the main anti-zigzag fix.
        if not is_curve and abs(error) < 0.045:
            error = 0.0

        raw_diff = error - self._prev_error

        if is_curve:
            error_diff = 0.78 * self._prev_diff + 0.22 * raw_diff
            max_delta = 0.018
            p_boost = 1.15
        else:
            error_diff = 0.88 * self._prev_diff + 0.12 * raw_diff
            max_delta = 0.010
            p_boost = 1.0

        self._prev_diff = error_diff
        self._prev_error = error

        raw_steering = (self.p_gain * p_boost) * error + self.d_gain * error_diff
        raw_steering = float(np.clip(raw_steering, -self.max_steer, self.max_steer))

        # Second deadband: do not turn for tiny steering values.
        if not is_curve and abs(raw_steering) < 0.014:
            raw_steering = 0.0

        delta = np.clip(raw_steering - self._filtered_steering, -max_delta, max_delta)

        self._filtered_steering += delta
        self._filtered_steering = float(
            np.clip(self._filtered_steering, -self.max_steer, self.max_steer)
        )

        return self._filtered_steering

    def _motor_commands(
        self,
        steering: float,
        recovery: bool,
        is_curve: bool,
        both_visible: bool,
        one_visible: bool,
    ):
        if recovery:
            return 0.0, 0.0

        speed = self.curve_speed if is_curve else self.base_speed

        if not both_visible and one_visible:
            speed *= 0.80

        speed *= max(0.75, 1.0 - abs(steering) * 1.1)

        # Straight-road tiny steering should become exactly straight.
        if not is_curve and abs(steering) < 0.012:
            steering = 0.0

        left = speed - steering
        right = speed + steering

        return float(np.clip(left, 0.0, 0.28)), float(np.clip(right, 0.0, 0.28))

    def _smooth(self, left, right, both_visible):
        buf = 5 if both_visible else 4

        if self._left_history.maxlen != buf:
            self._left_history = deque(maxlen=buf)
            self._right_history = deque(maxlen=buf)

        self._left_history.append(left)
        self._right_history.append(right)

        return (
            sum(self._left_history) / len(self._left_history),
            sum(self._right_history) / len(self._right_history),
        )

    def _confirmed_curve(self, yellow_xs, white_xs):
        curve_now, curve_dir = detect_curve(yellow_xs, white_xs, self.curve_threshold)

        if curve_now:
            self._curve_counter = min(3, self._curve_counter + 1)
        else:
            self._curve_counter = max(0, self._curve_counter - 1)

        # Needs two consecutive curve-like frames.
        return self._curve_counter >= 2, curve_dir

    def compute_commands(self, image: np.ndarray) -> Tuple[float, float]:
        self.frame_count += 1

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            mask_left, mask_right = student.detect_lane_markings(bgr)
        except Exception as e:
            print(f"[Agent] detect_lane_markings error: {e}")
            return 0.0, 0.0

        mask_y = (mask_left * 255).astype(np.uint8)
        mask_w = (mask_right * 255).astype(np.uint8)

        yellow_pixels = int(np.count_nonzero(mask_y))
        white_pixels = int(np.count_nonzero(mask_w))
        total_pixels = yellow_pixels + white_pixels

        h, w = mask_y.shape

        yellow_xs, white_xs = detect_lines_in_slices(mask_y, mask_w, h)
        yellow_xs, white_xs = self._apply_line_memory(yellow_xs, white_xs)

        left_det = len(yellow_xs) > 0
        right_det = len(white_xs) > 0

        both_visible = left_det and right_det
        one_visible = left_det or right_det

        if one_visible:
            self._lost_lane_frames = 0
        else:
            self._lost_lane_frames += 1

        recovery = self._lost_lane_frames > self._max_lost_lane_frames

        is_curve, curve_dir = self._confirmed_curve(yellow_xs, white_xs)

        raw_error = self._calculate_error(yellow_xs, white_xs, left_det, right_det, w)

        if is_curve:
            self._filtered_error = 0.86 * self._filtered_error + 0.14 * raw_error
        else:
            self._filtered_error = 0.94 * self._filtered_error + 0.06 * raw_error

        steering = self._calculate_steering(self._filtered_error, is_curve)

        left, right = self._motor_commands(
            steering,
            recovery,
            is_curve,
            both_visible,
            one_visible,
        )

        left, right = self._smooth(left, right, both_visible)

        combined = np.clip(mask_left + mask_right, 0, 1)

        self.last_debug_info = {
            "roi": image,
            "lane_mask": (combined * 255).astype(np.uint8),
            "white_mask": mask_w,
            "yellow_mask": mask_y,
            "total_lane_pixels": total_pixels,
            "lateral_error": float(np.clip(self._prev_error, -1.0, 1.0)),
            "lane_detected": not recovery,
            "frame_count": self.frame_count,
            "yellow_xs": yellow_xs,
            "white_xs": white_xs,
            "slice_ys": [int(h * r) for r in _SLICE_Y_RATIOS],
            "is_curve": is_curve,
            "curve_dir": curve_dir,
            "lost_lane_frames": self._lost_lane_frames,
            "curve_counter": self._curve_counter,
        }

        if self.frame_count % 20 == 0:
            print(
                f"[LaneServoingAgent] yellow_xs={yellow_xs}, white_xs={white_xs}, "
                f"curve={is_curve}, curve_count={self._curve_counter}, "
                f"lost={self._lost_lane_frames}, err={self._filtered_error:.3f}, "
                f"steer={steering:.3f}, left={left:.3f}, right={right:.3f}, "
                f"lane={not recovery}"
            )

        return left, right

    def step(self, image: np.ndarray, wheels_driver) -> Tuple[float, float]:
        left, right = self.compute_commands(image)
        wheels_driver.set_wheels_speed(left, right)
        return left, right

    def get_debug_info(self, image: np.ndarray) -> dict:
        return self.last_debug_info

    def _empty_debug_info(self, h, w):
        return {
            "roi": np.zeros((h, w, 3), dtype=np.uint8),
            "lane_mask": np.zeros((h, w), dtype=np.uint8),
            "white_mask": np.zeros((h, w), dtype=np.uint8),
            "yellow_mask": np.zeros((h, w), dtype=np.uint8),
            "total_lane_pixels": 0,
            "lateral_error": 0.0,
            "lane_detected": False,
            "frame_count": 0,
            "yellow_xs": [],
            "white_xs": [],
            "slice_ys": [],
            "is_curve": False,
            "curve_dir": 0,
            "lost_lane_frames": 0,
            "curve_counter": 0,
        }