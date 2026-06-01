"""
sign_behavior.py
================
AprilTag-based sign detection and intersection state machine for Duckiebot.
"""

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from tasks.sign_detection.packages.red_line_detection import detect_red_line
from tasks.sign_detection.packages.april_tag import detect_tags, confirm_tags

# ---------------------------------------------------------------------------
# Tag ID -> meaning
# ---------------------------------------------------------------------------
class TagID(IntEnum):
    STOP            = 1
    YIELD           = 0
    TURN_LEFT_FWD   = 0
    TURN_LEFT_RIGHT = 3
    TURN_RIGHT_FWD  = 4
    TURN_ALL        = 5


_TAG_TURNS = {
    TagID.TURN_LEFT_RIGHT: ["left",  "right"],
    TagID.TURN_LEFT_FWD:   ["left"],
    TagID.TURN_RIGHT_FWD:  ["right", "forward"],
    TagID.TURN_ALL:        ["left",  "right", "forward"],
}  # type: Dict[int, List[str]]


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class State(IntEnum):
    MOVING      = auto()
    SLOWING     = auto()
    YIELD_CREEP = auto()
    STOPPED     = auto()
    CHECKPATH   = auto()
    POST_STOP   = auto()
    INTERSECT   = auto()
    PRE_TURN    = auto()
    TURNING     = auto()
    EXITING     = auto()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SignBehaviorConfig:
    # Red-line detection
    red_strip_frac  = 0.50
    red_pixel_frac  = 0.03
    red_hsv_low1    = (0,   120,  80)   # type: Tuple[int, int, int]
    red_hsv_high1   = (10,  255, 255)   # type: Tuple[int, int, int]
    red_hsv_low2    = (170, 120,  80)   # type: Tuple[int, int, int]
    red_hsv_high2   = (180, 255, 255)   # type: Tuple[int, int, int]

    # Pre-turn alignment
    preturn_align_tolerance = 0.03
    preturn_align_speed     = 0.001

    # Stop-line hold
    stop_hold_frames = 0

    # Speed ramp
    slow_ramp_factor        = 0.92
    stopped_speed_threshold = 0.05

    # YIELD specific
    yield_min_speed    = 0.1
    yield_creep_frames = 5

    # CHECKPATH sweep
    check_left_frames   = 4
    check_right_frames  = 4
    check_turn_speed    = 0.04
    check_settle_frames = 5

    # POST_STOP
    post_stop_frames = 12
    post_stop_speed  = 0.2

    # Pre-turn forward creep
    preturn_right_frames = 30
    preturn_left_frames  = 40
    preturn_speed        = 0.20

    # Intersection manoeuvres
    intersect_forward_frames = 5
    intersect_left_frames    = 9
    intersect_right_frames   = 9
    intersect_forward_speed  = (0.20, 0.20)  # type: Tuple[float, float]
    intersect_left_speed     = (0.00, 0.18)  # type: Tuple[float, float]
    intersect_right_speed    = (0.18, 0.00)  # type: Tuple[float, float]

    # Exiting
    exit_speed            = 0.20
    exit_right_min_frac   = 0.02
    exit_timeout_frames   = 10
    exit_post_line_frames = 5
    exit_frames_forward   = 8
    exit_frames_left      = 8
    exit_frames_right     = 2

    # Vehicle detection
    vehicle_class_id      = 1
    vehicle_min_bbox_area = 2000.0

    # AprilTag detector
    min_margin         = 10.0
    tag_confirm_frames = 3
    camera_params      = None   # type: Optional[Tuple[float, float, float, float]]
    tag_size_m         = 0.065


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------
class SignBehaviorFSM:

    def __init__(self, config=None):
        # type: (Optional[SignBehaviorConfig]) -> None
        self.cfg   = config or SignBehaviorConfig()
        self.state = State.MOVING  # type: State

        self._preturn_correction = 0.0  # type: float

        print("[SignBehavior] initialised — detector ready")

        self._tag_buffer  = {}    # type: Dict[int, int]
        self._saved_tag   = None  # type: Optional[int]
        self._exit_line_seen = False  # type: bool

        self._hold_counter  = 0  # type: int
        self._turn_counter  = 0  # type: int
        self._creep_counter = 0  # type: int
        self._check_counter = 0  # type: int

        self._vehicle_seen_left  = False  # type: bool
        self._vehicle_seen_right = False  # type: bool

        self._possible_turns = []    # type: List[str]
        self._chosen_turn    = None  # type: Optional[str]

        self._slow_factor = 1.0   # type: float
        self._slow_target = 0.0   # type: float

        self._red_line_locked = False  # type: bool

        self.debug = {}  # type: dict

        self._frame_rgb = None  # type: Optional[np.ndarray]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self, frame_rgb, base_left, base_right, detections):
        # type: (np.ndarray, float, float, list) -> Tuple[float, float]
        self._frame_rgb = frame_rgb
        detections = detections or []

        tags      = detect_tags(self, frame_rgb)
        confirmed = confirm_tags(self, tags)

        red_line = False if self._red_line_locked else bool(detect_red_line(self, frame_rgb))

        left, right = self._fsm_step(confirmed, detections, red_line, base_left, base_right, frame_rgb)

        self.debug = {
            "state":          self.state.name,
            "confirmed_tags": [int(t) for t in confirmed],
            "saved_tag":      int(self._saved_tag) if self._saved_tag is not None else None,
            "red_line":       red_line,
            "red_locked":     self._red_line_locked,
            "possible_turns": list(self._possible_turns),
            "chosen_turn":    self._chosen_turn,
            "slow_factor":    round(float(self._slow_factor), 3),
            "hold_counter":   int(self._hold_counter),
            "creep_counter":  int(self._creep_counter),
            "check_counter":  int(self._check_counter),
            "turn_counter":   int(self._turn_counter),
        }
        return left, right

    @property
    def state_name(self):
        # type: () -> str
        return self.state.name

    # ------------------------------------------------------------------
    # FSM core
    # ------------------------------------------------------------------

    def _fsm_step(self, confirmed_tags, detections, red_line, base_left, base_right, frame_rgb):
        # type: (List[int], list, bool, float, float, np.ndarray) -> Tuple[float, float]

        # MOVING
        if self.state == State.MOVING:
            self._slow_factor = 1.0

            for tag_id in confirmed_tags:
                if tag_id in _TAG_TURNS:
                    if self._saved_tag != tag_id:
                        print(f"[SignBehavior] sign saved: intersection tag {tag_id} "
                              f"-> {_TAG_TURNS[tag_id]}")
                    self._saved_tag = tag_id
                    break
            else:
                for tag_id in confirmed_tags:
                    if tag_id in (int(TagID.STOP), int(TagID.YIELD)):
                        if self._saved_tag != tag_id:
                            label = "STOP" if tag_id == TagID.STOP else "YIELD"
                            print(f"[SignBehavior] sign saved: {label}")
                        self._saved_tag = tag_id
                        break

            if red_line:
                self._red_line_locked = True
                tag = self._saved_tag

                if tag in _TAG_TURNS:
                    self._possible_turns = _TAG_TURNS[tag]
                    self._chosen_turn    = self._pick_turn()
                    self._turn_counter   = 0
                    self._saved_tag      = None
                    self.state           = State.INTERSECT
                    print(f"[SignBehavior] >>> INTERSECT — direction: {self._chosen_turn} "
                          f"(from tag {tag}, options {self._possible_turns})")

                elif tag == int(TagID.YIELD):
                    self._slow_factor  = 1.0
                    self._slow_target  = self.cfg.yield_min_speed
                    self._saved_tag    = None
                    self.state         = State.SLOWING
                    print("[SignBehavior] >>> YIELD — slowing to creep speed")

                elif tag == int(TagID.STOP):
                    self._slow_factor  = 1.0
                    self._slow_target  = 0.0
                    self._saved_tag    = None
                    self.state         = State.SLOWING
                    print(f"[SignBehavior] >>> STOP — full stop before red line "
                          f"(saved_tag={tag})")
                else:
                    self._chosen_turn  = "forward"
                    self._turn_counter = 0
                    self._saved_tag    = None
                    self.state         = State.INTERSECT
                    print(f"[SignBehavior] >>> INTERSECT — direction: {self._chosen_turn} "
                          f"(No tag (Default behavior))")

            return base_left, base_right

        # SLOWING
        elif self.state == State.SLOWING:
            target = self._slow_target
            self._slow_factor *= self.cfg.slow_ramp_factor
            factor = max(self._slow_factor, target)
            left   = base_left  * factor
            right  = base_right * factor

            if target == 0.0:
                if self._slow_factor < self.cfg.stopped_speed_threshold:
                    self._slow_factor  = 0.0
                    self._hold_counter = 0
                    self.state         = State.STOPPED
                    print("[SignBehavior] >>> STOPPED at red line")
                    return 0.0, 0.0
            else:
                if self._slow_factor <= target + 0.02:
                    self._creep_counter = 0
                    self.state          = State.YIELD_CREEP
                    print(f"[SignBehavior] >>> YIELD_CREEP at speed ~{target:.2f}")
                    spd = self.cfg.yield_min_speed
                    return spd, spd

            return left, right

        # YIELD_CREEP
        elif self.state == State.YIELD_CREEP:
            if self._creep_counter < self.cfg.yield_creep_frames:
                self._creep_counter += 1
                if self._vehicle_detected(detections):
                    self._hold_counter = 0
                    self.state         = State.STOPPED
                    print("[SignBehavior] >>> vehicle seen during yield creep — STOPPED")
                    return 0.0, 0.0
                return self.cfg.yield_min_speed, self.cfg.yield_min_speed

            self._check_counter      = 0
            self._vehicle_seen_left  = False
            self._vehicle_seen_right = False
            self.state               = State.CHECKPATH
            print("[SignBehavior] >>> YIELD_CREEP done — CHECKPATH")
            return 0.0, 0.0

        # STOPPED
        elif self.state == State.STOPPED:
            self._hold_counter += 1
            if self._hold_counter < self.cfg.stop_hold_frames:
                return 0.0, 0.0

            self._check_counter      = 0
            self._vehicle_seen_left  = False
            self._vehicle_seen_right = False
            self.state               = State.CHECKPATH
            print("[SignBehavior] >>> CHECKPATH — sweeping for cars")
            return 0.0, 0.0

        # CHECKPATH
        elif self.state == State.CHECKPATH:
            return self._checkpath_step(detections)

        # POST_STOP
        elif self.state == State.POST_STOP:
            if self._turn_counter < self.cfg.post_stop_frames:
                self._turn_counter += 1
                return self.cfg.post_stop_speed, self.cfg.post_stop_speed

            self._red_line_locked = False
            self.state            = State.MOVING
            print("[SignBehavior] >>> POST_STOP done — MOVING (red-line unlocked)")
            return base_left, base_right

        # INTERSECT
        elif self.state == State.INTERSECT:
            return self._intersect_step(base_left, base_right)

        elif self.state == State.PRE_TURN:
            return self._preturn_step(base_left, base_right)

        elif self.state == State.TURNING:
            return self._turning_step(base_left, base_right)

        # EXITING
        elif self.state == State.EXITING:
            return self._exiting_step(base_left, base_right)

        return base_left, base_right

    # ------------------------------------------------------------------
    # CHECKPATH
    # ------------------------------------------------------------------

    def _checkpath_step(self, detections):
        # type: (list) -> Tuple[float, float]
        spd = self.cfg.check_turn_speed
        cl  = self.cfg.check_left_frames
        cr  = self.cfg.check_right_frames
        cs  = self.cfg.check_settle_frames
        c   = self._check_counter

        phase_a_end    = cl
        phase_a_settle = cl + cs
        phase_b_end    = phase_a_settle + 2 * cr
        phase_b_settle = phase_b_end + cs
        phase_c_end    = phase_b_settle + cl

        if c < phase_a_end:
            if self._vehicle_detected(detections):
                self._vehicle_seen_left = True
            self._check_counter += 1
            return -spd, spd

        elif c < phase_a_settle:
            if self._vehicle_detected(detections):
                self._vehicle_seen_left = True
            self._check_counter += 1
            return 0.0, 0.0

        elif c < phase_b_end:
            if self._vehicle_detected(detections):
                self._vehicle_seen_right = True
            self._check_counter += 1
            return spd, -spd

        elif c < phase_b_settle:
            if self._vehicle_detected(detections):
                self._vehicle_seen_right = True
            self._check_counter += 1
            return 0.0, 0.0

        elif c < phase_c_end:
            self._check_counter += 1
            return -spd, spd

        else:
            if self._vehicle_seen_left or self._vehicle_seen_right:
                print("[SignBehavior] vehicle seen during sweep — redoing check")
                self._check_counter      = 0
                self._vehicle_seen_left  = False
                self._vehicle_seen_right = False
                return 0.0, 0.0

            self._turn_counter = 0
            self.state         = State.POST_STOP
            print("[SignBehavior] >>> path clear — POST_STOP")
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # INTERSECT
    # ------------------------------------------------------------------

    def _intersect_step(self, base_left, base_right):
        # type: (float, float) -> Tuple[float, float]
        turn = self._chosen_turn or "forward"
        self._turn_counter = 0

        if turn in ("left", "right"):
            self.state = State.PRE_TURN
            print(f"[SignBehavior] >>> PRE_TURN for '{turn}'")
            return self.cfg.preturn_speed, self.cfg.preturn_speed

        self.state = State.TURNING
        print(f"[SignBehavior] >>> TURNING 'forward' (no pre-turn)")
        return self.cfg.intersect_forward_speed

    def _preturn_step(self, base_left, base_right):
        # type: (float, float) -> Tuple[float, float]
        turn  = self._chosen_turn or "forward"
        total = (self.cfg.preturn_right_frames if turn == "right"
                 else self.cfg.preturn_left_frames)

        if self._turn_counter < total:
            self._turn_counter += 1
            return self.cfg.preturn_speed, self.cfg.preturn_speed

        self._turn_counter = 0
        self.state = State.TURNING
        print(f"[SignBehavior] >>> PRE_TURN done — TURNING '{turn}'")
        return 0.0, 0.0

    def _turning_step(self, base_left, base_right):
        # type: (float, float) -> Tuple[float, float]
        turn = self._chosen_turn or "forward"
        speeds_map = {
            "forward": self.cfg.intersect_forward_speed,
            "left":    self.cfg.intersect_left_speed,
            "right":   self.cfg.intersect_right_speed,
        }
        frames_map = {
            "forward": self.cfg.intersect_forward_frames,
            "left":    self.cfg.intersect_left_frames,
            "right":   self.cfg.intersect_right_frames,
        }
        total = frames_map.get(turn, self.cfg.intersect_forward_frames)

        if self._turn_counter < total:
            self._turn_counter += 1
            return speeds_map.get(turn, (base_left, base_right))

        self._turn_counter   = 0
        self._exit_line_seen = False
        self.state           = State.EXITING
        print(f"[SignBehavior] >>> TURNING '{turn}' done — EXITING")
        return base_left, base_right

    # ------------------------------------------------------------------
    # EXITING
    # ------------------------------------------------------------------

    def _exiting_step(self, base_left, base_right):
        # type: (float, float) -> Tuple[float, float]
        frame_rgb = self._frame_rgb

        if not self._exit_line_seen:
            if frame_rgb is not None:
                h, w    = frame_rgb.shape[:2]
                strip_h = max(2, int(h * self.cfg.red_strip_frac))
                strip   = frame_rgb[h - strip_h:, :]
                hsv     = cv2.cvtColor(strip, cv2.COLOR_RGB2HSV)
                lo1 = np.array(self.cfg.red_hsv_low1, dtype=np.uint8)
                hi1 = np.array(self.cfg.red_hsv_high1, dtype=np.uint8)
                lo2 = np.array(self.cfg.red_hsv_low2, dtype=np.uint8)
                hi2 = np.array(self.cfg.red_hsv_high2, dtype=np.uint8)
                mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

                half       = w // 2
                right_frac = float(mask[:, half:].sum()) / (255.0 * strip_h * (w - half))
                total_frac = float(mask.sum()) / (255.0 * strip_h * w)

                line_seen     = total_frac >= self.cfg.red_pixel_frac
                not_left_only = right_frac >= self.cfg.exit_right_min_frac

                if line_seen and not_left_only:
                    print(f"[SignBehavior] EXITING: exit line seen — crossing it")
                    self._exit_line_seen = True
                    self._turn_counter   = 0

            self._turn_counter += 1
            if self._turn_counter >= self.cfg.exit_timeout_frames:
                print("[SignBehavior] EXITING: timeout — MOVING")
                self._exit_line_seen  = False
                self._red_line_locked = False
                self._possible_turns  = []
                self._chosen_turn     = None
                self.state            = State.MOVING
                return base_left, base_right

            return self.cfg.exit_speed, self.cfg.exit_speed

        else:
            if self._turn_counter < self.cfg.exit_post_line_frames:
                self._turn_counter += 1
                return self.cfg.exit_speed, self.cfg.exit_speed

            print("[SignBehavior] EXITING: past exit line — MOVING (red-line unlocked)")
            self._exit_line_seen  = False
            self._red_line_locked = False
            self._possible_turns  = []
            self._chosen_turn     = None
            self.state            = State.MOVING
            return base_left, base_right

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vehicle_detected(self, detections):
        # type: (list) -> bool
        for det in detections:
            bbox, score, cls_id = det
            if cls_id != self.cfg.vehicle_class_id:
                continue
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            if area >= self.cfg.vehicle_min_bbox_area:
                print(f"[SignBehavior] vehicle detected (area={area:.0f})")
                return True
        return False

    def _pick_turn(self):
        # type: () -> str
        return random.choice(self._possible_turns) if self._possible_turns else "forward"


# ---------------------------------------------------------------------------
# Debug overlay (BGR frame, in-place)
# ---------------------------------------------------------------------------
def draw_sign_debug(bgr, fsm):
    # type: (np.ndarray, SignBehaviorFSM) -> np.ndarray
    d = fsm.debug
    y = 30
    lines = [
        f"State:    {d.get('state', '?')}",
        f"Tags:     {d.get('confirmed_tags', [])}",
        f"Saved:    {d.get('saved_tag', '-')}",
        f"RedLine:  {d.get('red_line', False)}  locked={d.get('red_locked', False)}",
        f"Turns:    {d.get('possible_turns', [])}",
        f"Turn:     {d.get('chosen_turn', '-')}",
        f"SlowF:    {d.get('slow_factor', 1.0):.2f}",
        f"Counters: hold={d.get('hold_counter', 0)} "
        f"creep={d.get('creep_counter', 0)} "
        f"chk={d.get('check_counter', 0)} "
        f"t={d.get('turn_counter', 0)}",
    ]
    for line in lines:
        cv2.putText(bgr, line, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2, cv2.LINE_AA)
        y += 22
    return bgr