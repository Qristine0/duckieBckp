"""
sign_behavior.py
================
AprilTag-based sign detection and intersection state machine for Duckiebot.

Tag ID assignments (tag36h11 family):
    0  -> STOP             : full stop before red line, sweep left/right for
                             cars, if clear drive forward post_stop_frames, resume
    1  -> YIELD            : slow down at red line; if car seen → full stop +
                             sweep; if clear → sweep then post-stop forward
    2  -> TURN_LEFT_FWD    : intersection allows left or forward
    3  -> TURN_LEFT_RIGHT  : intersection allows left or right
    4  -> TURN_RIGHT_FWD   : intersection allows right or forward
    5  -> TURN_ALL         : intersection allows all directions

States
------
MOVING      : normal lane-following; signs saved passively
SLOWING     : ramp speed toward zero (STOP path) or toward yield_min_speed
              (YIELD path)
YIELD_CREEP : yield path — moving slowly past the red line while checking
              for cars; car seen → STOPPED; none seen → CHECKPATH
STOPPED     : halted, waiting stop_hold_frames
CHECKPATH   : sweep left then right to check for oncoming cars;
              car seen → redo sweep; clear → POST_STOP
POST_STOP   : drive straight forward post_stop_frames frames → MOVING
INTERSECT   : pick a random permitted direction and execute the turn manoeuvre
              (no car-check, no stop); done → EXITING
EXITING     : drive straight exit_frames frames to clear intersection → MOVING

Red-line lockout
----------------
Once a red line triggers any action (SLOWING or YIELD_CREEP) the robot
ignores further red-line readings until it returns to MOVING.

Intersection geometry (robot facing up, at position A)
------------------------------------------------------
       ↑ forward
  _ _
       |
 |  __(A)

  right turn  : curve right — does NOT cross any red line
  left turn   : curve left across intersection
  forward     : straight ahead

The robot decides its direction from the intersection tag the moment it
enters INTERSECT, executes the fixed manoeuvre, then exits.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import IntEnum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np
from pupil_apriltags import Detector as _Detector
from tasks.sign_detection.packages.red_line_detection import detect_red_line
from tasks.sign_detection.packages.april_tag import detect_tags, confirm_tags

# ---------------------------------------------------------------------------
# Tag ID → meaning
# ---------------------------------------------------------------------------
class TagID(IntEnum):
    STOP            = 0
    YIELD           = 1
    TURN_LEFT_FWD   = 2
    TURN_LEFT_RIGHT = 3
    TURN_RIGHT_FWD  = 4
    TURN_ALL        = 5


# Intersection tags and their permitted directions
_TAG_TURNS: dict[int, List[str]] = {
    TagID.TURN_LEFT_RIGHT: ["left",  "right"],
    TagID.TURN_LEFT_FWD:   ["left",  "forward"],
    TagID.TURN_RIGHT_FWD:  ["right", "forward"],
    TagID.TURN_ALL:        ["left",  "right", "forward"],
}


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------
class State(IntEnum):
    MOVING      = auto()
    SLOWING     = auto()   # ramping down (used by both STOP and YIELD paths)
    YIELD_CREEP = auto()   # yield: creeping forward slowly, watching for cars
    STOPPED     = auto()   # full stop, holding stop_hold_frames
    CHECKPATH   = auto()   # sweep left/right to check for cars → POST_STOP
    POST_STOP   = auto()   # drive straight briefly then → MOVING
    INTERSECT   = auto()   # pick direction, execute turn manoeuvre
    EXITING     = auto()   # drive straight to clear intersection → MOVING


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class SignBehaviorConfig:
    # ── Red-line detection ────────────────────────────────────────────────
    red_strip_frac: float = 0.50
    red_pixel_frac: float = 0.03
    red_hsv_low1:  Tuple[int, int, int] = (0,   120,  80)
    red_hsv_high1: Tuple[int, int, int] = (10,  255, 255)
    red_hsv_low2:  Tuple[int, int, int] = (170, 120,  80)
    red_hsv_high2: Tuple[int, int, int] = (180, 255, 255)

    # ── Stop-line hold ────────────────────────────────────────────────────
    # Frames to sit still at the red line before acting (STOP sign)
    stop_hold_frames: int = 20

    # ── Speed ramp (shared by STOP and YIELD paths) ───────────────────────
    slow_ramp_factor:        float = 0.92
    stopped_speed_threshold: float = 0.05   # below this → considered stopped

    # ── YIELD specific ───────────────────────────────────────────────────
    # Minimum creep speed while passing the line under a yield sign.
    yield_min_speed:    float = 0.30
    # How many frames to creep forward while checking for cars
    yield_creep_frames: int   = 15

    # ── CHECKPATH sweep (STOP and YIELD paths) ────────────────────────────
    # Phase A: rotate left  check_left_frames  frames  (look left)
    # Phase B: rotate right check_right_frames frames  (look right, 2x to sweep back)
    # Phase C: rotate left  check_left_frames  frames  (return to forward)
    check_left_frames:  int   = 5
    check_right_frames: int   = 5
    check_turn_speed:   float = 0.10   # in-place rotation speed during sweep

    # ── POST_STOP ─────────────────────────────────────────────────────────
    post_stop_frames: int   = 12
    post_stop_speed:  float = 0.2

    # ── Intersection manoeuvres ───────────────────────────────────────────
    intersect_forward_frames: int   = 15
    intersect_left_frames:    int   = 18
    intersect_right_frames:   int   = 18
    intersect_forward_speed:  Tuple[float, float] = (0.20, 0.20)
    intersect_left_speed:     Tuple[float, float] = (0.00, 0.18)
    intersect_right_speed:    Tuple[float, float] = (0.18, 0.00)

    # ── Exiting ───────────────────────────────────────────────────────────
    exit_frames: int   = 30
    exit_speed:  float = 0.20

    # ── Vehicle detection ─────────────────────────────────────────────────
    vehicle_class_id:      int   = 1
    vehicle_min_bbox_area: float = 1500.0

    # ── AprilTag detector ────────────────────────────────────────────────
    min_margin:         float = 10.0
    tag_confirm_frames: int   = 3
    camera_params: Optional[Tuple[float, float, float, float]] = None
    tag_size_m: float = 0.065


# ---------------------------------------------------------------------------
# FSM
# ---------------------------------------------------------------------------
class SignBehaviorFSM:
    """
    Sign-driven intersection state machine.

    STOP sign flow
    --------------
    MOVING → (red line) → SLOWING → STOPPED (hold) →
        CHECKPATH (sweep left/right) → POST_STOP → MOVING

    YIELD sign flow — no car during creep
    --------------------------------------
    MOVING → (red line) → SLOWING (to yield_min_speed) →
        YIELD_CREEP (slow forward, watching) → CHECKPATH → POST_STOP → MOVING

    YIELD sign flow — car detected during creep
    -------------------------------------------
    … → YIELD_CREEP → STOPPED (full stop, hold) → CHECKPATH → POST_STOP → MOVING

    Intersection sign flow
    ----------------------
    MOVING → (red line) → INTERSECT (pick direction, execute turn) →
        EXITING (drive straight) → MOVING
    No car check, no hold.
    """

    def __init__(self, config: SignBehaviorConfig = None):
        self.cfg   = config or SignBehaviorConfig()
        self.state: State = State.MOVING

        self._detector = _Detector(families="tag36h11", nthreads=2)
        print("[SignBehavior] initialised — detector ready")

        # Tag confirmation buffer
        self._tag_buffer: dict[int, int] = {}

        # Saved sign — set while MOVING, consumed when red line fires
        self._saved_tag: Optional[int] = None

        # Per-state counters
        self._hold_counter:  int = 0
        self._turn_counter:  int = 0   # INTERSECT / EXITING / POST_STOP
        self._creep_counter: int = 0   # YIELD_CREEP
        self._check_counter: int = 0   # CHECKPATH sweep

        # CHECKPATH vehicle flags
        self._vehicle_seen_left:  bool = False
        self._vehicle_seen_right: bool = False

        # Intersection direction
        self._possible_turns: List[str]     = []
        self._chosen_turn:    Optional[str] = None

        # Speed ramp state
        self._slow_factor: float = 1.0
        # 0.0 → full stop (STOP sign), yield_min_speed → creep (YIELD sign)
        self._slow_target: float = 0.0

        # Red-line lockout
        self._red_line_locked: bool = False

        # Public debug dict
        self.debug: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(
        self,
        frame_rgb:  np.ndarray,
        base_left:  float,
        base_right: float,
        detections: list,
    ) -> Tuple[float, float]:
        detections = detections or []

        tags      = detect_tags(self, frame_rgb)
        confirmed = confirm_tags(self, tags)

        red_line = False if self._red_line_locked else bool(detect_red_line(self, frame_rgb))

        left, right = self._fsm_step(confirmed, detections, red_line, base_left, base_right)

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
    def state_name(self) -> str:
        return self.state.name

    # ------------------------------------------------------------------
    # FSM core
    # ------------------------------------------------------------------

    def _fsm_step(
        self,
        confirmed_tags: List[int],
        detections:     list,
        red_line:       bool,
        base_left:      float,
        base_right:     float,
    ) -> Tuple[float, float]:

        # ── MOVING ──────────────────────────────────────────────────────────
        if self.state == State.MOVING:
            self._slow_factor = 1.0

            # Save the most relevant sign seen while moving.
            # Priority: intersection tag > STOP > YIELD
            for tag_id in confirmed_tags:
                if tag_id in _TAG_TURNS:
                    if self._saved_tag != tag_id:
                        print(f"[SignBehavior] sign saved: intersection tag {tag_id} "
                              f"→ {_TAG_TURNS[tag_id]}")
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
                    # Intersection: pick direction immediately, no stop/check
                    self._possible_turns = _TAG_TURNS[tag]
                    self._chosen_turn    = self._pick_turn()
                    self._turn_counter   = 0
                    self._saved_tag      = None
                    self.state           = State.INTERSECT
                    print(f"[SignBehavior] >>> INTERSECT — direction: {self._chosen_turn} "
                          f"(from tag {tag}, options {self._possible_turns})")

                elif tag == int(TagID.YIELD):
                    # Yield: ramp down to creep speed, then watch for cars
                    self._slow_factor  = 1.0
                    self._slow_target  = self.cfg.yield_min_speed
                    self._saved_tag    = None
                    self.state         = State.SLOWING
                    print("[SignBehavior] >>> YIELD — slowing to creep speed")

                else:
                    # STOP sign (tag 0) or no tag: full stop
                    self._slow_factor  = 1.0
                    self._slow_target  = 0.0
                    self._saved_tag    = None
                    self.state         = State.SLOWING
                    print(f"[SignBehavior] >>> STOP — full stop before red line "
                          f"(saved_tag={tag})")

            return base_left, base_right

        # ── SLOWING ─────────────────────────────────────────────────────────
        elif self.state == State.SLOWING:
            target = self._slow_target
            self._slow_factor *= self.cfg.slow_ramp_factor
            factor = max(self._slow_factor, target)
            left   = base_left  * factor
            right  = base_right * factor

            if target == 0.0:
                # STOP path: wait until essentially stopped
                if self._slow_factor < self.cfg.stopped_speed_threshold:
                    self._slow_factor  = 0.0
                    self._hold_counter = 0
                    self.state         = State.STOPPED
                    print("[SignBehavior] >>> STOPPED at red line")
                    return 0.0, 0.0
            else:
                # YIELD path: wait until ramped to creep speed
                if self._slow_factor <= target + 0.02:
                    self._creep_counter = 0
                    self.state          = State.YIELD_CREEP
                    print(f"[SignBehavior] >>> YIELD_CREEP at speed ~{target:.2f}")
                    spd = self.cfg.yield_min_speed
                    return spd, spd

            return left, right

        # ── YIELD_CREEP ──────────────────────────────────────────────────────
        elif self.state == State.YIELD_CREEP:
            if self._creep_counter < self.cfg.yield_creep_frames:
                self._creep_counter += 1
                if self._vehicle_detected(detections):
                    # Car seen — full stop, then sweep
                    self._hold_counter = 0
                    self.state         = State.STOPPED
                    print("[SignBehavior] >>> vehicle seen during yield creep — STOPPED")
                    return 0.0, 0.0
                return self.cfg.yield_min_speed, self.cfg.yield_min_speed

            # Creep complete, no car — go straight to sweep
            self._check_counter      = 0
            self._vehicle_seen_left  = False
            self._vehicle_seen_right = False
            self.state               = State.CHECKPATH
            print("[SignBehavior] >>> YIELD_CREEP done — CHECKPATH")
            return 0.0, 0.0

        # ── STOPPED ─────────────────────────────────────────────────────────
        elif self.state == State.STOPPED:
            self._hold_counter += 1
            if self._hold_counter < self.cfg.stop_hold_frames:
                return 0.0, 0.0

            # Hold complete — sweep for cars
            self._check_counter      = 0
            self._vehicle_seen_left  = False
            self._vehicle_seen_right = False
            self.state               = State.CHECKPATH
            print("[SignBehavior] >>> CHECKPATH — sweeping for cars")
            return 0.0, 0.0

        # ── CHECKPATH ────────────────────────────────────────────────────────
        elif self.state == State.CHECKPATH:
            return self._checkpath_step(detections)

        # ── POST_STOP ────────────────────────────────────────────────────────
        elif self.state == State.POST_STOP:
            if self._turn_counter < self.cfg.post_stop_frames:
                self._turn_counter += 1
                return self.cfg.post_stop_speed, self.cfg.post_stop_speed

            self._red_line_locked = False
            self.state            = State.MOVING
            print("[SignBehavior] >>> POST_STOP done — MOVING (red-line unlocked)")
            return base_left, base_right

        # ── INTERSECT ────────────────────────────────────────────────────────
        elif self.state == State.INTERSECT:
            return self._intersect_step(base_left, base_right)

        # ── EXITING ─────────────────────────────────────────────────────────
        elif self.state == State.EXITING:
            return self._exiting_step(base_left, base_right)

        return base_left, base_right

    # ------------------------------------------------------------------
    # CHECKPATH
    # ------------------------------------------------------------------
    # 3-phase sweep so the robot returns to its original heading:
    #   Phase A (0 → cl)             : rotate LEFT  (look left)
    #   Phase B (cl → cl + 2*cr)     : rotate RIGHT (sweep right, 2x to return)
    #   Phase C (cl+2*cr → 2*(cl+cr)): rotate LEFT  (return to forward)

    def _checkpath_step(self, detections: list) -> Tuple[float, float]:
        spd = self.cfg.check_turn_speed
        cl  = self.cfg.check_left_frames
        cr  = self.cfg.check_right_frames
        c   = self._check_counter

        if c < cl:                      # Phase A: look left
            if self._vehicle_detected(detections):
                self._vehicle_seen_left = True
            self._check_counter += 1
            return -spd, spd

        elif c < cl + 2 * cr:           # Phase B: sweep right
            if self._vehicle_detected(detections):
                self._vehicle_seen_right = True
            self._check_counter += 1
            return spd, -spd

        elif c < 2 * (cl + cr):         # Phase C: return to forward
            self._check_counter += 1
            return -spd, spd

        else:                           # Sweep complete
            if self._vehicle_seen_left or self._vehicle_seen_right:
                print("[SignBehavior] vehicle seen during sweep — redoing check")
                self._check_counter      = 0
                self._vehicle_seen_left  = False
                self._vehicle_seen_right = False
                return 0.0, 0.0

            # Path clear — drive forward
            self._turn_counter = 0
            self.state         = State.POST_STOP
            print("[SignBehavior] >>> path clear — POST_STOP")
            return 0.0, 0.0

    # ------------------------------------------------------------------
    # INTERSECT
    # ------------------------------------------------------------------

    def _intersect_step(
        self,
        base_left:  float,
        base_right: float,
    ) -> Tuple[float, float]:
        turn = self._chosen_turn or "forward"

        frames_map = {
            "forward": self.cfg.intersect_forward_frames,
            "left":    self.cfg.intersect_left_frames,
            "right":   self.cfg.intersect_right_frames,
        }
        speeds_map = {
            "forward": self.cfg.intersect_forward_speed,
            "left":    self.cfg.intersect_left_speed,
            "right":   self.cfg.intersect_right_speed,
        }

        total = frames_map.get(turn, self.cfg.intersect_forward_frames)

        if self._turn_counter < total:
            self._turn_counter += 1
            return speeds_map.get(turn, (base_left, base_right))

        self._turn_counter = 0
        self.state         = State.EXITING
        print(f"[SignBehavior] >>> INTERSECT '{turn}' done — "
              f"EXITING ({self.cfg.exit_frames} frames)")
        return base_left, base_right

    # ------------------------------------------------------------------
    # EXITING
    # ------------------------------------------------------------------

    def _exiting_step(
        self,
        base_left:  float,
        base_right: float,
    ) -> Tuple[float, float]:
        if self._turn_counter < self.cfg.exit_frames:
            self._turn_counter += 1
            return self.cfg.exit_speed, self.cfg.exit_speed

        self._red_line_locked = False
        self._possible_turns  = []
        self._chosen_turn     = None
        self.state            = State.MOVING
        print("[SignBehavior] >>> intersection cleared — MOVING (red-line unlocked)")
        return base_left, base_right

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _vehicle_detected(self, detections: list) -> bool:
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

    def _pick_turn(self) -> str:
        return random.choice(self._possible_turns) if self._possible_turns else "forward"


# ---------------------------------------------------------------------------
# Debug overlay (BGR frame, in-place)
# ---------------------------------------------------------------------------
def draw_sign_debug(bgr: np.ndarray, fsm: SignBehaviorFSM) -> np.ndarray:
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