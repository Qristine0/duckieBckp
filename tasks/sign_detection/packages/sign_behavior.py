import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from tasks.sign_detection.packages.red_line_detection import detect_red_line
from tasks.sign_detection.packages.april_tag import detect_tags, confirm_tags
from tasks.sign_detection.packages.detection import vehicle_detected
from tasks.sign_detection.packages.sign_behavior_config import (
    TagID,
    _TAG_TURNS,
    SignBehaviorConfig,
    State,
    resolve_tag,
)


class SignBehaviorFSM:

    def __init__(self, config=None):
        self.cfg = config or SignBehaviorConfig()
        self.state = State.MOVING

        print("[SignBehavior] initialised — detector ready")

        self._tag_buffer = {}       # type: Dict[int, int]
        self._saved_tag = None      # type: Optional[TagID]

        self._hold_counter = 0
        self._turn_counter = 0
        self._check_counter = 0

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False

        self._possible_turns = []   # type: List[str]
        self._chosen_turn = None    # type: Optional[str]

        self._slow_factor = 1.0
        self._slow_target = 0.0

        self._red_line_locked = False
        self._ignore_red_counter = 0

        self.debug = {}

        self._frame_rgb = None
        self._vehicle_detected = vehicle_detected

        self._left_offsets = []
        self._right_offsets = []

        self._active_sign_op = None
        self._ignored_red_print_cooldown = 0

    def reset(self):
        self.state = State.MOVING

        self._tag_buffer.clear()
        self._saved_tag = None

        self._hold_counter = 0
        self._turn_counter = 0
        self._check_counter = 0

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False

        self._possible_turns = []
        self._chosen_turn = None

        self._slow_factor = 1.0
        self._slow_target = 0.0

        self._red_line_locked = False
        self._ignore_red_counter = 0

        self._left_offsets.clear()
        self._right_offsets.clear()

        self._active_sign_op = None
        self._ignored_red_print_cooldown = 0

        self.debug = {}

        print("[SignBehavior] FSM reset")

    def step(self, frame_rgb, base_left, base_right, detections):
        self._frame_rgb = frame_rgb
        detections = detections or []

        tags = detect_tags(self, frame_rgb)
        confirmed = confirm_tags(self, tags)

        if self._ignore_red_counter > 0:
            self._ignore_red_counter -= 1
            red_line = False
        elif self._red_line_locked:
            red_line = False
        else:
            red_line = bool(detect_red_line(self, frame_rgb))

        left, right = self._fsm_step(
            confirmed,
            detections,
            red_line,
            base_left,
            base_right,
            frame_rgb,
        )

        self.debug = {
            "state": self.state.name,
            "confirmed_tags": [int(t) for t in confirmed],
            "saved_tag": int(self._saved_tag) if self._saved_tag is not None else None,
            "red_line": bool(red_line),
            "red_locked": bool(self._red_line_locked),
            "ignore_red_counter": int(self._ignore_red_counter),
            "possible_turns": list(self._possible_turns),
            "chosen_turn": self._chosen_turn,
            "slow_factor": round(float(self._slow_factor), 3),
            "hold_counter": int(self._hold_counter),
            "check_counter": int(self._check_counter),
            "turn_counter": int(self._turn_counter),
        }

        return left, right, self.state

    @property
    def state_name(self):
        return self.state.name

    def _save_confirmed_sign(self, confirmed_tags):
        # Prefer intersection signs over stop/yield if both are visible.
        for raw_tag_id in confirmed_tags:
            tag_id = resolve_tag(int(raw_tag_id))

            if tag_id is None:
                continue

            if tag_id in _TAG_TURNS:
                if self._saved_tag != tag_id:
                    print(
                        f"[SignBehavior] sign saved: intersection tag "
                        f"raw={raw_tag_id}, resolved={int(tag_id)} "
                        f"-> {_TAG_TURNS[tag_id]}"
                    )

                self._saved_tag = tag_id
                return

        for raw_tag_id in confirmed_tags:
            tag_id = resolve_tag(int(raw_tag_id))

            if tag_id is None:
                continue

            if tag_id in (TagID.STOP, TagID.YIELD):
                if self._saved_tag != tag_id:
                    label = "STOP" if tag_id == TagID.STOP else "YIELD"
                    print(
                        f"[SignBehavior] sign saved: {label} "
                        f"(raw={raw_tag_id}, resolved={int(tag_id)})"
                    )

                self._saved_tag = tag_id
                return

    def _fsm_step(self, confirmed_tags, detections, red_line, base_left, base_right, frame_rgb):
        # MOVING
        if self.state == State.MOVING:
            self._save_confirmed_sign(confirmed_tags)

            if red_line:
                tag = self._saved_tag

                # Important fix:
                # Red line alone must NOT trigger countdown/intersection behavior.
                # Otherwise false red-line detection causes move-stop-move-stop.
                if tag is None and not self.cfg.default_forward_on_red_without_tag:
                    if self._ignored_red_print_cooldown <= 0:
                        print("[SignBehavior] red line ignored — no saved sign/tag")
                        self._ignored_red_print_cooldown = 30
                    else:
                        self._ignored_red_print_cooldown -= 1

                    return base_left, base_right

                self._red_line_locked = True

                if tag in _TAG_TURNS:
                    self._possible_turns = list(_TAG_TURNS[tag])
                    self._chosen_turn = self._pick_turn()
                    self._active_sign_op = f"intersection turn '{self._chosen_turn}'"
                    self._turn_counter = 0
                    self._saved_tag = None
                    self.state = State.INTERSECT

                    print(
                        f"[SignBehavior] >>> INTERSECT — direction: {self._chosen_turn} "
                        f"(from tag {tag}, options {self._possible_turns})"
                    )

                    return base_left, base_right

                if tag == TagID.STOP or tag == TagID.YIELD:
                    label = "STOP" if tag == TagID.STOP else "YIELD"
                    self._active_sign_op = f"{label} sign"
                    self._slow_target = 0.0
                    self.state = State.SLOWING

                    print(
                        f"[SignBehavior] >>> {label} — red line close, braking to full stop "
                        f"(saved_tag={tag})"
                    )

                    return base_left, base_right

                # Optional fallback, normally disabled by config.
                self._chosen_turn = "forward"
                self._active_sign_op = "intersection turn 'forward' without tag"
                self._turn_counter = 0
                self._saved_tag = None
                self.state = State.INTERSECT

                print("[SignBehavior] >>> INTERSECT — default forward")
                return base_left, base_right

            # Sign detected but red line not close yet: slow approach.
            if self._saved_tag is not None:
                # target_factor = self.cfg.approach_speed_factor
                # ramp_factor = self.cfg.approach_ramp_factor

                # self._slow_factor = max(target_factor, self._slow_factor * ramp_factor)

                # return base_left * self._slow_factor, base_right * self._slow_factor
                return base_left, base_right

            self._slow_factor = 1.0
            return base_left, base_right

        # SLOWING
        if self.state == State.SLOWING:
            self._slow_factor *= self.cfg.slow_ramp_factor
            factor = max(self._slow_factor, self._slow_target)

            left = base_left * factor
            right = base_right * factor

            if self._slow_factor < self.cfg.stopped_speed_threshold:
                self._slow_factor = 0.0
                self._hold_counter = 0
                self._saved_tag = None
                self.state = State.STOPPED
                print("[SignBehavior] >>> STOPPED at red line")
                return 0.0, 0.0

            return left, right

        # STOPPED
        if self.state == State.STOPPED:
            self._hold_counter += 1

            if self._hold_counter < self.cfg.stop_hold_frames:
                return 0.0, 0.0

            self._check_counter = 0
            self._vehicle_seen_left = False
            self._vehicle_seen_right = False
            self._left_offsets.clear()
            self._right_offsets.clear()

            self.state = State.CHECKPATH
            print("[SignBehavior] >>> CHECKPATH — checking for vehicles")
            return 0.0, 0.0

        # CHECKPATH
        if self.state == State.CHECKPATH:
            return self._checkpath_step(detections)

        # POST_STOP
        if self.state == State.POST_STOP:
            if self._turn_counter < self.cfg.post_stop_frames:
                self._turn_counter += 1
                return self.cfg.post_stop_speed, self.cfg.post_stop_speed

            self._finish_behavior()
            return base_left, base_right

        # INTERSECT
        if self.state == State.INTERSECT:
            return self._intersect_step(base_left, base_right)

        # PRE_TURN
        if self.state == State.PRE_TURN:
            return self._preturn_step(base_left, base_right)

        # TURNING
        if self.state == State.TURNING:
            return self._turning_step(base_left, base_right)

        # EXITING
        if self.state == State.EXITING:
            return self._exiting_step(base_left, base_right)


        print("servoing")
        return base_left, base_right

    def _checkpath_step(self, detections):
        spd = self.cfg.check_turn_speed
        cl = self.cfg.check_left_frames
        cr = self.cfg.check_right_frames
        cs = self.cfg.check_settle_frames
        c = self._check_counter

        phase_a_end = cl
        phase_a_settle = cl + cs
        phase_b_end = phase_a_settle + cl + cr
        phase_b_settle = phase_b_end + cs
        phase_c_end = phase_b_settle + cl

        def is_stationary(offsets, threshold=0.05):
            if len(offsets) < 2:
                return True
            return (max(offsets) - min(offsets)) < threshold

        # Look left
        if c < phase_a_end:
            self._check_counter += 1
            return -spd, spd

        # Hold left and detect
        if c < phase_a_settle:
            if c > phase_a_end + 1:
                seen, offset = self._vehicle_detected(detections)
                if seen:
                    self._vehicle_seen_left = True
                    self._left_offsets.append(offset)

            self._check_counter += 1
            return 0.0, 0.0

        # Sweep to right
        if c < phase_b_end:
            self._check_counter += 1
            return spd, -spd

        # Hold right and detect
        if c < phase_b_settle:
            if c > phase_b_end + 1:
                seen, offset = self._vehicle_detected(detections)
                if seen:
                    self._vehicle_seen_right = True
                    self._right_offsets.append(offset)

            self._check_counter += 1
            return 0.0, 0.0

        # Re-center
        if c < phase_c_end:
            self._check_counter += 1
            return -spd, spd

        # Decision
        right_stationary = is_stationary(self._right_offsets)

        # Project rule:
        # Vehicle on right => wait.
        # Vehicle on left => we have priority.
        if self._vehicle_seen_right:
            if right_stationary:
                print("[SignBehavior] vehicle on right — yielding")
            else:
                print("[SignBehavior] moving vehicle on right — yielding")

            self._check_counter = 0
            self._vehicle_seen_left = False
            self._vehicle_seen_right = False
            self._left_offsets.clear()
            self._right_offsets.clear()
            return 0.0, 0.0

        if self._vehicle_seen_left:
            print("[SignBehavior] vehicle on left — priority, continuing")

        self._turn_counter = 0
        self.state = State.POST_STOP

        self._vehicle_seen_left = False
        self._vehicle_seen_right = False
        self._left_offsets.clear()
        self._right_offsets.clear()

        print("[SignBehavior] >>> path clear — POST_STOP")
        return 0.0, 0.0

    def _intersect_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"
        self._turn_counter = 0

        if turn in ("left", "right"):
            self.state = State.PRE_TURN
            print(f"[SignBehavior] >>> PRE_TURN for '{turn}'")
            return self.cfg.preturn_speed, self.cfg.preturn_speed

        self.state = State.TURNING
        print("[SignBehavior] >>> TURNING 'forward'")
        return self.cfg.intersect_forward_speed

    def _preturn_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"

        if turn == "right":
            total = self.cfg.preturn_right_frames
        else:
            total = self.cfg.preturn_left_frames

        if self._turn_counter < total:
            self._turn_counter += 1
            return self.cfg.preturn_speed, self.cfg.preturn_speed

        self._turn_counter = 0
        self.state = State.TURNING

        print(f"[SignBehavior] >>> PRE_TURN done — TURNING '{turn}'")
        return 0.0, 0.0

    def _turning_step(self, base_left, base_right):
        turn = self._chosen_turn or "forward"

        speeds_map = {
            "forward": self.cfg.intersect_forward_speed,
            "left": self.cfg.intersect_left_speed,
            "right": self.cfg.intersect_right_speed,
        }

        frames_map = {
            "forward": self.cfg.intersect_forward_frames,
            "left": self.cfg.intersect_left_frames,
            "right": self.cfg.intersect_right_frames,
        }

        total = frames_map.get(turn, self.cfg.intersect_forward_frames)

        if self._turn_counter < total:
            self._turn_counter += 1
            return speeds_map.get(turn, self.cfg.intersect_forward_speed)

        self._turn_counter = 0
        self.state = State.EXITING

        print(f"[SignBehavior] >>> TURNING '{turn}' done — EXITING")
        return base_left, base_right

    def _exiting_step(self, base_left, base_right):
        self._turn_counter += 1

        if self._turn_counter >= self.cfg.exit_timeout_frames:
            self._finish_behavior()
            return base_left, base_right

        return self.cfg.exit_speed, self.cfg.exit_speed

    def _finish_behavior(self):
        print(f"[SignBehavior] COMPLETE: {self._active_sign_op} — robot resumed driving")

        self._red_line_locked = False
        self._ignore_red_counter = self.cfg.red_ignore_after_frames

        self._possible_turns = []
        self._chosen_turn = None
        self._saved_tag = None
        self._active_sign_op = None

        self._slow_factor = 1.0
        self._turn_counter = 0
        self._hold_counter = 0
        self._check_counter = 0

        self.state = State.MOVING

    def _pick_turn(self):
        return random.choice(self._possible_turns) if self._possible_turns else "forward"
