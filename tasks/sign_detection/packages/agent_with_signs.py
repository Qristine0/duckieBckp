"""
agent_with_signs.py
===================
Drop-in replacement for LaneServoingAgent that adds AprilTag-based sign
behaviour (stop signs, intersection navigation, path-clearing checks).
"""

from typing import List, Optional, Tuple

import numpy as np

from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent
from tasks.sign_detection.packages.sign_behavior import SignBehaviorFSM, SignBehaviorConfig


class LaneServoingAgentWithSigns(LaneServoingAgent):
    """
    Extends LaneServoingAgent with AprilTag sign detection and intersection FSM.

    The only public API change is:

        compute_commands(image, detections=None)

    Pass in the list of object-detection hits so the path-check sweep can
    watch for oncoming vehicles.  If you leave it as None the sweep will
    never abort (conservative: wait the full sweep time regardless).

    Everything else (p_gain, d_gain, step(), get_debug_info() ...) is
    inherited unchanged.
    """

    def __init__(self, config_path=None, sign_config=None):
        super().__init__(config_path=config_path)

        cfg = sign_config or SignBehaviorConfig()
        self._sign_fsm = SignBehaviorFSM(config=cfg)

    def reset_behavior(self):
        if hasattr(self, "_sign_fsm") and self._sign_fsm is not None:
            self._sign_fsm.reset()

        self._prev_error = 0.0
        self._filtered_error = 0.0

        if hasattr(self, "_filtered_lateral_error"):
            self._filtered_lateral_error = 0.0

        if hasattr(self, "_filtered_heading_error"):
            self._filtered_heading_error = 0.0

        if hasattr(self, "_filtered_steering"):
            self._filtered_steering = 0.0

        if hasattr(self, "_lost_frames"):
            self._lost_frames = 0

        print("[LaneAgent] behavior reset")

    # ------------------------------------------------------------------
    # Override compute_commands
    # ------------------------------------------------------------------

    def compute_commands(self, image, detections=None):
        # type: (np.ndarray, Optional[List]) -> Tuple[float, float]
        """
        Parameters
        ----------
        image      : RGB frame (H x W x 3)
        detections : output of ObjectDetectionAgent.detect()
                     list of ((x1,y1,x2,y2), score, class_id)  or None
        """
        # 1. Lane-following commands from parent
        base_left, base_right = super().compute_commands(image)

        # 2. Sign FSM may override them
        left, right = self._sign_fsm.step(
            image, base_left, base_right, detections or []
        )

        return left, right

    @property
    def sign_state(self):
        # type: () -> str
        return self._sign_fsm.state_name

    @property
    def sign_debug(self):
        # type: () -> dict
        return self._sign_fsm.debug

    # ------------------------------------------------------------------
    # Override step() so wheels_driver callers also get sign behaviour
    # ------------------------------------------------------------------

    def step(self, image, wheels_driver, detections=None):
        # type: (np.ndarray, object, Optional[List]) -> Tuple[float, float]
        left, right = self.compute_commands(image, detections)
        wheels_driver.set_wheels_speed(left, right)
        return left, right