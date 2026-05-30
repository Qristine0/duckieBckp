"""
agent_with_signs.py
===================
Drop-in replacement for LaneServoingAgent that adds AprilTag-based sign
behaviour (stop signs, intersection navigation, path-clearing checks).

Drop this file next to the existing agent.py and import
    from tasks.visual_lane_servoing.packages.agent_with_signs import LaneServoingAgentWithSigns
anywhere you used LaneServoingAgent.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ── existing agent (unchanged) ────────────────────────────────────────────
from tasks.visual_lane_servoing.packages.agent import LaneServoingAgent

# ── new sign FSM ──────────────────────────────────────────────────────────
from tasks.sign_detection.packages.sign_behavior import SignBehaviorFSM, SignBehaviorConfig, draw_sign_debug



class LaneServoingAgentWithSigns(LaneServoingAgent):
    """
    Extends LaneServoingAgent with AprilTag sign detection and intersection FSM.

    The only public API change is:

        compute_commands(image, detections=None)

    Pass in the list of object-detection hits so the path-check sweep can
    watch for oncoming vehicles.  If you leave it as None the sweep will
    never abort (conservative: wait the full sweep time regardless).

    Everything else (p_gain, d_gain, step(), get_debug_info() …) is
    inherited unchanged.
    """

    def __init__(self, config_path: str = None, sign_config: SignBehaviorConfig = None):
        super().__init__(config_path=config_path)
        self._sign_fsm = SignBehaviorFSM(config=sign_config or SignBehaviorConfig())

    # ------------------------------------------------------------------
    # Override compute_commands
    # ------------------------------------------------------------------

    def compute_commands(          # type: ignore[override]
        self,
        image:      np.ndarray,
        detections: Optional[List] = None,
    ) -> Tuple[float, float]:
        """
        Parameters
        ----------
        image      : RGB frame (H×W×3)
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

    # Convenience passthrough so the server can log the FSM state
    @property
    def sign_state(self) -> str:
        return self._sign_fsm.state_name

    @property
    def sign_debug(self) -> dict:
        return self._sign_fsm.debug

    # ------------------------------------------------------------------
    # Override step() so wheels_driver callers also get sign behaviour
    # ------------------------------------------------------------------

    def step(                      # type: ignore[override]
        self,
        image:        np.ndarray,
        wheels_driver,
        detections:   Optional[List] = None,
    ) -> Tuple[float, float]:
        left, right = self.compute_commands(image, detections)
        wheels_driver.set_wheels_speed(left, right)
        return left, right
    
    
    