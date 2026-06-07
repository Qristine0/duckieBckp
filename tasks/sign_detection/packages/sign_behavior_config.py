from enum import IntEnum, auto
from typing import Dict, List


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39


_TAG_ID_MAP: Dict[int, TagID] = {
    # Your simulation/real sign sometimes reports STOP as raw tag 1.
    1: TagID.STOP,

    9: TagID.TURN_RIGHT_FWD,
    10: TagID.TURN_LEFT_FWD,
    11: TagID.TURN_LEFT_RIGHT,

    20: TagID.STOP,
    24: TagID.STOP,
    25: TagID.STOP,
    26: TagID.STOP,

    39: TagID.YIELD,
}


def resolve_tag(raw_id):
    return _TAG_ID_MAP.get(int(raw_id))


_TAG_TURNS = {
    TagID.TURN_LEFT_RIGHT: ["left", "right"],
    TagID.TURN_LEFT_FWD: ["left", "forward"],
    TagID.TURN_RIGHT_FWD: ["right", "forward"],
}  # type: Dict[int, List[str]]


class State(IntEnum):
    MOVING = auto()
    SLOWING = auto()
    YIELD_CREEP = auto()
    STOPPED = auto()
    CHECKPATH = auto()
    POST_STOP = auto()
    INTERSECT = auto()
    PRE_TURN = auto()
    TURNING = auto()
    EXITING = auto()


class SignBehaviorConfig:
    """
    Config object for sign behavior.

    Important:
    This class accepts keyword overrides, for example:

        SignBehaviorConfig(stop_hold_frames=10, min_margin=10.0)

    This fixes the real-bot crash:
        TypeError: __init__() got an unexpected keyword argument 'stop_hold_frames'
    """

    def __init__(self, **overrides):
        # Red-line detection
        self.red_strip_frac = 0.32
        self.red_pixel_frac = 0.025
        self.red_line_close_y2_ratio = 0.82

        self.red_hsv_low1 = (0, 120, 80)
        self.red_hsv_high1 = (10, 255, 255)
        self.red_hsv_low2 = (170, 120, 80)
        self.red_hsv_high2 = (180, 255, 255)

        # Sign approach behavior
        self.approach_speed_factor = 0.72
        self.approach_ramp_factor = 0.98

        # Stop-line hold
        self.stop_hold_frames = 0

        # Some real_server versions pass this.
        # Keep it here so the robot does not crash.
        self.min_margin = 10.0

        # Speed ramp
        self.slow_ramp_factor = 0.84
        self.stopped_speed_threshold = 0.06

        # YIELD specific
        self.yield_min_speed = 0.10
        self.yield_creep_frames = 5

        # CHECKPATH sweep
        self.check_left_frames = 4
        self.check_right_frames = 4
        self.check_turn_speed = 0.04
        self.check_settle_frames = 5

        # POST_STOP
        self.post_stop_frames = 25
        self.post_stop_speed = 0.20

        # Pre-turn forward creep
        self.preturn_right_frames = 11
        self.preturn_left_frames = 20
        self.preturn_speed = 0.20

        # Intersection manoeuvres
        self.intersect_forward_frames = 28
        self.intersect_left_frames = 18
        self.intersect_right_frames = 18

        self.intersect_forward_speed = (0.20, 0.20)
        self.intersect_left_speed = (0.15, 0.22)
        self.intersect_right_speed = (0.22, 0.15)

        # Exiting
        self.exit_speed = 0.20
        self.exit_timeout_frames = 5

        # Accept anything passed by real_server.py.
        # This prevents future keyword-argument crashes.
        for key, value in overrides.items():
            setattr(self, key, value) 