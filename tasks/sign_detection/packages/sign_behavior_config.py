from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple


class TagID(IntEnum):
    TURN_RIGHT_FWD = 9
    TURN_LEFT_FWD = 10
    TURN_LEFT_RIGHT = 11
    STOP = 26
    YIELD = 39


_TAG_ID_MAP: Dict[int, TagID] = {
    # In your simulation/logs, some STOP signs are detected as raw tag 1.
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


def resolve_tag(raw_id: int) -> Optional[TagID]:
    return _TAG_ID_MAP.get(raw_id)


_TAG_TURNS: Dict[TagID, List[str]] = {
    TagID.TURN_LEFT_RIGHT: ["left", "right"],
    TagID.TURN_LEFT_FWD: ["left", "forward"],
    TagID.TURN_RIGHT_FWD: ["right", "forward"],
}


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
    Config for sign behavior.

    Important:
    This class intentionally accepts **kwargs so old server code like

        SignBehaviorConfig(stop_hold_frames=0, min_margin=10.0)

    will not crash.
    """

    def __init__(self, **kwargs):
        # Red-line detection
        self.red_strip_frac: float = kwargs.pop("red_strip_frac", 0.32)
        self.red_pixel_frac: float = kwargs.pop("red_pixel_frac", 0.025)

        # Bigger = stop later / closer to red line.
        # Smaller = stop earlier / farther from red line.
        self.red_line_close_y2_ratio: float = kwargs.pop("red_line_close_y2_ratio", 0.82)

        self.red_hsv_low1: Tuple[int, int, int] = kwargs.pop("red_hsv_low1", (0, 120, 80))
        self.red_hsv_high1: Tuple[int, int, int] = kwargs.pop("red_hsv_high1", (10, 255, 255))
        self.red_hsv_low2: Tuple[int, int, int] = kwargs.pop("red_hsv_low2", (170, 120, 80))
        self.red_hsv_high2: Tuple[int, int, int] = kwargs.pop("red_hsv_high2", (180, 255, 255))

        # After sign is detected, keep driving but slower while approaching red line.
        self.approach_speed_factor: float = kwargs.pop("approach_speed_factor", 0.72)
        self.approach_ramp_factor: float = kwargs.pop("approach_ramp_factor", 0.98)

        # Stop-line hold
        self.stop_hold_frames: int = kwargs.pop("stop_hold_frames", 0)

        # Braking after close red line is detected
        self.slow_ramp_factor: float = kwargs.pop("slow_ramp_factor", 0.84)
        self.stopped_speed_threshold: float = kwargs.pop("stopped_speed_threshold", 0.06)

        # YIELD specific
        self.yield_min_speed: float = kwargs.pop("yield_min_speed", 0.10)
        self.yield_creep_frames: int = kwargs.pop("yield_creep_frames", 5)

        # CHECKPATH sweep
        self.check_left_frames: int = kwargs.pop("check_left_frames", 4)
        self.check_right_frames: int = kwargs.pop("check_right_frames", 4)
        self.check_turn_speed: float = kwargs.pop("check_turn_speed", 0.04)
        self.check_settle_frames: int = kwargs.pop("check_settle_frames", 5)

        # POST_STOP
        self.post_stop_frames: int = kwargs.pop("post_stop_frames", 25)
        self.post_stop_speed: float = kwargs.pop("post_stop_speed", 0.20)

        # Pre-turn forward creep
        self.preturn_right_frames: int = kwargs.pop("preturn_right_frames", 11)
        self.preturn_left_frames: int = kwargs.pop("preturn_left_frames", 20)
        self.preturn_speed: float = kwargs.pop("preturn_speed", 0.20)

        # Intersection manoeuvres
        self.intersect_forward_frames: int = kwargs.pop("intersect_forward_frames", 28)
        self.intersect_left_frames: int = kwargs.pop("intersect_left_frames", 18)
        self.intersect_right_frames: int = kwargs.pop("intersect_right_frames", 18)

        self.intersect_forward_speed: Tuple[float, float] = kwargs.pop(
            "intersect_forward_speed", (0.20, 0.20)
        )
        self.intersect_left_speed: Tuple[float, float] = kwargs.pop(
            "intersect_left_speed", (0.15, 0.22)
        )
        self.intersect_right_speed: Tuple[float, float] = kwargs.pop(
            "intersect_right_speed", (0.22, 0.15)
        )

        # Exiting
        self.exit_speed: float = kwargs.pop("exit_speed", 0.20)
        self.exit_timeout_frames: int = kwargs.pop("exit_timeout_frames", 5)

        # Compatibility:
        # If old real_server.py passes extra values like min_margin,
        # keep them instead of crashing.
        for key, value in kwargs.items():
            setattr(self, key, value)